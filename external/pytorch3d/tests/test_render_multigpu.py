# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.nn as nn
from pytorch3d.renderer import (
    AlphaCompositor,
    BlendParams,
    HardGouraudShader,
    Materials,
    MeshRasterizer,
    MeshRenderer,
    PointLights,
    PointsRasterizationSettings,
    PointsRasterizer,
    PointsRenderer,
    RasterizationSettings,
    SoftPhongShader,
    TexturesVertex,
)
from pytorch3d.renderer.cameras import FoVPerspectiveCameras, look_at_view_transform
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.utils.ico_sphere import ico_sphere

from .common_testing import get_random_cuda_device, TestCaseMixin


# Set the number of GPUS you want to test with
NUM_GPUS = 3
GPU_LIST = list({get_random_cuda_device() for _ in range(NUM_GPUS)})
print("GPUs: %s" % ", ".join(GPU_LIST))


class TestRenderMeshesMultiGPU(TestCaseMixin, unittest.TestCase):
    def _check_mesh_renderer_props_on_device(self, renderer, device):
        """
        Helper function to check that all the properties of the mesh
        renderer have been moved to the correct device.
        """
        # Cameras
        self.assertEqual(renderer.rasterizer.cameras.device, device)
        self.assertEqual(renderer.rasterizer.cameras.R.device, device)
        self.assertEqual(renderer.rasterizer.cameras.T.device, device)
        self.assertEqual(renderer.shader.cameras.device, device)
        self.assertEqual(renderer.shader.cameras.R.device, device)
        self.assertEqual(renderer.shader.cameras.T.device, device)

        # Lights and Materials
        self.assertEqual(renderer.shader.lights.device, device)
        self.assertEqual(renderer.shader.lights.ambient_color.device, device)
        self.assertEqual(renderer.shader.materials.device, device)
        self.assertEqual(renderer.shader.materials.ambient_color.device, device)

    def test_mesh_renderer_to(self):
        """
        Test moving all the tensors in the mesh renderer to a new device.
        """

        device1 = torch.device("cpu")

        R, T = look_at_view_transform(1500, 0.0, 0.0)

        # Init shader settings
        materials = Materials(device=device1)
        lights = PointLights(device=device1)
        lights.location = torch.tensor([0.0, 0.0, +1000.0], device=device1)[None]

        raster_settings = RasterizationSettings(
            image_size=256, blur_radius=0.0, faces_per_pixel=1
        )
        cameras = FoVPerspectiveCameras(
            device=device1, R=R, T=T, aspect_ratio=1.0, fov=60.0, zfar=100
        )
        rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)

        blend_params = BlendParams(
            1e-4,
            1e-4,
            background_color=torch.zeros(3, dtype=torch.float32, device=device1),
        )

        shader = SoftPhongShader(
            lights=lights,
            cameras=cameras,
            materials=materials,
            blend_params=blend_params,
        )
        renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)

        mesh = ico_sphere(2, device1)
        verts_padded = mesh.verts_padded()
        textures = TexturesVertex(
            verts_features=torch.ones_like(verts_padded, device=device1)
        )
        mesh.textures = textures
        self._check_mesh_renderer_props_on_device(renderer, device1)

        # Test rendering on cpu
        output_images = renderer(mesh)
        self.assertEqual(output_images.device, device1)

        # Move renderer and mesh to another device and re render
        # This also tests that background_color is correctly moved to
        # the new device
        device2 = torch.device("cuda:0")
        renderer = renderer.to(device2)
        mesh = mesh.to(device2)
        self._check_mesh_renderer_props_on_device(renderer, device2)
        output_images = renderer(mesh)
        self.assertEqual(output_images.device, device2)

    def test_render_meshes(self):
        test = self

        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                mesh = ico_sphere(3)
                self.register_buffer("faces", mesh.faces_padded())
                self.renderer = self.init_render()

            def init_render(self):

                cameras = FoVPerspectiveCameras()
                raster_settings = RasterizationSettings(
                    image_size=128, blur_radius=0.0, faces_per_pixel=1
                )
                lights = PointLights(
                    ambient_color=((1.0, 1.0, 1.0),),
                    diffuse_color=((0, 0.0, 0),),
                    specular_color=((0.0, 0, 0),),
                    location=((0.0, 0.0, 1e5),),
                )
                renderer = MeshRenderer(
                    rasterizer=MeshRasterizer(
                        cameras=cameras, raster_settings=raster_settings
                    ),
                    shader=HardGouraudShader(cameras=cameras, lights=lights),
                )
                return renderer

            def forward(self, verts, texs):
                batch_size = verts.size(0)
                self.renderer = self.renderer.to(verts.device)
                tex = TexturesVertex(verts_features=texs)
                faces = self.faces.expand(batch_size, -1, -1).to(verts.device)
                mesh = Meshes(verts, faces, tex).to(verts.device)

                test._check_mesh_renderer_props_on_device(self.renderer, verts.device)
                img_render = self.renderer(mesh)
                return img_render[:, :, :, :3]

        # DataParallel requires every input tensor be provided
        # on the first device in its device_ids list.
        verts = ico_sphere(3).verts_padded()
        texs = verts.new_ones(verts.shape)
        model = Model()
        model.to(GPU_LIST[0])
        model = nn.DataParallel(model, device_ids=GPU_LIST)

        # Test a few iterations
        for _ in range(100):
            model(verts, texs)


class TestRenderPointssMultiGPU(TestCaseMixin, unittest.TestCase):
    def _check_points_renderer_props_on_device(self, renderer, device):
        """
        Helper function to check that all the properties have
        been moved to the correct device.
        """
        # Cameras
        self.assertEqual(renderer.rasterizer.cameras.device, device)
        self.assertEqual(renderer.rasterizer.cameras.R.device, device)
        self.assertEqual(renderer.rasterizer.cameras.T.device, device)

    def test_points_renderer_to(self):
        """
        Test moving all the tensors in the points renderer to a new device.
        """

        device1 = torch.device("cpu")

        R, T = look_at_view_transform(1500, 0.0, 0.0)

        raster_settings = PointsRasterizationSettings(
            image_size=256, radius=0.001, points_per_pixel=1
        )
        cameras = FoVPerspectiveCameras(
            device=device1, R=R, T=T, aspect_ratio=1.0, fov=60.0, zfar=100
        )
        rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)

        renderer = PointsRenderer(rasterizer=rasterizer, compositor=AlphaCompositor())

        mesh = ico_sphere(2, device1)
        verts_padded = mesh.verts_padded()
        pointclouds = Pointclouds(
            points=verts_padded, features=torch.randn_like(verts_padded)
        )
        self._check_points_renderer_props_on_device(renderer, device1)

        # Test rendering on cpu
        output_images = renderer(pointclouds)
        self.assertEqual(output_images.device, device1)

        # Move renderer and pointclouds to another device and re render
        device2 = torch.device("cuda:0")
        renderer = renderer.to(device2)
        pointclouds = pointclouds.to(device2)
        self._check_points_renderer_props_on_device(renderer, device2)
        output_images = renderer(pointclouds)
        self.assertEqual(output_images.device, device2)
