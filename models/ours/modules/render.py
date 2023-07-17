#  Copyright (c) 3.2022. Yinyu Nie
#  License: MIT
import torch
import torch.nn as nn
import numpy as np
from pytorch3d.structures import join_meshes_as_scene, join_meshes_as_batch
from pytorch3d.renderer import (
    PerspectiveCameras,
    RasterizationSettings,
    SoftSilhouetteShader,
    MeshRasterizer,
    MeshRendererWithFragments)
from external.fast_transformers.fast_transformers.masking import LengthMask
from models.registers import MODULES
from net_utils.box_utils import get_box_corners, project_points_to_2d


def move_meshes_to_box3ds(meshes, centers, sizes):
    vertices = meshes.verts_padded()
    n_vertices = vertices.size(1)

    centers = centers.flatten(0, 1)[:, None]
    centers = centers.expand(-1, n_vertices, -1).contiguous()

    sizes = sizes.flatten(0, 1)[:, None]
    sizes = sizes.expand(-1, n_vertices, -1).contiguous()
    sizes = sizes / 2.

    vertices = vertices * sizes + centers

    meshes = meshes.update_padded(new_verts_padded=vertices)
    return meshes


@MODULES.register_module
class Proj2Img(nn.Module):
    def __init__(self, cfg, optim_spec=None, device='cuda'):
        '''
        Proj 3D boxes to 2D image plane.
        :param cfg: configuration file.
        :param optim_spec: optimizer parameters.
        '''
        super(Proj2Img, self).__init__()
        '''Optimizer parameters used in training'''
        self.optim_spec = optim_spec
        self.device = device
        image_sizes = cfg.image_size
        self.cfg = cfg
        split = cfg.config.test.finetune_split if cfg.config.mode == 'test' else cfg.config.mode
        n_views = cfg.config.data.n_views * cfg.config[split].batch_size // cfg.config.distributed.num_gpus
        downsample_ratio = cfg.config.data.downsample_ratio

        '''set renderer'''
        sigma = 1e-4
        raster_settings = RasterizationSettings(
            image_size=(int(image_sizes[1]) // downsample_ratio, int(image_sizes[0]) // downsample_ratio),
            blur_radius=np.log(1. / 1e-4 - 1.) * sigma,
            faces_per_pixel=50
        )
        rasterizer = MeshRasterizer(
            raster_settings=raster_settings
        )
        shader = SoftSilhouetteShader()

        # initialize cameras
        self.cameras = PerspectiveCameras(
            focal_length=torch.zeros(n_views, 2, device=device),
            principal_point=torch.zeros(n_views, 2, device=device),
            image_size=torch.ones(n_views, 2, device=device),
            R=torch.zeros(n_views, 3, 3, device=device),
            T=torch.zeros(n_views, 3, device=device),
            in_ndc=False,
            device=device)

        self.renderer = MeshRendererWithFragments(
            rasterizer=rasterizer,
            shader=shader)
        self.vertices_per_template = 2562
        self.faces_per_template = 5120

    def render_bbox(self, centers, sizes, cam_Ts, cam_Ks, image_sizes):
        # get box and project
        vectors = torch.diag_embed(sizes / 2.)
        box_corners = get_box_corners(centers, vectors)
        # get projected 2d bboxes
        # n_batch x n_views x n_objects x 8 x 2
        proj_box2ds, in_frustum = project_points_to_2d(box_corners, cam_Ks, cam_Ts, eps=1e-6)

        in_frustum = in_frustum.max(dim=-1)[0]

        # clamp to image sizes
        x1y1x2y2 = torch.clamp(torch.min(proj_box2ds, image_sizes[:, :, None, None] - 1), min=0)
        x1y1x2y2 = torch.cat([torch.min(x1y1x2y2, dim=3)[0], torch.max(x1y1x2y2, dim=3)[0]], dim=-1)

        return x1y1x2y2, in_frustum

    def render_instances(self, meshes, cam_Ts, cam_Ks, image_sizes, render_mask_tr=None):
        n_batch, n_view = cam_Ts.shape[:2]
        image_sizes = image_sizes[..., [1, 0]]
        n_object = len(meshes) // n_batch
        # merge meshes into scenes by batch
        scene_splits = torch.arange(n_batch * n_object).split(n_object)
        # n_scene x n_object
        scenes = [join_meshes_as_scene([meshes[idx.item()] for idx in scene_split]) for scene_split in scene_splits]
        # n_scene x n_view x n_object
        scenes_ext = join_meshes_as_batch([scene.extend(n_view) for scene in scenes])

        # set cameras (n_scene x n_view x dim)
        fcl_screen = torch.cat([cam_Ks[:, :, 0, [0]], cam_Ks[:, :, 1, [1]]], dim=-1).view(n_batch * n_view, 2)  # n_scene * n_view x 2
        prp_screen = cam_Ks[..., :2, 2].view(n_batch * n_view, 2)  # n_scene * n_view x 2
        image_sizes = image_sizes.view(n_batch * n_view, 2)  # n_scene * n_view x 2
        cam_Ts = cam_Ts.view(n_batch * n_view, 4, 4)  # n_scene * n_view x 4 x 4

        Rs = cam_Ts[:, :3, :3]
        # transform to pytorch3d camera system
        Rs[..., 0] *= -1
        Rs[..., 2] *= -1  # n_scene * n_view x 3 x 3

        cam_loc = cam_Ts[:, :3, 3]
        Ts = -torch.bmm(Rs.transpose(1, 2), cam_loc[:, :, None])[:, :, 0]  # n_scene * n_view x 3

        silhouettes, fragments = self.renderer(
            scenes_ext,
            cameras=self.cameras,
            focal_length=fcl_screen,
            principal_point=prp_screen,
            image_size=image_sizes,
            R=Rs,
            T=Ts,
            eps=1e-3)

        silhouettes = silhouettes[..., 3]
        silhouettes = silhouettes.view(n_batch, n_view, *silhouettes.shape[1:])

        '''Get instance masks'''
        instance_labels = fragments.pix_to_face[..., 0]

        if render_mask_tr is not None:
            render_mask_tr = torch.logical_not(render_mask_tr.flatten(0, 1))
            instance_labels = instance_labels.masked_fill(render_mask_tr, -1)

        # scene_ids = torch.div(instance_labels, (n_view * n_object * self.faces_per_template), rounding_mode='floor')
        remaining = torch.remainder(instance_labels, (n_view * n_object * self.faces_per_template))
        # view_ids = torch.div(remaining, (n_object * self.faces_per_template), rounding_mode='floor')
        remaining = torch.remainder(remaining, (n_object * self.faces_per_template))
        obj_ids = torch.div(remaining, (self.faces_per_template), rounding_mode='floor')
        # face_ids = torch.remainder(remaining, (self.faces_per_template))

        obj_ids[instance_labels < 0] = -1
        obj_ids = obj_ids.view(n_batch, n_view, *obj_ids.shape[1:])

        return silhouettes, obj_ids

    @staticmethod
    def project_points(meshes, cam_Ts, cam_Ks, image_sizes):
        n_batch, n_view = cam_Ts.shape[:2]
        n_object = len(meshes) // n_batch

        # get 3d points and project
        points_on_meshes = meshes.verts_padded()
        points_on_meshes = points_on_meshes.view(n_batch, n_object, -1, 3)

        proj_points, in_frustum = project_points_to_2d(points_on_meshes, cam_Ks, cam_Ts)

        in_frustum = in_frustum.max(dim=-1)[0]

        # clamp to image sizes
        proj_points = torch.clamp(torch.min(proj_points, image_sizes[:, :, None, None] - 1), min=0)

        return proj_points, in_frustum, points_on_meshes

    def generate(self, box3ds, meshes, cam_Ts, cam_Ks, image_sizes, render_mask_tr, start_deform=False, **kwargs):
        '''see forward'''
        n_batches = len(box3ds)
        outputs = []
        for b_id in range(n_batches):
            centers = box3ds[b_id][..., :3]
            sizes = box3ds[b_id][..., 3:6]
            classes = box3ds[b_id][..., 6:]

            # move meshes to box3ds
            posed_meshes = move_meshes_to_box3ds(meshes[b_id], centers, sizes)

            # from pytorch3d.vis.plotly_vis import plot_batch_individually, plot_scene
            # fig = plot_scene({
            #     "subplot1": {"mesh%d"%(i):posed_meshes[i] for i in range(len(posed_meshes))}
            # })
            # fig.show()

            # render points on meshes to points on 2D
            points_2d, in_frustum, points_on_meshes = self.project_points(posed_meshes, cam_Ts[[b_id]].clone(), cam_Ks[[b_id]].clone(),
                                                                          image_sizes[[b_id]].clone())

            '''render meshes to silhouettes'''
            if start_deform:
                render_mask = render_mask_tr[[b_id]] if render_mask_tr is not None else None
                silhouettes, obj_ids = self.render_instances(posed_meshes, cam_Ts[[b_id]].clone(), cam_Ks[[b_id]].clone(),
                                                             image_sizes[[b_id]].clone(),
                                                             render_mask)
            else:
                silhouettes = None
                obj_ids = None

            outputs.append(
                {'box3ds': box3ds[b_id],
                 'silhouettes': silhouettes,
                 'obj_ids': obj_ids,
                 'points_2d':points_2d,
                 'posed_meshes': posed_meshes})

        return outputs

    def forward(self, box3ds, meshes, cam_Ts, cam_Ks, image_sizes, render_mask_tr, start_deform=False,
                    pred_gt_matching=None, pred_mask=None, **kwargs):
        '''
        Render generated boxes given cam params
        :param box3ds: n_batch x n_box x box_dim
        :param meshes: (n_batch * n_view) x pytorch3d mesh
        :param cam_Ts: n_batch x n_view x 4 x 4
        :param cam_Ks: n_batch x n_view x 3 x 3
        :param render_mask_tr: n_batch x n_view x im_height x im_width
        :param image_sizes: n_batch x n_view x 2
        :return:
        '''
        centers = box3ds[..., :3]
        sizes = box3ds[..., 3:6]
        classes_completeness = box3ds[..., 6:]

        if self.cfg.config.mode == 'train':
            if pred_mask is not None:
                pred_mask = torch.logical_not(LengthMask(pred_mask).bool_matrix)
                pred_mask = pred_mask[:, :, None].expand(-1, -1, 3)
                sizes.masked_fill_(pred_mask, 0.)
                centers.masked_fill_(pred_mask, 0.)
        elif (self.cfg.config.mode == 'demo' and self.cfg.config.data.n_views == 1) or (
                    self.cfg.config.mode == 'test' and self.cfg.config.test.n_views_for_finetune == 1):
            if pred_gt_matching is not None:
                masks = torch.ones_like(sizes, dtype=torch.bool)
                for batch_id, pair_batch in enumerate(pred_gt_matching):
                    masks[batch_id][pair_batch[0]] = False
                sizes.masked_fill_(masks, 0.)
                centers.masked_fill_(masks, 0.)

        # move meshes to box3ds
        meshes = move_meshes_to_box3ds(meshes, centers, sizes)

        # from pytorch3d.vis.plotly_vis import plot_batch_individually, plot_scene
        # fig = plot_scene({
        #     "subplot1": {"mesh%d"%(i):meshes[i] for i in range(len(meshes))}
        # })
        # fig.show()

        # render points on meshes to points on 2D
        points_2d, in_frustum, points_on_meshes = self.project_points(meshes, cam_Ts.clone(), cam_Ks.clone(), image_sizes.clone())

        '''render meshes to silhouettes'''
        if start_deform:
            silhouettes, obj_ids = self.render_instances(meshes, cam_Ts.clone(), cam_Ks.clone(), image_sizes.clone(),
                                                         render_mask_tr)
        else:
            silhouettes = None
            obj_ids = None

        return {'points_2d': points_2d,
                'points_3d': points_on_meshes,
                'in_frustum': in_frustum,
                'classes_completeness': classes_completeness,
                'silhouettes': silhouettes,
                'obj_ids': obj_ids}