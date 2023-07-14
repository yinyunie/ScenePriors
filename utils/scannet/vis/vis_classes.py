#  Copyright (c) 7.2022. Yinyu Nie
#  License: MIT
import vtk
import numpy as np
import seaborn as sns
from vtk.util.numpy_support import numpy_to_vtk
from PIL import Image, ImageDraw, ImageFont
from typing import List, Union

from utils.vis_base import VIS_BASE
from utils.tools import binary_mask_to_polygon
from utils.threed_front.tools.threed_front_scene import rotation_matrix

golden = (1 + 5 ** 0.5) / 2


class VIS_ScanNet(VIS_BASE):
    '''
    visualization class for scannet frames.
    '''
    def __init__(self, cam_Ks, cam_Ts, scene_dir, scan_data, class_names):
        super(VIS_ScanNet, self).__init__()
        self.cam_Ks = cam_Ks
        self.cam_Ts = cam_Ts
        self._view_id = 0
        self.update_view(view_id=self.view_id)

        scene_name = scene_dir.name
        self._mesh_file = scene_dir.joinpath(scene_name + '_vh_clean_2.ply')

        self.vertices = scan_data['vertices']
        self.semantic_labels = scan_data['semantic_labels']
        self.instance_labels = scan_data['instance_labels']
        self.instance_bboxes = scan_data['instance_bboxes']
        self.instance2semantic = scan_data['instance2semantic']

        self.class_names = [class_names[int(class_id)] for class_id in self.instance_bboxes[:, -1]]
        self.class_ids = self.instance_bboxes[:, -1].astype(np.int32)
        self.cls_palette = np.array([(0., 0., 0.), *sns.color_palette("hls", len(class_names)-1)])
        self.inst_palette = np.array([(0., 0., 0.), *sns.color_palette("hls", max(self.instance_labels))])

    @property
    def mesh_file(self):
        return self._mesh_file

    @property
    def view_id(self):
        return self._view_id

    def update_view(self, view_id):
        self._view_id = view_id
        self._cam_K = self.cam_Ks[view_id]

    def set_ply_property(self, plyfile):

        plydata = vtk.vtkPLYReader()
        plydata.SetFileName(plyfile)
        plydata.Update()

        '''replace aligned points'''
        polydata = plydata.GetOutput()
        points_array = numpy_to_vtk(self.vertices[..., :3], deep=True)
        # Update the point information of vtk
        polydata.GetPoints().SetData(points_array)
        # update changes
        plydata.Update()

        return plydata

    def set_render(self, detection=True, *args, **kwargs):
        renderer = vtk.vtkRenderer()
        renderer.ResetCamera()

        '''draw world system'''
        renderer.AddActor(self.set_axes_actor())

        '''Move camera to the view'''
        render_cam_pose = self.cam_Ts[self.view_id]
        cam_loc = render_cam_pose[:3, 3]
        render_cam_R = render_cam_pose[:3, :3]
        cam_forward_vec = -render_cam_R[:, 2]
        cam_fp = cam_loc + cam_forward_vec
        cam_up = render_cam_R[:, 1]
        fov_y = (2 * np.arctan((self.cam_K[1][2] * 2 + 1) / 2. / self.cam_K[1][1])) / np.pi * 180
        camera = self.set_camera(cam_loc, cam_fp, cam_up, fov_y=fov_y)
        renderer.SetActiveCamera(camera)

        if 'cam_pose' in kwargs['type']:
            for cam_T in self.cam_Ts:
                # draw cam center
                cam_center = cam_T[:3, 3]
                sphere_actor = self.set_actor(
                    self.set_mapper(self.set_sphere_property(cam_center, 0.1), mode='model'))
                sphere_actor.GetProperty().SetColor([0.8, 0.1, 0.1])
                sphere_actor.GetProperty().SetInterpolationToPBR()
                renderer.AddActor(sphere_actor)

                # draw cam orientations
                color = [[1, 0, 0], [0, 1, 0], [0., 0., 1.]]
                vectors = cam_T[:3, :3].T
                for index in range(vectors.shape[0]):
                    arrow_actor = self.set_arrow_actor(cam_center, vectors[index])
                    arrow_actor.GetProperty().SetColor(color[index])
                    renderer.AddActor(arrow_actor)

        if 'bbox' in kwargs['type']:
            '''load bounding boxes'''
            for cls_id, cls_name, instance_bbox in zip(self.class_ids, self.class_names, self.instance_bboxes):
                if cls_id == 0:
                    continue
                centroid = instance_bbox[:3]
                vectors = np.diag(instance_bbox[3:6]) / 2.
                box_actor = self.get_bbox_line_actor(centroid, vectors, self.cls_palette[cls_id]*255, 1., 6)
                box_actor.GetProperty().SetInterpolationToPBR()
                renderer.AddActor(box_actor)

                # draw class text
                text_actor = self.add_text(tuple(centroid + [0, vectors[1, 1] + 0.2, 0]), cls_name, scale=0.15)
                text_actor.SetCamera(renderer.GetActiveCamera())
                renderer.AddActor(text_actor)

        if 'mesh' in kwargs['type']:
            '''load ply mesh file'''
            ply_actor = self.set_actor(self.set_mapper(self.set_ply_property(self.mesh_file), 'model'))
            ply_actor.GetProperty().SetInterpolationToPBR()
            renderer.AddActor(ply_actor)

        if 'pointcloud' in kwargs['type']:
            '''load point actor'''
            instance_labels = [inst_id if cls_id!=0 else 0 for cls_id, inst_id in zip(self.semantic_labels, self.instance_labels)]

            colors = self.inst_palette[instance_labels] * 255
            point_actor = self.set_actor(self.set_mapper(self.set_points_property(self.vertices[:, :3], colors), 'box'))
            point_actor.GetProperty().SetPointSize(4)
            point_actor.GetProperty().SetInterpolationToPBR()
            renderer.AddActor(point_actor)

        '''light'''
        positions = [(10, 10, 10), (-10, 10, 10), (10, 10, -10), (-10, 10, -10)]
        for position in positions:
            light = vtk.vtkLight()
            light.SetIntensity(1)
            light.SetPosition(*position)
            light.SetPositional(True)
            light.SetFocalPoint(0, 0, 0)
            light.SetColor(1., 1., 1.)
            renderer.AddLight(light)

        renderer.SetBackground(1., 1., 1.)
        return renderer

    def set_render_window(self, offline, *args, **kwargs):
        render_window = vtk.vtkRenderWindow()
        renderer = self.set_render(*args, **kwargs)
        # renderer.SetUseDepthPeeling(1)
        render_window.AddRenderer(renderer)
        if 'image_size' in kwargs:
            render_window.SetSize(kwargs['image_size'][1], kwargs['image_size'][0])
        else:
            render_window.SetSize(*np.int32((self.cam_K[:2, 2] * 2 + 1)))
        render_window.SetOffScreenRendering(offline)
        return render_window


def image_grid(imgs: Union[List[np.ndarray], np.ndarray]):
    h, w = imgs[0].shape[:2]

    if len(imgs) == 1:
        cols = 1
        rows = 1
    else:
        cols = np.ceil(np.sqrt(h * golden * len(imgs) / w)).astype(np.uint16)
        rows = np.ceil(len(imgs) / cols).astype(np.uint16)

    grid = Image.new('RGB', size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(Image.fromarray(img), box=(i % cols * w, i // cols * h))
    return grid


class VIS_ScanNet_2D(VIS_BASE):
    def __init__(self, color_maps, inst_info, cls_maps, **kwargs):
        super(VIS_ScanNet_2D, self).__init__()
        self.color_maps = np.array(color_maps, dtype=color_maps[0].dtype)
        self.inst_info = inst_info
        self.cls_maps = np.array(cls_maps, dtype=cls_maps[0].dtype)
        self.projected_inst_boxes = kwargs.get('projected_inst_boxes', None)
        if 'class_names' in kwargs:
            self.class_names = kwargs['class_names']
        self.cls_palette = (np.array(sns.color_palette('hls', len(self.class_names))) * 255).astype(np.uint8)

    def draw_colors(self):
        image_grid(self.color_maps).show()

    def draw_cls_maps(self):
        cls_color_maps = np.zeros(shape=(*self.cls_maps.shape, 3), dtype=np.uint8)
        for cls_id, color in enumerate(self.cls_palette):
            cls_color_maps += self.cls_palette[cls_id] * np.ones_like(cls_color_maps) * (self.cls_maps == cls_id)[..., np.newaxis]
        image_grid(cls_color_maps).show()

    def draw_inst_maps(self, type=()):
        masked_images = self.color_maps.astype(np.uint8).copy()
        font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeSans.ttf", 75, encoding="unic")

        inst_maps = []
        for im_id in range(len(masked_images)):
            insts_per_img = self.inst_info[im_id]
            source_img = Image.fromarray(masked_images[im_id]).convert("RGB")
            img_draw = ImageDraw.Draw(source_img, 'RGBA')
            # Number of instances
            if not len(insts_per_img):
                print("\n*** No instances to display *** \n")
                continue
            for inst in insts_per_img:
                color = tuple(self.cls_palette[inst['category_id']])
                x_min, y_min, width, height = inst['bbox2d']
                x_max = x_min + width - 1
                y_max = y_min + height - 1
                img_draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=9)
                img_draw.text((x_min, y_min), self.class_names[inst['category_id']], font=font, fill='white',
                              stroke_width=6, stroke_fill='black')
                if 'mask' in type:
                    mask = np.zeros(masked_images.shape[1:3], dtype=bool)
                    mask[y_min: y_max + 1, x_min: x_max + 1] = inst['mask']
                    inst_mask = binary_mask_to_polygon(mask, tolerance=2)
                    for verts in inst_mask:
                        img_draw.polygon(verts, fill=(*color, 125))
            inst_maps.append(np.array(source_img))
        image_grid(inst_maps).show()

    def draw_box2d_from_3d(self):
        masked_images = self.color_maps.copy()
        font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeSans.ttf", 25, encoding="unic")
        inst_maps = []
        width = 20
        for im_id in range(len(masked_images)):
            insts_per_img = self.inst_info[im_id]
            projected_insts_per_img = self.projected_inst_boxes[im_id]
            source_img = Image.fromarray(masked_images[im_id]).convert("RGB")
            img_draw = ImageDraw.Draw(source_img)
            # Number of instances
            if not len(insts_per_img):
                print("\n*** No instances to display *** \n")
                continue
            for inst_info, proj_corners in zip(insts_per_img, projected_insts_per_img):
                if proj_corners is None: continue
                color = tuple(self.cls_palette[inst_info['category_id']])
                proj_corners = [tuple(corner) for corner in proj_corners]
                img_draw.line([proj_corners[0], proj_corners[1], proj_corners[3], proj_corners[2], proj_corners[0]],
                          fill=color, width=width)
                img_draw.line([proj_corners[4], proj_corners[5], proj_corners[7], proj_corners[6], proj_corners[4]],
                          fill=color, width=width)
                img_draw.line([proj_corners[0], proj_corners[4]],
                          fill=color, width=width)
                img_draw.line([proj_corners[1], proj_corners[5]],
                          fill=color, width=width)
                img_draw.line([proj_corners[2], proj_corners[6]],
                          fill=color, width=width)
                img_draw.line([proj_corners[3], proj_corners[7]],
                          fill=color, width=width)
            inst_maps.append(np.array(source_img))
        image_grid(inst_maps).show()

class Vis_ScanNet_GT(VIS_BASE):
    '''
    visualization a GT sample from ScanNet.
    '''
    def __init__(self, cam_Ks, cam_Ts, instance_attrs, class_names):
        super(Vis_ScanNet_GT, self).__init__()
        self.cam_Ks = cam_Ks
        self.cam_Ts = cam_Ts
        self.instance_attrs = instance_attrs
        self.class_names = class_names
        self._view_id = 0
        self.update_view(view_id=self.view_id)
        self.cls_palette = np.array([(0., 0., 0.), *sns.color_palette("hls", len(class_names) - 1)])

    @property
    def view_id(self):
        return self._view_id

    @property
    def bbox_params(self):
        return self._bbox_params

    @property
    def inst_class_ids(self):
        return self._inst_class_ids

    @property
    def inst_classes(self):
        return self._inst_classes

    @property
    def cam_T(self):
        return self._cam_T

    def update_view(self, view_id):
        self._view_id = view_id
        self._cam_K = self.cam_Ks[view_id]
        self._cam_T = self.cam_Ts[view_id]
        inst_info = self.instance_attrs[view_id]

        self._bbox_params = [inst['bbox3d'] for inst in inst_info]
        self._inst_class_ids = [inst['category_id'] for inst in inst_info]
        self._inst_classes = [self.class_names[idx] for idx in self.inst_class_ids]

    def set_render(self, *args, **kwargs):
        renderer = vtk.vtkRenderer()
        renderer.ResetCamera()

        '''draw world system'''
        renderer.AddActor(self.set_axes_actor())

        '''Move camera to the view'''
        render_cam_pose = self.cam_T
        cam_loc = render_cam_pose[:3, 3]
        render_cam_R = render_cam_pose[:3, :3]
        cam_forward_vec = -render_cam_R[:, 2]
        cam_fp = cam_loc + cam_forward_vec
        cam_up = render_cam_R[:, 1]
        fov_y = (2 * np.arctan((self.cam_K[1][2] * 2 + 1) / 2. / self.cam_K[1][1])) / np.pi * 180
        camera = self.set_camera(cam_loc, cam_fp, cam_up, fov_y=fov_y)
        renderer.SetActiveCamera(camera)

        if 'cam_pose' in kwargs['type']:
            for cam_T in self.cam_Ts:
                # draw cam center
                cam_center = cam_T[:3, 3]
                sphere_actor = self.set_actor(
                    self.set_mapper(self.set_sphere_property(cam_center, 0.1), mode='model'))
                sphere_actor.GetProperty().SetColor([0.8, 0.1, 0.1])
                sphere_actor.GetProperty().SetInterpolationToPBR()
                renderer.AddActor(sphere_actor)

                # draw cam orientations
                color = [[1, 0, 0], [0, 1, 0], [0., 0., 1.]]
                vectors = cam_T[:3, :3].T
                for index in range(vectors.shape[0]):
                    arrow_actor = self.set_arrow_actor(cam_center, vectors[index])
                    arrow_actor.GetProperty().SetColor(color[index])
                    renderer.AddActor(arrow_actor)

        if 'bbox' in kwargs['type']:
            '''load bounding boxes'''
            for cls_id, cls_name, instance_bbox in zip(self.inst_class_ids, self.inst_classes, self.bbox_params):
                centroid = instance_bbox[:3]
                vectors = np.diag(instance_bbox[3:6]) / 2.
                box_actor = self.get_bbox_line_actor(centroid, vectors, self.cls_palette[cls_id]*255, 1., 6)
                box_actor.GetProperty().SetInterpolationToPBR()
                renderer.AddActor(box_actor)

                # draw class text
                text_actor = self.add_text(tuple(centroid + [0, vectors[1, 1] + 0.2, 0]), cls_name, scale=0.15)
                text_actor.SetCamera(renderer.GetActiveCamera())
                renderer.AddActor(text_actor)

        '''light'''
        positions = [(10, 10, 10), (-10, 10, 10), (10, 10, -10), (-10, 10, -10)]
        for position in positions:
            light = vtk.vtkLight()
            light.SetIntensity(1)
            light.SetPosition(*position)
            light.SetPositional(True)
            light.SetFocalPoint(0, 0, 0)
            light.SetColor(1., 1., 1.)
            renderer.AddLight(light)

        renderer.SetBackground(1., 1., 1.)
        return renderer

    def set_render_window(self, offline, *args, **kwargs):
        render_window = vtk.vtkRenderWindow()
        renderer = self.set_render(*args, **kwargs)
        # renderer.SetUseDepthPeeling(1)
        render_window.AddRenderer(renderer)
        if 'image_size' in kwargs:
            render_window.SetSize(kwargs['image_size'][1], kwargs['image_size'][0])
        else:
            render_window.SetSize(*np.int32((self.cam_K[:2, 2] * 2 + 1)))
        render_window.SetOffScreenRendering(offline)
        return render_window


class VIS_ScanNet_RESULT(VIS_BASE):
    def __init__(self, cam_Ks, cam_Ts, box3ds, mesh_files, category_ids, class_names, **kwargs):
        super(VIS_ScanNet_RESULT, self).__init__()
        self.cam_Ks = cam_Ks
        self.cam_Ts = cam_Ts
        self._view_id = 0
        self.update_view(view_id=self.view_id)

        self.box3ds = box3ds
        self.mesh_files = mesh_files if len(mesh_files) else [None] * box3ds.shape[0]
        self.class_ids = category_ids
        self.class_names = [class_names[cls_id] for cls_id in category_ids]
        self.cls_palette = np.array(sns.color_palette('hls', len(class_names)))
        self.color_maps = np.array(kwargs.get('color_maps', None))
        self.projected_inst_boxes = kwargs.get('projected_inst_boxes', None)

    @property
    def view_id(self):
        return self._view_id

    def update_view(self, view_id):
        self._view_id = view_id
        self._cam_K = self.cam_Ks[view_id]

    def draw_box2d_from_3d(self, save_dir=None):
        masked_images = self.color_maps.copy()
        font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeSans.ttf", 25, encoding="unic")
        inst_maps = []
        width = 5
        for im_id in range(len(masked_images)):
            projected_insts_per_img = self.projected_inst_boxes[im_id]
            source_img = Image.fromarray(masked_images[im_id]).convert("RGB")
            img_draw = ImageDraw.Draw(source_img)
            # Number of instances
            if not len(projected_insts_per_img):
                print("\n*** No instances to display *** \n")
                continue
            for inst_id, proj_box2d in enumerate(projected_insts_per_img):
                if proj_box2d is None: continue
                color = tuple(np.int32(self.cls_palette[self.class_ids[inst_id]] * 255))
                proj_corners = [tuple(corner) for corner in proj_box2d]
                img_draw.line([proj_corners[0], proj_corners[1], proj_corners[3], proj_corners[2], proj_corners[0]],
                          fill=color, width=width)
                img_draw.line([proj_corners[4], proj_corners[5], proj_corners[7], proj_corners[6], proj_corners[4]],
                          fill=color, width=width)
                img_draw.line([proj_corners[0], proj_corners[4]],
                          fill=color, width=width)
                img_draw.line([proj_corners[1], proj_corners[5]],
                          fill=color, width=width)
                img_draw.line([proj_corners[2], proj_corners[6]],
                          fill=color, width=width)
                img_draw.line([proj_corners[3], proj_corners[7]],
                          fill=color, width=width)
            inst_maps.append(np.array(source_img))

        grid_image = image_grid(inst_maps)
        if save_dir is not None:
            grid_image.save(save_dir)
        else:
            grid_image.show()

    def set_render(self, *args, **kwargs):
        renderer = vtk.vtkRenderer()
        renderer.ResetCamera()

        # '''draw world system'''
        # renderer.AddActor(self.set_axes_actor())

        render_cam_pose = self.cam_Ts[self.view_id]

        cam_loc = render_cam_pose[:3, 3]
        render_cam_R = render_cam_pose[:3, :3]
        cam_forward_vec = -render_cam_R[:, 2]
        cam_fp = cam_loc + cam_forward_vec
        cam_up = render_cam_R[:, 1]
        fov_y = (2 * np.arctan((self.cam_K[1][2] * 2 + 1) / 2. / self.cam_K[1][1])) / np.pi * 180
        camera = self.set_camera(cam_loc, cam_fp, cam_up, fov_y=fov_y)
        renderer.SetActiveCamera(camera)

        '''draw camera positions'''
        if 'cam_pose' in kwargs['type']:
            cam_T_list = self.cam_Ts if len(self.cam_Ts.shape) > 2 else [self.cam_Ts]
            # draw cam center
            for cam_T in cam_T_list:
                cam_center = cam_T[:3, 3]
                sphere_actor = self.set_actor(
                    self.set_mapper(self.set_sphere_property(cam_center, 0.1), mode='model'))
                sphere_actor.GetProperty().SetColor([0.8, 0.1, 0.1])
                sphere_actor.GetProperty().SetInterpolationToPBR()
                renderer.AddActor(sphere_actor)

                # draw cam orientations
                color = [[1, 0, 0], [0, 1, 0], [0., 0., 1.]]
                vectors = cam_T[:3, :3].T
                for index in range(vectors.shape[0]):
                    arrow_actor = self.set_arrow_actor(cam_center, vectors[index])
                    arrow_actor.GetProperty().SetColor(color[index])
                    renderer.AddActor(arrow_actor)

        if 'bbox' in kwargs['type']:
            '''draw instance bboxes and meshes'''
            for cls_id, cls_name, box3d, mesh_file in zip(self.class_ids, self.class_names, self.box3ds,
                                                          self.mesh_files):
                # centroid = box3d[0:3]
                # R_mat = rotation_matrix([0, 1, 0], box3d[6])
                #
                # vectors = np.diag(np.array(box3d[3:6]) / 2.).dot(R_mat.T)
                # box_actor = self.get_bbox_line_actor(centroid, vectors, self.cls_palette[cls_id] * 255, 1.)
                # box_actor.GetProperty().SetInterpolationToPBR()
                # renderer.AddActor(box_actor)
                #
                # # draw class text
                # text_actor = self.add_text(tuple(centroid + [0, vectors[1, 1] + 0.2, 0]), cls_name, scale=0.15)
                # text_actor.SetCamera(renderer.GetActiveCamera())
                # renderer.AddActor(text_actor)

                # draw meshes
                if mesh_file is not None:
                    obj_actor = self.get_obj_actor(mesh_file)
                    obj_actor.GetProperty().SetColor(self.cls_palette[cls_id])
                    obj_actor.GetProperty().SetInterpolationToPBR()
                    renderer.AddActor(obj_actor)

        '''draw floor plane'''
        plane_actor = self.get_plane_actor((-5, 0, -5), (-5, 0, 5), (5, 0, -5), (0.9, 0.9, 0.9), opacity=1.)
        plane_actor.GetProperty().SetInterpolationToPBR()
        renderer.AddActor(plane_actor)

        '''light'''
        positions = [(10, 10, 10), (-10, 10, 10), (10, 10, -10), (-10, 10, -10)]
        for position in positions:
            light = vtk.vtkLight()
            light.SetIntensity(1)
            light.SetPosition(*position)
            light.SetPositional(True)
            light.SetFocalPoint(0, 0, 0)
            light.SetColor(1., 1., 1.)
            renderer.AddLight(light)

        renderer.SetBackground(1., 1., 1.)
        return renderer


