#  Copyright (c) 1.2022. Yinyu Nie
#  License: MIT

import vtk
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
from typing import List, Union
from utils.tools import binary_mask_to_polygon
import cv2

from utils.vis_base import VIS_BASE
from utils.threed_front.tools.threed_front_scene import rotation_matrix


golden = (1 + 5 ** 0.5) / 2

def read_3dfront_obj2vtk(instance):
    '''Read and transform mesh from 3d front to vtk'''
    '''Read mesh to vtk'''
    vtk_object = vtk.vtkOBJReader()
    vtk_object.SetFileName(instance.raw_model_path)
    vtk_object.Update()

    '''Transform mesh'''
    # get points from object
    polydata = vtk_object.GetOutput()
    # read points using vtk_to_numpy
    obj_points = vtk_to_numpy(polydata.GetPoints().GetData()).astype(np.float)
    obj_points_transformed = instance._transform(obj_points)
    points_array = numpy_to_vtk(obj_points_transformed[..., :3], deep=True)
    polydata.GetPoints().SetData(points_array)
    vtk_object.Update()

    return vtk_object

def read_3dfront_extra(instance):
    '''Read and transform mesh from 3d front to vtk'''
    '''Transform vertices'''
    obj_points_transformed = instance._transform(instance.xyz)
    return obj_points_transformed, instance.faces


class VIS_3DFRONT(VIS_BASE):
    def __init__(self, rooms, cam_K, cam_Ts, inst_info, layout_boxes, class_names):
        super(VIS_3DFRONT, self).__init__()
        self._cam_K = cam_K
        self.cam_Ts = cam_Ts
        self.layout_boxes = layout_boxes
        self.insts_vtk = [read_3dfront_obj2vtk(bbox) for room in rooms for bbox in room.bboxes]
        self.bbox_params = np.array([np.concatenate([bbox.centroid(), bbox.size, [bbox.z_angle]]) for room in rooms for bbox in room.bboxes])
        # focus on objects of interest
        self.class_ids = np.zeros(len(self.insts_vtk), dtype=np.uint16)
        # only need unique 3D boxes
        all_inst_info = sum(inst_info, [])
        unique_inst_marks = set([inst['inst_mark'] for inst in all_inst_info])
        unique_inst_info = []
        for unique_inst_mark in unique_inst_marks:
            unique_inst = next((inst for inst in all_inst_info if inst['inst_mark'] == unique_inst_mark), None)
            if unique_inst and unique_inst['bbox3d'] is not None:
                unique_inst_info.append(unique_inst)
        focused_3dboxes = np.array([inst['bbox3d'] for inst in unique_inst_info])
        pairwise_box_dists = np.linalg.norm(self.bbox_params[:, np.newaxis] - focused_3dboxes[np.newaxis], axis=-1)
        focused_idx_to_all = pairwise_box_dists.argmin(axis=0)
        unique_inst_classes = [inst['category_id'] for inst in unique_inst_info]
        self.class_ids[focused_idx_to_all] = unique_inst_classes
        self.class_names = [class_names[idx] for idx in self.class_ids]
        self.floors = [read_3dfront_extra(ei) for room in rooms for ei in room.extras if ei.model_type == 'Floor']
        self.ceilings = [read_3dfront_extra(ei) for room in rooms for ei in room.extras if ei.model_type == 'Ceiling']
        self.walls = [read_3dfront_extra(ei) for room in rooms for ei in room.extras if ei.model_type == 'WallInner']
        self.doors_windows = [read_3dfront_extra(ei) for room in rooms for ei in room.extras if ei.model_type in ['Window', 'Door']]
        self.cls_palette = np.array(sns.color_palette('hls', len(class_names)))

    def set_render(self, *args, **kwargs):
        renderer = vtk.vtkRenderer()
        renderer.ResetCamera()

        '''draw world system'''
        renderer.AddActor(self.set_axes_actor())

        if 'view_id' in kwargs:
            view_id = kwargs['view_id']
            render_cam_pose = self.cam_Ts[view_id]
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

        '''draw class lookup table'''
        if 'lookup_class' in kwargs['type']:
            scalar_bar_actor = self.set_scalar_bar_actor(self.class_names, [self.cls_palette[idx] for idx in self.class_ids])
            renderer.AddActor(scalar_bar_actor)

        '''draw instance meshes, bboxes'''
        for inst_vtk, inst_bbox, cls_id, cls_name in zip(self.insts_vtk, self.bbox_params, self.class_ids, self.class_names):
            # draw instance bbox
            if 'mesh' in kwargs['type']:
                object_actor = self.set_actor(self.set_mapper(inst_vtk, 'model'))
                object_actor.GetProperty().SetColor(self.cls_palette[cls_id])
                object_actor.GetProperty().SetInterpolationToPBR()
                renderer.AddActor(object_actor)

            # draw instance bbox
            if 'bbox' in kwargs['type']:
                centroid = inst_bbox[0:3]
                R_mat = rotation_matrix([0, 1, 0], inst_bbox[6])

                vectors = np.diag(np.array(inst_bbox[3:6]) / 2.).dot(R_mat.T)
                box_actor = self.get_bbox_line_actor(centroid, vectors, self.cls_palette[cls_id]*255, 1., 6)
                box_actor.GetProperty().SetInterpolationToPBR()
                renderer.AddActor(box_actor)

                # draw class text
                text_actor = self.add_text(tuple(centroid + [0, vectors[1, 1] + 0.2, 0]), cls_name, scale=0.15)
                text_actor.SetCamera(renderer.GetActiveCamera())
                renderer.AddActor(text_actor)

                # draw orientations
                color = [[1, 0, 0], [0, 1, 0], [0., 0., 1.]]

                for index in range(vectors.shape[0]):
                    arrow_actor = self.set_arrow_actor(centroid, vectors[index])
                    arrow_actor.GetProperty().SetColor(color[index])
                    renderer.AddActor(arrow_actor)

        # draw layout boxes.
        if 'layout_box' in kwargs['type']:
            for layout_box in self.layout_boxes:
                floor_center, x_vec, y_vec, z_vec = layout_box[:3], layout_box[3:6], layout_box[6:9], layout_box[9:12]
                centroid = floor_center + y_vec/2
                vectors = np.array([x_vec, y_vec/2, z_vec])
                box_actor = self.get_bbox_line_actor(centroid, vectors, [125, 125, 125], 1., 6)
                box_actor.GetProperty().SetInterpolationToPBR()
                renderer.AddActor(box_actor)

                # draw orientations
                color = [[1, 0, 0], [0, 1, 0], [0., 0., 1.]]

                for index in range(vectors.shape[0]):
                    arrow_actor = self.set_arrow_actor(centroid, vectors[index])
                    arrow_actor.GetProperty().SetColor(color[index])
                    renderer.AddActor(arrow_actor)

        # draw original layout.
        if 'ori_layout' in kwargs['type']:
            '''draw floors'''
            for floor in self.floors:
                floor_prop = self.set_polygon_property(floor[0], floor[1])
                floor_actor = self.set_actor(self.set_mapper(floor_prop, 'box'))
                floor_actor.GetProperty().SetOpacity(1)
                floor_actor.GetProperty().SetInterpolationToPBR()
                renderer.AddActor(floor_actor)

            '''draw ceillings'''
            for ceiling in self.ceilings:
                ceiling_prop = self.set_polygon_property(ceiling[0], ceiling[1])
                ceiling_actor = self.set_actor(self.set_mapper(ceiling_prop, 'box'))
                ceiling_actor.GetProperty().SetOpacity(0.2)
                ceiling_actor.GetProperty().SetInterpolationToPBR()
                renderer.AddActor(ceiling_actor)

            '''draw walls'''
            for wall in self.walls:
                wall_prop = self.set_polygon_property(wall[0], wall[1])
                wall_actor = self.set_actor(self.set_mapper(wall_prop, 'box'))
                wall_actor.GetProperty().SetOpacity(1)
                wall_actor.GetProperty().SetInterpolationToPBR()
                renderer.AddActor(wall_actor)

            '''draw doors and windows'''
            for extra in self.doors_windows:
                extra_prop = self.set_polygon_property(extra[0], extra[1])
                extra_actor = self.set_actor(self.set_mapper(extra_prop, 'box'))
                extra_actor.GetProperty().SetColor([1, 0, 0])
                extra_actor.GetProperty().SetOpacity(1)
                extra_actor.GetProperty().SetInterpolationToPBR()
                renderer.AddActor(extra_actor)

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

def image_grid(imgs: Union[List[np.ndarray], np.ndarray]):
    h, w = imgs[0].shape[:2]

    cols = np.floor(np.sqrt(h * golden * len(imgs) / w)).astype(np.uint16)
    rows = np.ceil(len(imgs) / cols).astype(np.uint16)

    grid = Image.new('RGB', size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(Image.fromarray(img), box=(i % cols * w, i // cols * h))
    return grid


class VIS_3DFRONT_2D(object):
    '''This class is to visualize the renderings of 3DFRONT scenes.'''
    def __init__(self, color_maps, inst_info, cls_maps, **kwargs):
        self.color_maps = np.array(color_maps, dtype=color_maps[0].dtype)
        self.inst_info = inst_info
        self.cls_maps = np.array(cls_maps, dtype=cls_maps[0].dtype)
        self.projected_inst_boxes = kwargs.get('projected_inst_boxes', None)
        if 'class_names' in kwargs:
            self.class_names = kwargs['class_names']
        self.cls_palette = (np.array(sns.color_palette('hls', len(self.class_names))) * 255).astype(np.uint8)

    def draw_box2d_from_3d(self):
        masked_images = self.color_maps.copy()
        font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeSans.ttf", 25, encoding="unic")
        inst_maps = []
        width = 5
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

    def draw_colors(self):
        image_grid(self.color_maps).show()

    def draw_cls_maps(self):
        cls_color_maps = np.zeros(shape=(*self.cls_maps.shape, 3), dtype=np.uint8)
        for cls_id, color in enumerate(self.cls_palette):
            cls_color_maps += self.cls_palette[cls_id] * np.ones_like(cls_color_maps) * (self.cls_maps == cls_id)[..., np.newaxis]
        image_grid(cls_color_maps).show()

    def draw_inst_maps(self, type=()):
        masked_images = self.color_maps.astype(np.uint8).copy()
        font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeSans.ttf", 25, encoding="unic")

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
                img_draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=3)
                img_draw.text((x_min, y_min), self.class_names[inst['category_id']], font=font, fill='white',
                              stroke_width=3, stroke_fill='black')
                if 'mask' in type:
                    mask = np.zeros(masked_images.shape[1:3], dtype=bool)
                    mask[y_min: y_max + 1, x_min: x_max + 1] = inst['mask']
                    inst_mask = binary_mask_to_polygon(mask, tolerance=2)
                    for verts in inst_mask:
                        img_draw.polygon(verts, fill=(*color, 75))
            inst_maps.append(np.array(source_img))
        image_grid(inst_maps).show()

def read_obj2vtk_from_box(model_path, bbox3d):
    '''Read and transform mesh from 3d front to vtk based on the bbox3d'''
    '''Read mesh to vtk'''
    if model_path in [None, 'null']:
        return None

    vtk_object = vtk.vtkOBJReader()
    vtk_object.SetFileName(model_path)
    vtk_object.Update()

    '''Transform mesh'''
    box_center = bbox3d[:3]
    box_sizes = bbox3d[3:6]
    box_angle = bbox3d[6]
    rot_mat = rotation_matrix([0, 1, 0], box_angle)

    # get points from object
    polydata = vtk_object.GetOutput()
    # read points using vtk_to_numpy
    obj_points = vtk_to_numpy(polydata.GetPoints().GetData()).astype(np.float)
    original_center = (obj_points.max(0) + obj_points.min(0))/2.
    original_size = (obj_points.max(0) - obj_points.min(0))
    obj_points -= original_center
    obj_points = obj_points / original_size * box_sizes
    obj_points_transformed = obj_points.dot(rot_mat.T) + box_center

    points_array = numpy_to_vtk(obj_points_transformed[..., :3], deep=True)
    polydata.GetPoints().SetData(points_array)
    vtk_object.Update()

    return vtk_object


class VIS_3DFRONT_SAMPLE(VIS_BASE):
    def __init__(self, cam_K, cam_Ts, inst_info, layout_boxes, class_names):
        super(VIS_3DFRONT_SAMPLE, self).__init__()
        self._cam_K = cam_K
        self.cam_Ts = cam_Ts
        self.bbox_params = [[inst['bbox3d'] for inst in rendering] for rendering in inst_info]
        self.insts_vtk = [[read_obj2vtk_from_box(inst['model_path'], inst['bbox3d']) for inst in rendering] for rendering in inst_info]
        self.class_ids = [[inst['category_id'] for inst in rendering] for rendering in inst_info]
        self.class_names = [[class_names[cls_id] for cls_id in rendering] for rendering in self.class_ids]
        self.cls_palette = np.array(sns.color_palette('hls', len(class_names)))
        self.layout_boxes = layout_boxes

    def set_render(self, *args, **kwargs):
        renderer = vtk.vtkRenderer()
        renderer.ResetCamera()

        '''draw world system'''
        renderer.AddActor(self.set_axes_actor())

        view_id = kwargs['view_id'] if 'view_id' in kwargs else 0
        if 'view_id' in kwargs:
            render_cam_pose = self.cam_Ts[view_id]
            if len(render_cam_pose.shape) == 3:
                render_cam_pose = render_cam_pose[0]
            cam_loc = render_cam_pose[:3, 3]
            render_cam_R = render_cam_pose[:3, :3]
            cam_forward_vec = -render_cam_R[:, 2]
            cam_fp = cam_loc + cam_forward_vec
            cam_up = render_cam_R[:, 1]
            fov_y = (2 * np.arctan((self.cam_K[1][2] * 2 + 1) / 2. / self.cam_K[1][1])) / np.pi * 180
            camera = self.set_camera(cam_loc, cam_fp, cam_up, fov_y=fov_y)
            renderer.SetActiveCamera(camera)
        else:
            cam_loc = np.array([0, 5, 0])
            cam_forward_vec = np.array([0, -1, 0])
            cam_fp = cam_loc + cam_forward_vec
            cam_up = np.array([0, 0, 1])
            fov_y = (2 * np.arctan((self.cam_K[1][2] * 2 + 1) / 2. / self.cam_K[1][1])) / np.pi * 180
            camera = self.set_camera(cam_loc, cam_fp, cam_up, fov_y=fov_y)
            renderer.SetActiveCamera(camera)

        '''draw camera positions'''
        if 'cam_pose' in kwargs['type']:
            cam_T_list = self.cam_Ts[view_id]
            if len(cam_T_list.shape) == 2:
                cam_T_list = [cam_T_list]
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

        '''draw class lookup table'''
        if 'lookup_class' in kwargs['type']:
            scalar_bar_actor = self.set_scalar_bar_actor(self.class_names[view_id], [self.cls_palette[idx] for idx in self.class_ids[view_id]])
            renderer.AddActor(scalar_bar_actor)

        '''draw instance meshes, bboxes'''
        for inst_vtk, inst_bbox, cls_id, cls_name in zip(self.insts_vtk[view_id], self.bbox_params[view_id], self.class_ids[view_id], self.class_names[view_id]):
            if inst_vtk is None: continue
            # draw instance mesh
            if 'mesh' in kwargs['type']:
                object_actor = self.set_actor(self.set_mapper(inst_vtk, 'model'))
                object_actor.GetProperty().SetColor(self.cls_palette[cls_id])
                object_actor.GetProperty().SetInterpolationToPBR()
                renderer.AddActor(object_actor)

            # draw instance bbox
            if 'bbox' in kwargs['type']:
                centroid = inst_bbox[0:3]
                R_mat = rotation_matrix([0, 1, 0], inst_bbox[6])

                vectors = np.diag(np.array(inst_bbox[3:6]) / 2.).dot(R_mat.T)
                box_actor = self.get_bbox_line_actor(centroid, vectors, self.cls_palette[cls_id]*255, 1., 6)
                box_actor.GetProperty().SetInterpolationToPBR()
                renderer.AddActor(box_actor)

                # draw class text
                text_actor = self.add_text(tuple(centroid + [0, vectors[1, 1] + 0.2, 0]), cls_name, scale=0.15)
                text_actor.SetCamera(renderer.GetActiveCamera())
                renderer.AddActor(text_actor)

                # draw orientations
                color = [[1, 0, 0], [0, 1, 0], [0., 0., 1.]]

                for index in range(vectors.shape[0]):
                    arrow_actor = self.set_arrow_actor(centroid, vectors[index])
                    arrow_actor.GetProperty().SetColor(color[index])
                    renderer.AddActor(arrow_actor)

        # draw layout boxes.
        if 'layout_box' in kwargs['type']:
            layout_box = self.layout_boxes[view_id]
            floor_center, x_vec, y_vec, z_vec = layout_box[:3], layout_box[3:6], layout_box[6:9], layout_box[9:12]
            centroid = floor_center + y_vec/2
            vectors = np.array([x_vec, y_vec/2, z_vec])
            box_actor = self.get_bbox_line_actor(centroid, vectors, [125, 125, 125], 1., 6)
            box_actor.GetProperty().SetInterpolationToPBR()
            renderer.AddActor(box_actor)

            # draw orientations
            color = [[1, 0, 0], [0, 1, 0], [0., 0., 1.]]

            for index in range(vectors.shape[0]):
                arrow_actor = self.set_arrow_actor(centroid, vectors[index])
                arrow_actor.GetProperty().SetColor(color[index])
                renderer.AddActor(arrow_actor)

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

class VIS_3DFRONT_ROOM_SAMPLE(VIS_3DFRONT_SAMPLE):
    def __init__(self, cam_K, cam_Ts, inst3D_info, layout_box, class_names):
        super(VIS_3DFRONT_SAMPLE, self).__init__()
        self._cam_K = cam_K
        self.cam_Ts = cam_Ts
        self.insts = [{'inst_mark': key, 'category_id': item['category_id'], 'bbox3d': item['bbox3d'],
                       'vtk': read_obj2vtk_from_box(item['model_path'], item['bbox3d'])} for key, item in
                      inst3D_info.items()]
        self.class_names = class_names
        self.cls_palette = np.array(sns.color_palette('hls', len(class_names)))
        self.layout_box = layout_box


    def set_render(self, *args, **kwargs):
        renderer = vtk.vtkRenderer()
        renderer.ResetCamera()

        '''draw world system'''
        renderer.AddActor(self.set_axes_actor())

        view_id = kwargs['view_id'] if 'view_id' in kwargs else 0

        render_cam_pose = self.cam_Ts[view_id]
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
            cam_T_list = np.array(self.cam_Ts)
            if len(cam_T_list.shape) == 2:
                cam_T_list = [cam_T_list]
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

        '''draw instance meshes, bboxes'''
        for inst in self.insts:
            if inst['vtk'] is None: continue
            # draw instance mesh
            if 'mesh' in kwargs['type']:
                object_actor = self.set_actor(self.set_mapper(inst['vtk'], 'model'))
                object_actor.GetProperty().SetColor(self.cls_palette[inst['category_id']])
                object_actor.GetProperty().SetInterpolationToPBR()
                renderer.AddActor(object_actor)

            # draw instance bbox
            if 'bbox' in kwargs['type']:
                centroid = inst['bbox3d'][0:3]
                R_mat = rotation_matrix([0, 1, 0], inst['bbox3d'][6])

                vectors = np.diag(np.array(inst['bbox3d'][3:6]) / 2.).dot(R_mat.T)
                box_actor = self.get_bbox_line_actor(centroid, vectors, self.cls_palette[inst['category_id']]*255, 1., 6)
                box_actor.GetProperty().SetInterpolationToPBR()
                renderer.AddActor(box_actor)

                # draw class text
                text_actor = self.add_text(tuple(centroid + [0, vectors[1, 1] + 0.2, 0]),
                                           self.class_names[inst['category_id']], scale=0.15)
                text_actor.SetCamera(renderer.GetActiveCamera())
                renderer.AddActor(text_actor)

                # draw orientations
                color = [[1, 0, 0], [0, 1, 0], [0., 0., 1.]]

                for index in range(vectors.shape[0]):
                    arrow_actor = self.set_arrow_actor(centroid, vectors[index])
                    arrow_actor.GetProperty().SetColor(color[index])
                    renderer.AddActor(arrow_actor)

        # draw layout boxes.
        if 'layout_box' in kwargs['type']:
            layout_box = self.layout_box
            floor_center, x_vec, y_vec, z_vec = layout_box[:3], layout_box[3:6], layout_box[6:9], layout_box[9:12]
            centroid = floor_center + y_vec/2
            vectors = np.array([x_vec, y_vec/2, z_vec])
            box_actor = self.get_bbox_line_actor(centroid, vectors, [125, 125, 125], 1., 6)
            box_actor.GetProperty().SetInterpolationToPBR()
            renderer.AddActor(box_actor)

            # draw orientations
            color = [[1, 0, 0], [0, 1, 0], [0., 0., 1.]]

            for index in range(vectors.shape[0]):
                arrow_actor = self.set_arrow_actor(centroid, vectors[index])
                arrow_actor.GetProperty().SetColor(color[index])
                renderer.AddActor(arrow_actor)

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


class VIS_3DFRONT_RESULT(VIS_BASE):
    def __init__(self, cam_K, cam_Ts, box2ds, box3ds, masks, mesh_files, category_ids, class_names, **kwargs):
        super(VIS_3DFRONT_RESULT, self).__init__()
        self._cam_K = cam_K
        self.cam_Ts = cam_Ts
        self.box2ds = box2ds
        self.box3ds = box3ds
        self.masks = masks
        self.mesh_files = mesh_files
        self.class_ids = category_ids
        self.class_names = [class_names[cls_id] for cls_id in category_ids]
        self.cls_palette = np.array(sns.color_palette('hls', len(class_names)))
        self.color_maps = np.array(kwargs.get('color_maps', None))
        self.projected_inst_boxes = kwargs.get('projected_inst_boxes', None)

    def draw_box2d(self):
        masked_images = 255*np.ones_like(self.color_maps)
        inst_maps = []
        width = 10
        for im_id in range(len(masked_images)):
            box2ds_per_img = self.box2ds[im_id]
            source_img = Image.fromarray(masked_images[im_id]).convert("RGB")
            img_draw = ImageDraw.Draw(source_img)
            for inst_id, box2d in enumerate(box2ds_per_img):
                color = tuple(np.int32(self.cls_palette[self.class_ids[inst_id]] * 255))
                img_draw.rectangle(box2d, outline=color, width=width)
            inst_maps.append(np.array(source_img))
        image_grid(inst_maps).show()

    def draw_mask(self):
        pred_masks_rgb = -np.ones(shape=(*self.masks.shape, 3))
        for inst_id, cls_id in enumerate(self.class_ids):
            color = self.cls_palette[cls_id]
            pred_masks_rgb[self.masks == inst_id] = color
        pred_masks_rgb[self.masks == -1] = np.array([1, 1, 1])

        ratio = 4
        output_masks = []
        for view_id, im in enumerate(pred_masks_rgb):
            output_masks.append(255*cv2.resize(im, (im.shape[1] * ratio, im.shape[0] * ratio), interpolation=cv2.INTER_NEAREST_EXACT))
        pred_masks_rgb = np.array(output_masks).astype(np.uint8)
        image_grid(pred_masks_rgb).show()

    def draw_box2d_from_3d(self):
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
        image_grid(inst_maps).show()

    def set_render(self, *args, **kwargs):
        renderer = vtk.vtkRenderer()
        renderer.ResetCamera()

        '''draw world system'''
        # renderer.AddActor(self.set_axes_actor())

        view_id = kwargs['view_id'] if 'view_id' in kwargs else 0

        render_cam_pose = self.cam_Ts[view_id] if len(self.cam_Ts.shape) > 2 else self.cam_Ts

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

        '''draw class lookup table'''
        if 'lookup_class' in kwargs['type']:
            scalar_bar_actor = self.set_scalar_bar_actor(self.class_names, [self.cls_palette[idx] for idx in self.class_ids])
            renderer.AddActor(scalar_bar_actor)

        '''draw instance bboxes and meshes'''
        for cls_id, cls_name, box3d, mesh_file in zip(self.class_ids, self.class_names, self.box3ds, self.mesh_files):
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
            if mesh_file.endswith('.obj'):
                obj_actor = self.get_obj_actor(mesh_file)
            elif mesh_file.endswith('.ply'):
                obj_actor = self.set_actor(self.set_mapper(self.set_ply_property(mesh_file), 'model'))
            else:
                raise ValueError('Undefined model type.')

            obj_actor.GetProperty().SetColor(self.cls_palette[cls_id])
            obj_actor.GetProperty().SetInterpolationToPBR()
            renderer.AddActor(obj_actor)

            # # draw orientations
            # color = [[1, 0, 0], [0, 1, 0], [0., 0., 1.]]
            #
            # for index in range(vectors.shape[0]):
            #     arrow_actor = self.set_arrow_actor(centroid, vectors[index])
            #     arrow_actor.GetProperty().SetColor(color[index])
            #     renderer.AddActor(arrow_actor)

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


class VIS_3DFRONT_2D_RESULT(object):
    def __init__(self, insts, class_names):
        super(VIS_3DFRONT_2D_RESULT, self).__init__()
        self.insts = insts
        self.class_names = class_names
        self.cls_palette = np.array(sns.color_palette('hls', len(class_names)))

    def draw_inst_maps(self, image_size):
        n_view = len(self.insts)
        template_size = (n_view, int(image_size[1]), int(image_size[0]), 3)
        masked_images = 255 * np.ones(shape=template_size, dtype=np.uint8)
        font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeSans.ttf", 25, encoding="unic")
        inst_maps = []

        for im_id in range(len(masked_images)):
            insts_per_view = self.insts[im_id]
            source_img = Image.fromarray(masked_images[im_id]).convert("RGB")
            img_draw = ImageDraw.Draw(source_img)
            # Number of instances
            if not len(insts_per_view):
                print("\n*** No instances to display *** \n")
                continue
            for box2d, cls_id in zip(insts_per_view['box2ds'], insts_per_view['category_ids']):
                color = tuple(np.array(self.cls_palette[cls_id]*255, dtype=np.uint8))
                x_min, y_min, width, height = box2d
                x_max = x_min + width - 1
                y_max = y_min + height - 1
                img_draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=3)
                img_draw.text((x_min, y_min), self.class_names[cls_id], font=font, fill='white',
                              stroke_width=3, stroke_fill='black')
            inst_maps.append(np.array(source_img))
        image_grid(inst_maps).show()