#  Copyright (c) 7.2022. Yinyu Nie
#  License: MIT

''' Ref: https://github.com/ScanNet/ScanNet/blob/master/BenchmarkScripts '''
import os
import sys
import csv
from collections import defaultdict
import imageio
from utils.scannet.tools.load_scannet_data import export
from typing import Dict
from utils.tools import project_points_to_2d, get_box_corners
from utils.scannet.tools.load_scannet_data import load_transform_matrix

try:
    import numpy as np
except:
    print("Failed to import numpy package.")
    sys.exit(-1)

try:
    from plyfile import PlyData, PlyElement
except:
    print("Please install the module 'plyfile' for PLY i/o, e.g.")
    print("pip install plyfile")
    sys.exit(-1)

def represents_int(s):
    ''' if string s represents an int. '''
    try: 
        int(s)
        return True
    except ValueError:
        return False


def read_label_mapping(filename, label_from='raw_category', label_to='nyu40id'):
    assert os.path.isfile(filename)
    mapping = dict()
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:
            mapping[row[label_from]] = row[label_to]
    if represents_int(list(mapping.keys())[0]):
        mapping = {int(k):v for k,v in mapping.items()}
    return mapping

def read_mesh_vertices(filename):
    """ read XYZ for each vertex.
    """
    assert os.path.isfile(filename)
    with open(filename, 'rb') as f:
        plydata = PlyData.read(f)
        num_verts = plydata['vertex'].count
        vertices = np.zeros(shape=[num_verts, 3], dtype=np.float32)
        vertices[:,0] = plydata['vertex'].data['x']
        vertices[:,1] = plydata['vertex'].data['y']
        vertices[:,2] = plydata['vertex'].data['z']
    return vertices

def read_mesh_vertices_rgb(filename):
    """ read XYZ RGB for each vertex.
    Note: RGB values are in 0-255
    """
    assert os.path.isfile(filename)
    with open(filename, 'rb') as f:
        plydata = PlyData.read(f)
        num_verts = plydata['vertex'].count
        vertices = np.zeros(shape=[num_verts, 6], dtype=np.float32)
        vertices[:,0] = plydata['vertex'].data['x']
        vertices[:,1] = plydata['vertex'].data['y']
        vertices[:,2] = plydata['vertex'].data['z']
        vertices[:,3] = plydata['vertex'].data['red']
        vertices[:,4] = plydata['vertex'].data['green']
        vertices[:,5] = plydata['vertex'].data['blue']
    return vertices

def read_2d_info(scene_dir, labels=('pose'), max_frames=int(1e5)):
    assert 'pose' in labels

    pose_dir = scene_dir.joinpath('pose')
    color_dir = scene_dir.joinpath('color')
    instance_dir = scene_dir.joinpath('instance-filt')
    semantic_dir = scene_dir.joinpath('label-filt')
    depth_dir = scene_dir.joinpath('depth')
    intrinsic_color_file = scene_dir.joinpath('intrinsic/intrinsic_color.txt')
    intrinsic_depth_file = scene_dir.joinpath('intrinsic/intrinsic_depth.txt')
    extrinsic_color_file = scene_dir.joinpath('intrinsic/extrinsic_color.txt')
    extrinsic_depth_file = scene_dir.joinpath('intrinsic/extrinsic_depth.txt')

    frame_ids = sorted([int(file.name[:-4]) for file in pose_dir.iterdir()])
    if max_frames < len(frame_ids):
        selected_ids = np.linspace(0, len(frame_ids) - 1, max_frames).round().astype(np.uint16)
        frame_ids = [frame_ids[idx] for idx in selected_ids]

    frames_data = []
    for frame_id in frame_ids:
        outputs = defaultdict(lambda: None)

        frame_idx = str(frame_id)
        outputs['frame_idx'] = frame_idx

        if 'pose' in labels:
            pose_file = pose_dir.joinpath(frame_idx + '.txt')
            pose = np.loadtxt(pose_file).astype(float)
            if float('-inf') in pose or float('inf') in pose:
                print("Frame: %s in Scene: %s in invalid." % (frame_idx, scene_dir.name))
                continue
            outputs['pose'] = pose

        if 'color' in labels:
            color_file = color_dir.joinpath(frame_idx + '.jpg')
            color = imageio.imread(color_file)
            outputs['color'] = color

        if 'instance' in labels:
            instance_file = instance_dir.joinpath(frame_idx + '.png')
            instance_map = imageio.imread(instance_file)
            outputs['instance_map'] = instance_map

        if 'semantic' in labels:
            semantic_file = semantic_dir.joinpath(frame_idx + '.png')
            semantic_map = imageio.imread(semantic_file)
            outputs['semantic_map'] = semantic_map

        if 'depth' in labels:
            depth_file = depth_dir.joinpath(frame_idx + '.png')
            depth = np.asarray(imageio.imread(depth_file), dtype=np.int64) / 1000.
            outputs['depth'] = depth

        if 'intrinsic_color' in labels:
            intrinsic_color = np.loadtxt(intrinsic_color_file).astype(float)
            outputs['intrinsic_color'] = intrinsic_color

        if 'intrinsic_depth' in labels:
            intrinsic_depth = np.loadtxt(intrinsic_depth_file).astype(float)
            outputs['intrinsic_depth'] = intrinsic_depth

        if 'extrinsic_color' in labels:
            extrinsic_color = np.loadtxt(extrinsic_color_file).astype(float)
            outputs['extrinsic_color'] = extrinsic_color

        if 'extrinsic_depth' in labels:
            extrinsic_depth = np.loadtxt(extrinsic_depth_file).astype(float)
            outputs['extrinsic_depth'] = extrinsic_depth

        frames_data.append(outputs)

    return frames_data

def read_scenetype(txt_file):
    scene_type = 'null'
    with open(txt_file, 'r') as file:
        for line in file:
            if line.startswith('sceneType'):
                scene_type = line.split('=')[-1].strip()
                break
    scene_type = scene_type.replace('/', 'or').replace(' ', '_')
    return scene_type


def read_3d_info(scene_dir, dataset_config):
    scene_name = scene_dir.name
    mesh_file = scene_dir.joinpath(scene_name + '_vh_clean_2.ply')
    agg_file = scene_dir.joinpath(scene_name + '.aggregation.json')
    seg_file = scene_dir.joinpath(scene_name + '_vh_clean_2.0.010000.segs.json')
    meta_file = scene_dir.joinpath(scene_name + '.txt')  # includes axis
    label_mapping = dataset_config.label_mapping
    vertices, semantic_labels, instance_labels, instance_bboxes, instance2semantic = \
        export(mesh_file, agg_file, seg_file, meta_file, label_mapping, None)

    scan_data = {'vertices': vertices,
                 'semantic_labels': semantic_labels,
                 'instance_labels': instance_labels,
                 'instance_bboxes': instance_bboxes,
                 'instance2semantic': instance2semantic,
                 'scene_name': scene_name}

    return scan_data

def label_mapping_2D(img: np.ndarray, mapping_dict: Dict):
    '''To map the labels in img following the rule in mapping_dict.'''
    out_img = np.zeros_like(img)
    existing_labels = np.unique(img)
    for label in existing_labels:
        if label == 0:
            continue
        out_img[img==label] = mapping_dict[label]
    return out_img

def project_insts_to_2d(inst_info, cam_K, cam_T):
    '''project vertices of 3D bboxes to image plane.'''
    if len(cam_T.shape) == 3:
        cam_T = cam_T[0]
    projected_boxes = []
    for inst in inst_info:
        if not isinstance(inst['bbox3d'], np.ndarray):
            projected_boxes.append(None)
        elif not np.issubdtype(inst['bbox3d'].dtype, np.floating):
            projected_boxes.append(None)
        else:
            centroid = inst['bbox3d'][:3]
            vectors = np.diag(np.array(inst['bbox3d'][3:6]) / 2.)
            box_corners = np.array(get_box_corners(centroid, vectors))

            pixels = project_points_to_2d(box_corners, cam_K, cam_T)
            projected_boxes.append(pixels[:, :2])
    return projected_boxes

def process_cam_pose(cam_T, axis_align_matrix):
    cam_center = cam_T[:, 3].dot(axis_align_matrix.transpose())
    cam_pose = axis_align_matrix[:3, :3].dot(cam_T[:3, :3])
    cam_pose[:, 1] *= -1
    cam_pose[:, 2] *= -1
    cam_T[:3, :3] = cam_pose
    cam_T[:, 3] = cam_center
    return cam_T

def process_data(frames_data, scan_data, scene_dir, dataset_config):
    instance_3d_bboxes = scan_data['instance_bboxes']
    instance2semantic = scan_data['instance2semantic']

    axis_align_matrix = load_transform_matrix(meta_file=scene_dir.joinpath(scan_data['scene_name'] + '.txt'))

    colors = []
    cam_Ks = []
    cam_Ts = []
    class_maps = []
    instance_attrs = []
    projected_inst_boxes = []
    for frame_data in frames_data:
        color = frame_data['color']
        cam_K = frame_data['intrinsic_color'][:3, :3]
        cam_T = frame_data['pose']
        cam_T = process_cam_pose(cam_T, axis_align_matrix)
        class_segmap = frame_data['semantic_map']
        instance_segmap = frame_data['instance_map']
        class_segmap = label_mapping_2D(class_segmap, dataset_config.id_mapping)

        inst_marks = np.unique(instance_segmap)

        inst_info = []
        for inst_mark in inst_marks:
            if inst_mark == 0: continue
            category_id = instance2semantic[inst_mark]

            if category_id == 0:
                continue

            # get 2D mask
            inst_mask = instance_segmap == inst_mark

            # get 2D bbox
            mask_mat = np.argwhere(inst_mask)
            y_min, x_min = mask_mat.min(axis=0)
            y_max, x_max = mask_mat.max(axis=0)
            bbox = [x_min, y_min, x_max-x_min+1, y_max-y_min+1] # [x,y,width,height]
            if min(bbox[2:]) <= dataset_config.min_bbox_edge_len:
                continue

            inst_dict = dict()
            inst_dict['category_id'] = category_id
            inst_dict['mask'] = inst_mask[y_min:y_max + 1, x_min:x_max + 1]
            inst_dict['bbox2d'] = bbox

            # get 3D bbox
            inst_dict['bbox3d'] = instance_3d_bboxes[inst_mark-1]
            inst_dict['inst_mark'] = inst_mark
            inst_info.append(inst_dict)

        '''Project objects to original cam poses'''
        projected_box2d_list = project_insts_to_2d(inst_info, cam_K, cam_T)

        # '''Backproject image center to floor'''
        # lambda_y = cam_T[1, 3] / cam_T[1, 2]
        # cam_floor_center = cam_T[:3, 3] - lambda_y * cam_T[:3, 2]

        colors.append(color)
        cam_Ts.append(cam_T)
        cam_Ks.append(cam_K)
        class_maps.append(class_segmap)
        instance_attrs.append(inst_info)
        projected_inst_boxes.append(projected_box2d_list)

    return colors, cam_Ts, cam_Ks, class_maps, instance_attrs, projected_inst_boxes