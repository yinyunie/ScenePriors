#  Copyright (c) 7.2022. Yinyu Nie
#  License: MIT
import sys
sys.path.append('.')
import numpy as np
import h5py
from copy import deepcopy

from utils.scannet import ScanNet_Config
from utils.scannet.tools.scannet_utils import read_2d_info, read_3d_info, process_data, read_scenetype
from utils.tools import write_data_to_hdf5

if __name__ == '__main__':
    dataset_config = ScanNet_Config()
    labels_to_read = ('color', 'pose', 'instance', 'semantic', 'intrinsic_color')

    for scene_dir in dataset_config.scene_paths:
        frames_data = read_2d_info(scene_dir, labels=labels_to_read, max_frames=dataset_config.max_frames_per_scene)
        scan_data = read_3d_info(scene_dir, dataset_config=dataset_config)
        colors, cam_Ts, cam_Ks, class_maps, instance_attrs, projected_inst_boxes = process_data(
            frames_data, scan_data, scene_dir, dataset_config)

        unique_marks = sorted(list(set([inst['inst_mark'] for view in instance_attrs for inst in view])))

        scene_name = scan_data['scene_name']
        scene_type = read_scenetype(scene_dir.joinpath(scene_name + '.txt'))
        output_dir = dataset_config.dump_dir_to_samples.joinpath(scene_type)
        if not output_dir.exists():
            output_dir.mkdir()

        floor_center = np.array([0., scan_data['instance_bboxes'][:, 1].min(), 0.])

        for view_id in range(len(frames_data)):
            frame_id = frames_data[view_id]['frame_idx']

            output_file = output_dir.joinpath(scene_dir.name + '_' + frame_id + '.hdf5')
            if output_file.exists():
                print('File already exists.')
                continue

            color = colors[view_id]
            cam_T = cam_Ts[view_id]
            cam_K = cam_Ks[view_id]
            class_segmap = class_maps[view_id]
            inst_info = deepcopy(instance_attrs[view_id])

            # pass renderings if no objects detected in a rendering.
            if not len(inst_info):
                print('No instances in %s, frame: %s.' % (scene_name, frame_id))
                continue

            '''make all objects and camera poses in a room, move to the room reference system'''
            # transform camera poses
            cam_T[:3, 3] -= floor_center

            # transform objects
            for inst in inst_info:
                inst['bbox3d'][:3] -= floor_center

            '''Export sample'''
            file_handle = h5py.File(output_file, "w")
            write_data_to_hdf5(file_handle, name='room_type', data=scene_type)
            write_data_to_hdf5(file_handle, name='room_uid', data=scene_name)
            write_data_to_hdf5(file_handle, name='colors', data=color)
            write_data_to_hdf5(file_handle, name='image_size', data=(color.shape[1], color.shape[0]))
            write_data_to_hdf5(file_handle, name='cam_T', data=cam_T)
            write_data_to_hdf5(file_handle, name='cam_K', data=cam_K)
            write_data_to_hdf5(file_handle, name='class_segmap', data=class_segmap)
            write_data_to_hdf5(file_handle, name='inst_info', data=inst_info)
            write_data_to_hdf5(file_handle, name='unique_inst_marks', data=unique_marks)
