#  Process data for training and evaluation.
#  Copyright (c) 2.2022. Yinyu Nie
#  License: MIT
import sys
sys.path.append('.')
import argparse
import h5py
import numpy as np
import json
from multiprocessing import Pool
from utils.threed_front import Threed_Front_Config
from utils.threed_front.tools.threed_front import ThreedFront
from utils.tools import label_mapping_2D, normalize, heading2arc
from utils.threed_front.tools.utils import parse_inst_from_3dfront, get_room_uid_from_rendering
from utils.tools import R_from_pitch_yaw_roll, write_data_to_hdf5


def parse_args():
    parser = argparse.ArgumentParser(description="Process for training")
    parser.add_argument("--room_type",
                        default="living",
                        choices=["bed", "living", "dining", "library"],
                        help="The type of dataset filtering to be used.")
    parser.add_argument("--n_processes", type=int, default=4,
                        help="Number of processes to generate samples.")
    return parser.parse_args()


def per_run(render_path):
    # read rendering info from blender output
    print('Processing %s.' % str(render_path))
    try:
        with h5py.File(render_path) as f:
            colors = np.array(f["colors"])[:, ::-1]
            cam_T = np.array(f["cam_Ts"])
            class_segmap = np.array(f["class_segmaps"])[:, ::-1]
            instance_segmap = np.array(f["instance_segmaps"])[:, ::-1]
            instance_attribute_mapping = json.loads(f["instance_attribute_maps"][()])
    except:
        print('Unable to open %s.' % str(render_path))
        return None

    ### get scene_name
    scene_json = render_path.parent.name

    #### class mapping
    class_segmap = label_mapping_2D(class_segmap, dataset_config.label_mapping)

    #### get instance info
    inst_marks = set([inst['inst_mark'] for inst in instance_attribute_mapping if
                      inst['inst_mark'] != '' and 'layout' not in inst['inst_mark']])

    inst_info = []
    for inst_mark in inst_marks:
        parts = [part for part in instance_attribute_mapping if part['inst_mark'] == inst_mark]

        # remove background objects.
        category_id = dataset_config.label_mapping[parts[0]['category_id']]

        # get 2D masks
        part_indices = [part['idx'] for part in parts]
        inst_mask = np.sum([instance_segmap == idx for idx in part_indices], axis=0, dtype=bool)

        # get 2D bbox
        mask_mat = np.argwhere(inst_mask)
        y_min, x_min = mask_mat.min(axis=0)
        y_max, x_max = mask_mat.max(axis=0)
        bbox = [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1]  # [x,y,width,height]
        if min(bbox[2:]) <= dataset_config.min_bbox_edge_len:
            continue

        inst_dict = {key: parts[0][key] for key in ['inst_mark', 'uid', 'jid', 'room_id']}
        inst_dict['category_id'] = category_id
        inst_dict['mask'] = inst_mask[y_min:y_max + 1, x_min:x_max + 1]
        inst_dict['bbox2d'] = bbox

        # get 3D bbox
        inst_rm_uid = "_".join([scene_json, inst_dict['room_id']])
        inst_3d_info = parse_inst_from_3dfront(parts[0], d.rooms, inst_rm_uid)
        inst_dict = {**inst_dict, **inst_3d_info, **{'room_uid': inst_rm_uid}}

        inst_info.append(inst_dict)

    # pass renderings if no objects detected in a rendering.
    if not len(inst_info):
        print('No instances in %s.' % str(render_path))
        return None

    # get the platform plane for rendering from various poses
    # get the room_id of the current rendering
    target_room_uid = get_room_uid_from_rendering(inst_info)

    output_file = output_dir.joinpath(target_room_uid + '_' + render_path.name)
    if output_file.exists():
        print('File already exists.')
        return None

    # get target room
    target_room = next((rm for rm in d.rooms if rm.uid == target_room_uid), None)

    if (target_room is None) or (args.room_type not in target_room.room_type):
        print('Not the room type we ask for.')
        return None

    # for bedroom, remove those rooms with a bed
    if args.room_type == 'bed':
        from utils.threed_front.base import THREED_FRONT_BEDROOM_FURNITURE
        all_furniture_in_room = [THREED_FRONT_BEDROOM_FURNITURE.get(key, 'none').lower() for key in target_room.furniture_in_room]
        if True not in ('bed' in item for item in all_furniture_in_room):
            print('There is no bed in this bedroom. maybe wrong gt.')
            return None

    # remove objects that not located in the same room
    inst_info = [inst for inst in inst_info if inst['room_uid'] == target_room_uid and inst['category_id'] != 0]

    # pass renderings if no objects detected in the target room.
    if not len(inst_info):
        print('No instances in %s.' % str(render_path))
        return None

    # process cam_T from blender to ours
    cam_T = dataset_config.blender2opengl_cam(cam_T)

    '''make all objects and camera poses in a room, move to the room reference system'''
    layout_box = np.copy(target_room.layout_box)
    floor_center = np.copy(layout_box[:3])
    room_heading = normalize(layout_box[9:12])
    room_heading_angle = heading2arc(room_heading)[0]
    rot_mat = R_from_pitch_yaw_roll(0, room_heading_angle, 0)[0]

    # transform layout
    layout_box[:3] -= floor_center
    layout_box[3:6] = np.dot(layout_box[3:6], rot_mat)
    layout_box[6:9] = np.dot(layout_box[6:9], rot_mat)
    layout_box[9:12] = np.dot(layout_box[9:12], rot_mat)

    # transform camera poses
    cam_T[:3, 3] -= floor_center
    cam_T[:3, 3] = np.dot(cam_T[:3, 3], rot_mat)
    cam_T[:3, :3] = np.dot(rot_mat.T, cam_T[:3, :3])

    # transform objects
    for inst in inst_info:
        if inst['bbox3d'] is not None:
            inst['bbox3d'][6] -= room_heading_angle
            inst['bbox3d'][:3] -= floor_center
            inst['bbox3d'][:3] = np.dot(inst['bbox3d'][:3], rot_mat)

    '''Export sample'''
    file_handle = h5py.File(output_file, "w")
    write_data_to_hdf5(file_handle, name='room_type', data=target_room.room_type)
    write_data_to_hdf5(file_handle, name='room_uid', data=target_room.uid)
    write_data_to_hdf5(file_handle, name='colors', data=colors)
    write_data_to_hdf5(file_handle, name='layout_box', data=layout_box)
    write_data_to_hdf5(file_handle, name='cam_T', data=cam_T)
    write_data_to_hdf5(file_handle, name='class_segmap', data=class_segmap)
    write_data_to_hdf5(file_handle, name='inst_info', data=inst_info)


if __name__ == '__main__':
    args = parse_args()
    # initialize category labels and mapping dict for specific room type.
    dataset_config = Threed_Front_Config()
    dataset_config.init_generic_categories_by_room_type(args.room_type)

    '''Read 3D-Front Data'''
    d = ThreedFront.from_dataset_directory(
        str(dataset_config.threed_front_dir),
        str(dataset_config.model_info_path),
        str(dataset_config.threed_future_dir),
        str(dataset_config.dump_dir_to_scenes),
        path_to_room_masks_dir=None,
        path_to_bounds=None,
        filter_fn=lambda s: s)
    print(d)

    output_dir = dataset_config.dump_dir_to_samples.joinpath(args.room_type)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    all_rendering_files = list(dataset_config.threed_front_rendering_dir.rglob('*.hdf5'))

    p = Pool(processes=args.n_processes)
    stats = p.map(per_run, all_rendering_files)
    p.close()
    p.join()