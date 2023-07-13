#  Copyright (c) 2.2022. Yinyu Nie
#  License: MIT
import h5py
import argparse
from utils.threed_front import Threed_Front_Config
from utils.threed_front.vis.vis_classes import VIS_3DFRONT_2D, VIS_3DFRONT_SAMPLE
from utils.threed_front.tools.utils import project_insts_to_2d


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize a 3D-FRONT gt sample.")
    parser.add_argument("--scene_json", type=str, default='6a0e73bc-d0c4-4a38-bfb6-e083ce05ebe9',
                        help="give the scene json name to visualize.")
    parser.add_argument("--room_id", type=str, default='MasterBedroom-2679',
                        help="give the room_id to visualize.")
    parser.add_argument("--n_samples", type=int, default=12,
                        help="Max number of images to visualize.")
    return parser.parse_args()


def get_ez_room_type(room_type):
    ez_room_types = ['bed', 'living', 'dining', 'library']
    output_ez_type = None
    for es_type in ez_room_types:
        if es_type in room_type:
            output_ez_type = es_type
            break
    return output_ez_type


if __name__ == '__main__':
    args = parse_args()
    dataset_config = Threed_Front_Config()

    '''load data'''
    cam_K = dataset_config.cam_K
    cam_Ts = []
    room_imgs = []
    instance_attrs = []
    class_maps = []
    projected_inst_boxes = []
    layout_boxes = []
    for sample_file in dataset_config.dump_dir_to_samples.rglob('*.hdf5'):
        # filter room_id
        scene_json, room_id = sample_file.name.split('_')[:2]
        if not scene_json == args.scene_json or not room_id == args.room_id:
            continue
        '''read data'''
        with h5py.File(sample_file, "r") as sample_data:
            room_uid = sample_data['room_uid'][0].decode("utf-8")
            room_type = sample_data['room_type'][0].decode("utf-8")
            colors = sample_data['colors'][:]
            layout_box = sample_data['layout_box'][:]
            cam_T = sample_data['cam_T'][:]
            class_segmap = sample_data['class_segmap'][:]
            inst_h5py = sample_data['inst_info']
            inst_info = []
            for inst_id in inst_h5py:
                inst = {}
                inst['bbox2d'] = inst_h5py[inst_id]['bbox2d'][:]
                inst['bbox3d'] = inst_h5py[inst_id]['bbox3d'][:]
                inst['category_id'] = inst_h5py[inst_id]['category_id'][0]
                inst['inst_mark'] = inst_h5py[inst_id]['inst_mark'][0].decode('utf-8')
                inst['uid'] = inst_h5py[inst_id]['uid'][0].decode('utf-8')
                inst['jid'] = inst_h5py[inst_id]['jid'][0].decode('utf-8')
                inst['mask'] = inst_h5py[inst_id]['mask'][:]
                inst['model_path'] = inst_h5py[inst_id]['model_path'][0].decode('utf-8')
                inst['room_id'] = inst_h5py[inst_id]['room_id'][0].decode('utf-8')
                inst_info.append(inst)

        '''Project objects to original cam poses'''
        projected_box2d_list = project_insts_to_2d(inst_info, cam_K, cam_T)

        cam_Ts.append(cam_T)
        room_imgs.append(colors)
        instance_attrs.append(inst_info)
        class_maps.append(class_segmap)
        projected_inst_boxes.append(projected_box2d_list)
        layout_boxes.append(layout_box)

        if len(cam_Ts) >= args.n_samples:
            break
    # initialize category labels and mapping dict for specific room type.
    dataset_config.init_generic_categories_by_room_type(get_ez_room_type(room_type))

    viser_2D = VIS_3DFRONT_2D(color_maps=room_imgs, inst_info=instance_attrs, cls_maps=class_maps,
                              class_names=dataset_config.label_names, projected_inst_boxes=projected_inst_boxes)
    viser_2D.draw_colors()
    viser_2D.draw_cls_maps()
    viser_2D.draw_box2d_from_3d()
    viser_2D.draw_inst_maps(type=('mask'))
    idx = 0
    viser = VIS_3DFRONT_SAMPLE(cam_K=cam_K, cam_Ts=[cam_Ts[idx]], inst_info=[instance_attrs[idx]],
                               layout_boxes=[layout_boxes[idx]], class_names=dataset_config.label_names)
    viser.visualize(view_id=0, type=['mesh', 'bbox', 'layout_box'])