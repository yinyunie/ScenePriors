#  Copyright (c) 8.2022. Yinyu Nie
#  License: MIT
import h5py
import argparse

from utils.scannet import ScanNet_Config
from utils.scannet.vis.vis_classes import Vis_ScanNet_GT, VIS_ScanNet_2D
from utils.scannet.tools.scannet_utils import project_insts_to_2d


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize a ScanNet gt sample.")
    parser.add_argument("--scene_id", type=str, default='scene0000_00',
                        help="give the scene json name to visualize.")
    parser.add_argument("--n_samples", type=int, default=20,
                        help="Max number of images to visualize.")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    dataset_config = ScanNet_Config()

    '''load data'''
    cam_Ks = []
    cam_Ts = []
    room_imgs = []
    instance_attrs = []
    class_maps = []
    projected_inst_boxes = []
    for sample_file in dataset_config.dump_dir_to_samples.rglob('%s*.hdf5' % (args.scene_id)):
        '''read data'''
        with h5py.File(sample_file, "r") as sample_data:
            room_uid = sample_data['room_uid'][0].decode("utf-8")
            room_type = sample_data['room_type'][0].decode("utf-8")
            colors = sample_data['colors'][:]
            cam_T = sample_data['cam_T'][:]
            cam_K = sample_data['cam_K'][:]
            class_segmap = sample_data['class_segmap'][:]
            inst_h5py = sample_data['inst_info']
            inst_info = []
            for inst_id in inst_h5py:
                inst = {}
                inst['bbox2d'] = inst_h5py[inst_id]['bbox2d'][:]
                inst['bbox3d'] = inst_h5py[inst_id]['bbox3d'][:]
                inst['category_id'] = inst_h5py[inst_id]['category_id'][0]
                inst['inst_mark'] = inst_h5py[inst_id]['inst_mark'][0]
                inst['mask'] = inst_h5py[inst_id]['mask'][:]
                inst_info.append(inst)

        '''Project objects to original cam poses'''
        projected_box2d_list = project_insts_to_2d(inst_info, cam_K, cam_T)

        cam_Ts.append(cam_T)
        cam_Ks.append(cam_K)
        room_imgs.append(colors)
        instance_attrs.append(inst_info)
        class_maps.append(class_segmap)
        projected_inst_boxes.append(projected_box2d_list)

        if len(cam_Ts) >= args.n_samples:
            break

    viser_2D = VIS_ScanNet_2D(color_maps=room_imgs, inst_info=instance_attrs, cls_maps=class_maps,
                              class_names=dataset_config.label_names, projected_inst_boxes=projected_inst_boxes)
    # viser_2D.draw_colors()
    viser_2D.draw_cls_maps()
    viser_2D.draw_inst_maps(type=('mask'))
    viser_2D.draw_box2d_from_3d()

    scene = Vis_ScanNet_GT(
        cam_Ks=cam_Ks,
        cam_Ts=cam_Ts,
        instance_attrs=instance_attrs,
        class_names=dataset_config.label_names)
    scene.update_view(view_id=1)
    scene.visualize(type=['bbox', 'mesh', 'cam_pose'], image_size=room_imgs[0].shape[:2])