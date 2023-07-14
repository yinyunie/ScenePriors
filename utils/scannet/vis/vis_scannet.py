#  Copyright (c) 7.2022. Yinyu Nie
#  License: MIT

import argparse

from utils.scannet import ScanNet_Config
from utils.scannet.vis.vis_classes import VIS_ScanNet, VIS_ScanNet_2D
from utils.scannet.tools.scannet_utils import read_2d_info, read_3d_info, process_data
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize a ScanNet scene.")
    parser.add_argument("--scene_id", type=str, default='scene0001_00',
                        help="Please give the scene id for visualization.")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    dataset_config = ScanNet_Config()
    scene_dir = Path('datasets/ScanNet/scans').joinpath(args.scene_id)

    labels_to_read = ('color', 'pose', 'instance', 'semantic', 'intrinsic_color')

    frames_data = read_2d_info(scene_dir, labels=labels_to_read, max_frames=100)
    scan_data = read_3d_info(scene_dir, dataset_config=dataset_config)

    colors, cam_Ts, cam_Ks, class_maps, instance_attrs, projected_inst_boxes = process_data(frames_data, scan_data,
                                                                                            scene_dir, dataset_config)

    viser_2D = VIS_ScanNet_2D(color_maps=colors, inst_info=instance_attrs, cls_maps=class_maps,
                              class_names=dataset_config.label_names, projected_inst_boxes=projected_inst_boxes)
    # viser_2D.draw_colors()
    # viser_2D.draw_cls_maps()
    # viser_2D.draw_inst_maps(type=('mask'))
    # viser_2D.draw_box2d_from_3d()

    scene = VIS_ScanNet(
        cam_Ks=cam_Ks,
        cam_Ts=cam_Ts,
        scene_dir=scene_dir,
        scan_data=scan_data, class_names=dataset_config.label_names)
    scene.update_view(view_id=10)
    scene.visualize(type=['cam_pose', 'bbox', 'pointcloud'], image_size=colors[0].shape[:2])
