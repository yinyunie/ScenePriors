#  Copyright (c) 2.2022. Yinyu Nie
#  License: MIT
import logging
import os
from net_utils.distributed import is_master_proc
from utils.tools import read_json
from collections import defaultdict


class CONFIG(object):
    def __init__(self, config):
        self.config = config
        self.is_master = is_master_proc(config.distributed.num_gpus)
        aug_num = 4 if config.data.aug else 1
        if config.data.dataset == '3D-Front':
            from utils.threed_front import Threed_Front_Config
            dataset_config = Threed_Front_Config(dataset_name=config.data.dataset,
                                                 proj_dir=config.root_dir,
                                                 split_dir=config.data.split_dir)
            dataset_config.init_generic_categories_by_room_type(config.data.split_type)
            split_path = str(dataset_config.split_path_dir[config.data.split_type])
            self.label_names = dataset_config.label_names
            self.cam_K = dataset_config.cam_K
            self.image_size = dataset_config.image_size
            self.max_n_obj = {'bed': 15, 'living': 28}[config.data.split_type] # living: 28 for only living room
            split_data = read_json(split_path)
            # write augmentation tag in split_data
            split_data_aug = defaultdict(dict)
            for mode in split_data:
                for scene_name in split_data[mode]:
                    if mode == 'train':
                        for aug_idx in range(aug_num):
                            split_data_aug[mode]['%s_aug_%d' % (scene_name, aug_idx)] = split_data[mode][scene_name]
                    else:
                        split_data_aug[mode]['%s_aug_%d' % (scene_name, 0)] = split_data[mode][scene_name]
            self.split_data = split_data_aug
            self.room_uids = {mode: sorted(set(rm_name for rm_name in split)) for mode, split in self.split_data.items()}
            self.unique_inst_mark = read_json(dataset_config.unique_inst_mark_path)
            self.room_types = ['bed', 'dining', 'library', 'living']

            if config.mode == 'demo':
                from pathlib import Path
                self.demo_samples = list((Path(config.demo.input_dir).iterdir()))

        elif config.data.dataset == 'ScanNet':
            from utils.scannet import ScanNet_Config
            dataset_config = ScanNet_Config(
                dataset_name=config.data.dataset,
                proj_dir=config.root_dir,
                split_dir=config.data.split_dir)
            split_path = dataset_config.split_root.joinpath(config.data.split_type + '_split.json')
            self.label_names = dataset_config.label_names
            self.image_size = dataset_config.image_size
            self.max_n_obj = 53
            split_data = read_json(split_path)
            # write augmentation tag in split_data
            split_data_aug = defaultdict(dict)
            for mode in split_data:
                for scene_name in split_data[mode]:
                    if mode == 'train':
                        for aug_idx in range(aug_num):
                            split_data_aug[mode]['%s_aug_%d' % (scene_name, aug_idx)] = split_data[mode][scene_name]
                    else:
                        split_data_aug[mode]['%s_aug_%d' % (scene_name, 0)] = split_data[mode][scene_name]

            self.split_data = split_data_aug
            self.room_uids = {
                mode: sorted(set(rm_name for rm_name in split)) for
                mode, split in self.split_data.items()}
            self.room_types = ['Apartment', 'Bathroom', 'Bedroom_or_Hotel', 'Bookstore_or_Library', 'Classroom',
                               'Closet', 'ComputerCluster', 'Conference_Room', 'CopyorMail_Room', 'Dining_Room',
                               'Game_room', 'Gym', 'Hallway', 'Kitchen', 'Laundry_Room', 'Living_room_or_Lounge',
                               'Lobby', 'Misc.', 'Office', 'Stairs', 'StorageorBasementorGarage']
            if config.mode == 'demo':
                from pathlib import Path
                test_img_paths = read_json(os.path.join(config.root_dir, 'datasets/ScanNet/total3d_test.json'))
                test_filenames = [path.split('/')[-1][:-4] for path in test_img_paths]
                test_scene_names = list(set(['_'.join(filename.split('_')[:2]) for filename in test_filenames]))
                scene_roots = {scene_name:'/'.join(split_data['test'][scene_name][0].split('/')[:-1]) for scene_name in test_scene_names}
                test_samples = [scene_roots['_'.join(filename.split('_')[:2])] + '/%s.hdf5' % (filename) for filename in test_filenames]
                self.demo_samples = [Path(os.path.join(config.root_dir, sample)) for sample in test_samples if os.path.exists(os.path.join(config.root_dir, sample))]
                # self.demo_samples = list((Path(config.demo.input_dir).iterdir()))

    def info(self, content):
        if self.is_master:
            logging.info(content)

