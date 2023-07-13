#  Copyright (c) 7.2022. Yinyu Nie
#  License: MIT

import numpy as np
from utils import Data_Process_Config
from utils.scannet.tools.scannet_utils import read_label_mapping
import pickle


NYUCLASSES = [
    'void',
    'bathtub', 'bed', 'bookshelf', 'cabinet', 'chair',
    'counter', 'desk', 'dresser', 'lamp', 'night stand',
    'refridgerator', 'shelves', 'sink', 'sofa', 'table',
    'television', 'toilet', 'whiteboard']


class ScanNet_Config(Data_Process_Config):
    def __init__(self, dataset_name='ScanNet', proj_dir='.', split_dir='splits'):
        super(ScanNet_Config, self).__init__(dataset_name, proj_dir)
        self.scannet_scans_dir = self.root_path.joinpath('scans')
        self.split_root = self.root_path.joinpath(split_dir)
        self.split_files = {'train': self.split_root.joinpath('scannetv2_train.txt'),
                            'val': self.split_root.joinpath('scannetv2_val.txt')}
        self.image_size = (1296, 968)
        self.max_frames_per_scene = 100

        all_scene_names = []
        for split_file in self.split_files.values():
            all_scene_names += list(np.loadtxt(split_file, dtype=str))

        all_scene_names = sorted(all_scene_names)

        self.scene_paths = [self.scannet_scans_dir.joinpath(scene_name) for scene_name in all_scene_names]
        self.dump_dir_to_samples = self.root_path.joinpath('ScanNet_samples')

        label_type = 'nyu40class'
        self._label_names = NYUCLASSES

        self._label_mapping = self.read_mapping(label_from='raw_category', label_to=label_type)
        self._id_mapping = self.read_mapping(label_from='id', label_to=label_type)

        self.min_bbox_edge_len = 4 # bboxes with min edge length <= it will be discarded.

        if not self.dump_dir_to_samples.exists():
            self.dump_dir_to_samples.mkdir(parents=True)

    @property
    def label_names(self):
        return self._label_names

    @property
    def label_mapping(self):
        return self._label_mapping

    @property
    def id_mapping(self):
        return self._id_mapping

    def read_mapping(self, label_from='raw_category', label_to='nyu40class'):
        LABEL_MAP_FILE = self.root_path.joinpath('scannetv2-labels.combined.tsv')

        map_file = self.root_path.joinpath(label_from + '_to_' + label_to + '.pkl')
        if not map_file.exists():
            raw_map = read_label_mapping(LABEL_MAP_FILE, label_from=label_from, label_to=label_to)
            label_mapping = {}
            for key, item in raw_map.items():
                if item in self.label_names:
                    label_mapping[key] = self.label_names.index(item)
                else:
                    label_mapping[key] = 0

            with open(map_file, 'wb') as file:
                pickle.dump(label_mapping, file)

        with open(map_file, 'rb') as file:
            label_mapping = pickle.load(file)

        return label_mapping


if __name__ == '__main__':
    dataset_config = ScanNet_Config()
    print(dataset_config.scene_paths)


