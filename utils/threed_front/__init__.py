#  Copyright (c) 1.2022. Yinyu Nie
#  License: MIT
import math
import numpy as np
from utils import Data_Process_Config
from utils.tools import read_mapping_csv
from utils.threed_front.base import THREED_FRONT_BEDROOM_FURNITURE, THREED_FRONT_LIBRARY_FURNITURE, THREED_FRONT_LIVINGROOM_FURNITURE


class Threed_Front_Config(Data_Process_Config):
    def __init__(self, dataset_name='3D-Front', proj_dir='.', split_dir='splits'):
        super(Threed_Front_Config, self).__init__(dataset_name, proj_dir)
        self.threed_front_dir = self.root_path.joinpath('3D-FRONT')
        self.threed_future_dir = self.root_path.joinpath('3D-FUTURE-model')
        self.model_info_path = self.threed_future_dir.joinpath('model_info_revised.json')
        self.layout_texture_path = self.root_path.joinpath('3D-FRONT-texture')
        self.dump_dir_to_scenes = self.root_path.joinpath('3D-FRONT_scenes')
        self.dump_dir_to_samples = self.root_path.joinpath('3D-FRONT_samples')
        self.dump_dir_to_room_samples = self.root_path.joinpath('3D-FRONT_room_samples')
        self.threed_front_rendering_dir = self.root_path.joinpath('3D-FRONT_renderings_improved_mat')
        self.cls_importance_path = self.root_path.joinpath('cls_importance_ratio.json')
        self.unique_inst_mark_path = self.root_path.joinpath('unique_inst_mark.json')

        self.room_types = ['bed', 'library', 'living', 'dining']

        self.split_root = self.root_path.joinpath(split_dir)
        if not self.split_root.is_dir():
            self.split_root.mkdir()
        self.split_path_dir = {room_type: self.split_root.joinpath('%s_threed_front_splits.json' % (room_type)) for room_type in self.room_types}
        self.split_path_dir['all'] = self.split_root.joinpath('all_samples.json')

        self.split_ratio = {'per_room_type':
                                {'train': 0.9,
                                 'test': 0.1},
                            'all':
                                {'train': 0.9,
                                 'test': 0.1}}
        self.cam_K = np.load(self.threed_front_rendering_dir.joinpath('cam_K.npy'))
        self.image_size = self.cam_K[:2, 2] * 2 + 1
        self.blender_label_mapping_path = self.root_path.joinpath('blender_label_mapping.csv')
        self._raw_threed_front_mapping = read_mapping_csv(self.blender_label_mapping_path, from_label='id', to_label='name')
        self.cam_sample_info = {'height': [1.4, 1.8], 'pitch': [math.pi/2 - 1.338, math.pi/2 - 1.2217]}
        self.cam_loc_padding = 1 # pad 1 meter away from any objects
        self._label_names = []
        self._label_mapping = []
        self._generic_mapping = []
        self.min_bbox_edge_len = 4 # bboxes with min edge length <= it will be discarded.

        if not self.dump_dir_to_scenes.exists():
            self.dump_dir_to_scenes.mkdir(parents=True)

        if not self.dump_dir_to_samples.exists():
            self.dump_dir_to_samples.mkdir(parents=True)

    @property
    def label_names(self):
        return self._label_names

    @property
    def label_mapping(self):
        return self._label_mapping

    @property
    def generic_mapping(self):
        return self._generic_mapping

    @property
    def raw_threed_front_mapping(self):
        return self._raw_threed_front_mapping

    def blender2opengl_cam(self, cam_T):
        y_z_swap = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        cam_T = y_z_swap.dot(cam_T)
        # revert x-axis
        cam_T[:3, 0] *= -1
        return cam_T

    def init_generic_categories_by_room_type(self, room_type='all'):
        '''room_type should be in ['all', 'bed', 'living', 'dining', 'library']'''
        if room_type == 'all':
            generic_mapping = {**THREED_FRONT_BEDROOM_FURNITURE, **THREED_FRONT_LIBRARY_FURNITURE,
                               **THREED_FRONT_LIVINGROOM_FURNITURE}
        elif room_type == 'bed':
            generic_mapping = THREED_FRONT_BEDROOM_FURNITURE
        elif room_type == 'living' or room_type == 'dining':
            generic_mapping = THREED_FRONT_LIVINGROOM_FURNITURE
        elif room_type == 'library':
            generic_mapping = THREED_FRONT_LIBRARY_FURNITURE
        else:
            raise Exception('Not defined room type.')

        # to make sure generic_mapping maps (a subset of) labels from raw 3DFront raw categories.
        assert not len([threed_label for threed_label in generic_mapping.keys() if
                        threed_label not in self.raw_threed_front_mapping.values()])

        bg_label = 'void'
        raw_label_to_generic_label = dict()
        for raw_3dfront_id, raw_3dfront_label in self.raw_threed_front_mapping.items():
            if raw_3dfront_label in generic_mapping.keys():
                raw_label_to_generic_label[raw_3dfront_id] = generic_mapping[raw_3dfront_label]
            else:
                raw_label_to_generic_label[raw_3dfront_id] = bg_label

        new_categories = [bg_label] + sorted(list(set(raw_label_to_generic_label.values()) - {bg_label}))
        new_label_mapping = {int(key): new_categories.index(value) for key, value in raw_label_to_generic_label.items()}

        # Update label names and mapping dict.
        self._label_names = new_categories
        self._label_mapping = new_label_mapping
        self._generic_mapping = generic_mapping