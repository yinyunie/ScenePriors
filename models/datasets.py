#  Copyright (c) 5.2021. Yinyu Nie
#  License: MIT

from torch.utils.data import Dataset
from utils.tools import read_json
import os
import numpy as np

class Base_Dataset(Dataset):
    def __init__(self, cfg, mode):
        '''
        initiate a base dataset for data loading in other networks
        :param cfg: config file
        :param mode: train/val/test mode
        '''
        self.cfg = cfg
        self.mode = mode
        self._room_uids = cfg.room_uids[self.mode]
        self._split = cfg.split_data[mode]

    @property
    def split(self):
        return self._split

    @property
    def room_uids(self):
        return self._room_uids

    def update_split(self, n_views_for_finetune=-1, room_uid=None, **kwargs):
        if n_views_for_finetune > 0:
            self.sample_n_views_per_scene(n_views_for_finetune)

        if room_uid is not None:
            if 'aug' not in room_uid:
                room_uid = room_uid+'_aug_0'
            self._split = {room_uid: self.split[room_uid]}
            self._room_uids = [room_uid]

    def sample_n_views_per_scene(self, n_views_for_finetune):
        '''Sample n views per scene.'''
        for room_uid in self.room_uids:
            samples_in_room = self.split[room_uid]
            if len(samples_in_room) > n_views_for_finetune:
                samples_in_room = np.random.choice(samples_in_room, n_views_for_finetune, replace=False).tolist()
            self._split[room_uid] = samples_in_room

    def __len__(self):
        return len(self.room_uids)