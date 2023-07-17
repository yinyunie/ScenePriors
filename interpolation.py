#  Copyright (c) 5.2022. Yinyu Nie
#  License: MIT
import torch
from time import time
import os
from net_utils.utils import CheckpointIO, LossRecorder, AverageMeter
from models.optimizers import load_optimizer
from net_utils.utils import load_device, load_model, load_tester
import numpy as np


class Interpolation(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.is_master = cfg.is_master

        '''Load save path and checkpoint handler.'''
        cfg.info('Data save path: %s' % (os.getcwd()))
        cfg.info('Loading checkpoint handler')
        self.checkpoint = CheckpointIO(cfg, self.is_master)

        '''Load device'''
        cfg.info('Loading device settings.')
        self.device = load_device(cfg)

        self.split = cfg.config.test.finetune_split

        '''Load model'''
        cfg.info('Loading model')
        self.net = load_model(cfg, device=self.device)
        self.checkpoint.register_modules(net=self.net)
        cfg.info(self.net)

        '''Read network weights (finetune mode)'''
        self.checkpoint.parse_checkpoint(self.device)

        '''Load tester.'''
        cfg.info('Loading method tester.')
        self.subtester = load_tester(cfg=cfg, net=self.net, device=self.device)

        '''Output network size'''
        self.subtester.show_net_n_params()

    def interpolate_step(self, interval_id, interval, start_deform=False):
        sample_1 = self.cfg.config.interpolation.sample_1
        sample_2 = self.cfg.config.interpolation.sample_2
        if sample_1 == 'random':
            room_idx_1 = np.random.choice(len(self.cfg.room_uids[self.split]), 1)[0]
        else:
            room_idx_1 = self.cfg.room_uids[self.split].index(sample_1)
        if sample_2 == 'random':
            room_idx_2 = np.random.choice(len(self.cfg.room_uids[self.split]), 1)[0]
        else:
            room_idx_2 = self.cfg.room_uids[self.split].index(sample_2)
        room_type_idx = torch.tensor([[self.cfg.room_types.index(self.cfg.config.interpolation.room_type)]], device=self.device).long()
        est_data = self.subtester.interpolate(room_idx_1, room_idx_2, interval, room_type_idx, start_deform=start_deform)
        # visualize intermediate results.
        if self.cfg.config.generation.dump_results:
            output_filename = self.cfg.room_uids[self.split][room_idx_1] + '_' + self.cfg.room_uids[self.split][room_idx_2] + '_' + "{:05d}".format(interval_id)
            self.subtester.visualize_interp('interpolation', output_filename, est_data)

        torch.cuda.empty_cache()

    def run(self):
        '''Produce interpolation results between samples.'''
        '''Start to finetune latent codes'''
        self.cfg.info('Start to interpolate latent code between samples.')
        # ---------------------------------------------------------------------------------------
        # set mode
        self.subtester.set_mode(self.cfg.config.mode)
        intervals = self.cfg.config.interpolation.intervals
        with torch.no_grad():
            for interval_id, interval in enumerate(torch.linspace(0, 1, intervals + 1)):
                start = time()
                self.interpolate_step(interval_id, interval, start_deform=self.cfg.config.start_deform)
                self.cfg.info('Test time elapsed: (%f).' % (time() - start))
        # ---------------------------------------------------------------------------------------
        self.cfg.info('Interpolation finished.')