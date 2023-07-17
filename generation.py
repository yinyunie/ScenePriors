#  Copyright (c) 9.2022. Yinyu Nie
#  License: MIT
import torch
from time import time
import os
from net_utils.utils import CheckpointIO
from net_utils.utils import load_device, load_model, load_tester


class Generation(object):
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

    def generate_step(self, sample_idx, start_deform=False):
        room_type_idx = torch.tensor([[self.cfg.room_types.index(self.cfg.config.generation.room_type)]], device=self.device).long()
        est_data = self.subtester.generate(room_type_idx, start_deform=start_deform)
        # visualize intermediate results.
        if self.cfg.config.generation.dump_results:
            output_filename = 'sample_%d' % (sample_idx)
            self.subtester.visualize_interp('generation', output_filename, est_data)
        torch.cuda.empty_cache()

    def run(self):
        '''Sample results '''
        '''Start to finetune latent codes'''
        self.cfg.info('Start to interpolate latent code between samples.')
        # ---------------------------------------------------------------------------------------
        # set mode
        self.subtester.set_mode(self.cfg.config.mode)
        n_generations = self.cfg.config.generation.n_generations

        with torch.no_grad():
            for sample_idx in range(n_generations):
                start = time()
                self.generate_step(sample_idx, start_deform=self.cfg.config.start_deform)
                self.cfg.info('Test time elapsed: (%f).' % (time() - start))
                # ---------------------------------------------------------------------------------------
            self.cfg.info('Generation finished.')



