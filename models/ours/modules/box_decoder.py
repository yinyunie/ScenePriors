#  Copyright (c) 7.2022. Yinyu Nie
#  License: MIT

import torch
import torch.nn as nn
from models.registers import MODULES
from models.ours.modules.hidden_to_output import DeterminsticOutput


@MODULES.register_module
class BoxDecoder(nn.Module):
    def __init__(self, cfg, optim_spec=None, device='cuda'):
        '''
        Decode shapes from a latent vector.
        :param cfg: configuration file.
        :param optim_spec: optimizer parameters.
        '''
        super(BoxDecoder, self).__init__()

        '''Optimizer parameters used in training'''
        self.optim_spec = optim_spec
        self.device = device

        '''Modules'''
        inst_latent_len = cfg.config.data.backbone_latent_len
        self.n_classes = len(cfg.label_names)
        self.hidden2box = DeterminsticOutput(hidden_size=inst_latent_len,
                                             n_classes=self.n_classes,
                                             with_extra_fc=False)

    def generate(self, box_codes):
        output_box3ds = []
        for box_codes_batch in box_codes:
            box3ds = self.box_regressor(box_codes_batch)
            output_box3ds.append(box3ds)
        return output_box3ds

    def box_regressor(self, box_codes):
        box3ds = self.hidden2box(box_codes)
        return box3ds

    def forward(self, box_codes, completeness_score):
        box3ds = self.box_regressor(box_codes)
        return torch.cat([box3ds, completeness_score], dim=-1)