#  Copyright (c) 6.2022. Yinyu Nie
#  License: MIT

import torch
import torch.nn as nn
from models.registers import MODULES
from pytorch3d.utils import ico_sphere


@MODULES.register_module
class ShapeDecoder(nn.Module):
    def __init__(self, cfg, optim_spec=None, device='cuda'):
        '''
        Decode shapes from a latent vector.
        :param cfg: configuration file.
        :param optim_spec: optimizer parameters.
        '''
        super(ShapeDecoder, self).__init__()

        '''Optimizer parameters used in training'''
        self.optim_spec = optim_spec
        self.device = device

        '''Modules'''
        self.src_mesh = ico_sphere(4, 'cuda')
        inst_latent_len = cfg.config.data.backbone_latent_len
        self.mlp_feat = nn.Sequential(nn.Linear(inst_latent_len, 512), nn.ReLU(),
                                      nn.Linear(512, 256))

        self.mlp_deform = nn.Sequential(nn.Linear(256 + 3, 256), nn.ReLU(),
                                        nn.Linear(256, 256), nn.ReLU(),
                                        nn.Linear(256, 256), nn.ReLU(),
                                        nn.Linear(256, 3))

    def generate(self, shape_feat):
        batch_meshes = []
        for feat in shape_feat:
             batch_meshes.append(self(feat))
        return batch_meshes

    def forward(self, shape_feat):
        shape_feat = self.mlp_feat(shape_feat)

        n_batch, n_object, feat_dim = shape_feat.shape
        shape_feat = shape_feat.view(n_batch * n_object, feat_dim)
        meshes = self.src_mesh.extend(n_batch * n_object)
        vertices = meshes.verts_padded()
        n_vertices = vertices.size(1)
        shape_feat = shape_feat[:, None].expand(-1, n_vertices, -1)
        shape_feat = torch.cat([shape_feat, vertices], dim=-1)
        offsets = self.mlp_deform(shape_feat)
        meshes = meshes.update_padded(new_verts_padded=vertices + offsets)
        return meshes

