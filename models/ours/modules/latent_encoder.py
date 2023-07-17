import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from models.registers import MODULES
from net_utils.frozen_batchnorm import FrozenBatchNorm2d


@MODULES.register_module
class Latent_Encoder(nn.Module):
    def __init__(self, cfg, optim_spec=None, device='cuda'):
        '''
        Encode scene priors from embeddings
        :param cfg: configuration file.
        :param optim_spec: optimizer parameters.
        '''
        super(Latent_Encoder, self).__init__()
        '''Optimizer parameters used in training'''
        self.optim_spec = optim_spec
        self.device = device

        '''Network'''
        feature_size = 64
        self.z_dim = cfg.config.data.z_dim

        self.feature_extractor = models.resnet18(pretrained=True)
        if cfg.config.model.generator.arch.latent_encode.freeze_bn:
            FrozenBatchNorm2d.freeze(self.feature_extractor)

        self.feature_extractor.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_extractor.fc = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, feature_size), nn.ReLU())

        self.mean_fc = nn.Sequential(nn.Linear(feature_size, 64), nn.ReLU(),
                                     nn.Linear(64, self.z_dim))
        self.logstd_fc = nn.Sequential(nn.Linear(feature_size, 64), nn.ReLU(),
                                       nn.Linear(64, self.z_dim))

    def forward(self, img):
        n_batch, n_view, n_channel, n_width, n_height = img.shape
        img = img.view(n_batch * n_view, n_channel, n_width, n_height)
        feature = self.feature_extractor(img)
        feature = feature.view(n_batch, n_view, -1)
        feature = torch.mean(feature, dim=1, keepdim=True)
        mean = self.mean_fc(feature)
        logstd = self.logstd_fc(feature)
        q_z = dist.Normal(mean, torch.exp(logstd))
        return q_z


@MODULES.register_module
class Latent_Embedding(nn.Module):
    def __init__(self, cfg, optim_spec=None, device='cuda'):
        '''
        Encode scene priors from embeddings
        :param cfg: configuration file.
        :param optim_spec: optimizer parameters.
        '''
        super(Latent_Embedding, self).__init__()

        '''Optimizer parameters used in training'''
        self.optim_spec = optim_spec
        self.device = device

        '''Network params'''
        self.cfg = cfg
        n_modes = cfg.config.data.n_modes
        embed_dim = cfg.config.data.z_dim
        if cfg.config.mode == 'train':
            self.split = 'train'
        elif cfg.config.mode in ['test', 'interpolation']:
            self.split = cfg.config.test.finetune_split
        else:
            self.split = cfg.config.mode

        self.weight_embedding = nn.ModuleDict()
        if cfg.config.mode in ['train', 'test', 'interpolation']:
            for mode, room_uids in cfg.room_uids.items():
                n_samples = len(room_uids)
                self.weight_embedding["split_" + mode] = nn.Embedding(n_samples, n_modes).requires_grad_(
                    mode == self.split)
        else:
            # for overfitting a single demo
            self.weight_embedding["split_" + self.split] = nn.Embedding(1, n_modes)

        init_main_modes = torch.randn(n_modes, embed_dim)
        # project main modes to the surface of a hyper-sphere.
        init_main_modes = F.normalize(init_main_modes, p=2, dim=-1)
        self.main_modes = nn.Parameter(init_main_modes, requires_grad=False)

    def interpolate(self, idx1, idx2, interval):
        '''Interpolate latent_z between two embeddings'''
        latent_1 = self(idx1)
        latent_2 = self(idx2)
        interplated = (1 - interval) * latent_1 + interval * latent_2
        interplated = F.normalize(interplated, p=2, dim=-1)
        return interplated

    def sample_latent(self):
        '''sample a latent code on latent sphere'''
        '''sample weights'''
        mode_weights = torch.randn(1, self.cfg.config.data.n_modes, device=self.device)
        mode_weights = mode_weights.softmax(dim=-1)

        '''Sample main modes'''
        latent_z = torch.mm(mode_weights, self.main_modes)
        latent_z = F.normalize(latent_z, p=2, dim=-1)
        return latent_z.unsqueeze(1)

    def forward(self, idx):
        '''Obtain latent codes for generation'''
        '''Normalize mode weights'''
        mode_weights = self.weight_embedding["split_" + self.split](idx)
        mode_weights = mode_weights.softmax(dim=-1)
        # mode_weights = F.gumbel_softmax(mode_weights, dim=-1)

        '''Sample main modes'''
        latent_z = torch.mm(mode_weights, self.main_modes)
        latent_z = F.normalize(latent_z, p=2, dim=-1)
        return latent_z.unsqueeze(1)
