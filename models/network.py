#  Copyright (c) 5.2021. Yinyu Nie
#  License: MIT
from models.registers import MODULES, LOSSES
import torch.nn as nn


class BaseNetwork(nn.Module):
    '''
    Base Network Module for other networks
    '''
    def __init__(self, cfg, net_type='generator', device='cuda'):
        '''
        load submodules for the network.
        :param config: customized configurations.
        '''
        super(BaseNetwork, self).__init__()
        self.cfg = cfg
        self.device = device
        config = cfg.config

        '''load network blocks'''
        for phase_name, net_spec in config.model[net_type].arch.items():
            method_name = net_spec['module']
            # load specific optimizer parameters
            optim_spec = self.load_optim_spec(config, net_spec, net_type)
            subnet = MODULES.get(method_name)(cfg, optim_spec)
            self.add_module(phase_name, subnet)

            '''load corresponding loss functions'''
            setattr(self, phase_name + '_loss', LOSSES.get(config.model[net_type].arch[phase_name].loss, 'Null')(
                config.model[net_type].arch[phase_name].get('weight', 1), cfg))

        '''freeze submodules or not'''
        self.freeze_modules(config)

    def freeze_modules(self, config):
        '''
        Freeze modules in training
        '''
        freeze_layers = config.train.freeze
        for layer in freeze_layers:
            if not multi_hasattr(self, layer):
                continue
            for param in multi_getattr(self, layer).parameters():
                param.requires_grad = False
            self.cfg.info('The module: %s is frozen.' % (layer))

    def set_mode(self):
        '''
        Set train/eval mode for the network.
        :param phase: train or eval
        :return:
        '''
        freeze_layers = self.cfg.config.train.freeze
        for name, child in self.named_children():
            if name in freeze_layers:
                child.train(False)

    def load_weight(self, pretrained_model):
        model_dict = self.state_dict()
        # remove the 'module' string.
        pretrained_dict = {'.'.join(k.split('.')[1:]): v for k, v in pretrained_model.items() if
                           '.'.join(k.split('.')[1:]) in model_dict}
        self.cfg.info(self._get_name() + ': ' +
            str(set([key for key in model_dict if key not in pretrained_dict])) + ' subnet missed.')
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def load_optim_spec(self, config, net_spec, net_type):
        # load specific optimizer parameters
        if config.mode == 'train':
            if 'optimizer' in net_spec.keys():
                optim_spec = config.optimizer.copy()
                for key in optim_spec.keys():
                    optim_spec[key] = net_spec['optimizer'].get(key, optim_spec[key])
            else:
                optim_spec = config.optimizer.copy()  # else load default optimizer

            optim_spec['type'] = net_type
        else:
            optim_spec = None

        return optim_spec

    def forward(self, *args, **kwargs):
        ''' Performs a forward step.
        '''
        raise NotImplementedError

    def loss(self, *args, **kwargs):
        ''' calculate losses.
        '''
        raise NotImplementedError

def multi_getattr(layer, attr, default=None):
    attributes = attr.split(".")
    for i in attributes:
        try:
            layer = getattr(layer, i)
        except AttributeError:
            if default:
                return default
            else:
                raise
    return layer

def multi_hasattr(layer, attr):
    attributes = attr.split(".")
    hasattr_flag = True
    for i in attributes:
        if hasattr(layer, i):
            layer = getattr(layer, i)
        else:
            hasattr_flag = False
            break
    return hasattr_flag