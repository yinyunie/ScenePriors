#  Trainer for P2RNet.
#  Copyright (c) 5.2021. Yinyu Nie
#  License: MIT
from models.training import BaseTrainer
from net_utils.distributed import reduce_dict


class Trainer(BaseTrainer):
    '''
    Trainer object for total3d.
    '''
    def __init__(self, cfg, net, optimizer, device=None):
        super(Trainer, self).__init__(cfg, net, optimizer, device)
        self.latent_input = net['latent_input']
        self.generator = net['generator']

    def eval_step(self, data):
        '''
        performs a step in evaluation
        :param data (dict): data dictionary
        :return:
        '''
        '''load input and ground-truth data'''
        data = self.to_device(data)

        '''network forwarding'''
        latent_z = self.latent_input(data)
        est_data = self.generator(latent_z, data)

        '''compute losses'''
        loss = self.generator.module.loss(est_data, data)

        # for logging
        loss_reduced = reduce_dict(loss)
        loss_dict = {k: v.item() for k, v in loss_reduced.items()}
        return loss_dict

    def train_step(self, data, stage='all', start_deform=False, **kwargs):
        '''
        performs a step training
        :param data (dict): data dictionary
        :return:
        '''
        if stage == 'all':
            net_types = self.optimizer.keys()
        elif stage == 'latent_only':
            net_types = ['latent_input']
        else:
            raise ValueError('No such stage.')

        for net_type in net_types:
            self.optimizer[net_type].zero_grad()

        loss, extra_output = self.compute_loss(data, start_deform=start_deform, **kwargs)

        if loss['total'].requires_grad:
            loss['total'].backward()
            if self.cfg.config.train.clip_norm:
                self.clip_grad_norm(net=self.net['generator'])
            for net_type in net_types:
                self.optimizer[net_type].step()

        # for logging
        loss_reduced = reduce_dict(loss)
        loss_dict = {k: v.item() for k, v in loss_reduced.items()}
        return loss_dict, extra_output

    def visualize_step(self, *args, **kwargs):
        ''' Performs a visualization step.
        '''
        if not self.cfg.is_master:
            return
        pass

    def to_device(self, data):
        device = self.device
        for key in data:
            if key in ['sample_name']: continue
            data[key] = data[key].to(device)
        return data

    def compute_loss(self, data, start_deform=False, **kwargs):
        '''
        compute the overall loss.
        :param data (dict): data dictionary
        :return:
        '''
        '''load input and ground-truth data'''
        data = self.to_device(data)

        '''network forwarding'''
        latent_z = self.latent_input(data)
        est_data = self.generator(latent_z, data, start_deform=start_deform, **kwargs)

        '''compute losses'''
        loss, extra_output = self.generator.module.loss(est_data, data, start_deform=start_deform, **kwargs)
        return loss, extra_output
