#  Copyright (c) 2.2022. Yinyu Nie
#  License: MIT

import torch
import wandb
from time import time
import os
from net_utils.utils import CheckpointIO, LossRecorder, AverageMeter
from models.optimizers import load_optimizer, load_scheduler, load_bnm_scheduler
from net_utils.utils import load_device, load_model, load_trainer, load_dataloader

class Train(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.is_master = cfg.is_master

        '''Load save path and checkpoint handler.'''
        cfg.info('Data save path: %s' % (os.getcwd()))
        cfg.info('Loading checkpoint handler')
        self.checkpoint = CheckpointIO(cfg, self.is_master)

        '''Load device'''
        cfg.info('Loading device settings.')
        device = load_device(cfg)

        '''Load data'''
        cfg.info('Loading dataset.')
        train_loader = load_dataloader(cfg, mode='train')
        self.dataloaders = {'train': train_loader}

        '''Load model'''
        cfg.info('Loading model')
        self.net = load_model(cfg, device=device)
        self.checkpoint.register_modules(net=self.net)
        cfg.info(self.net)

        '''Load optimizer'''
        cfg.info('Loading optimizer.')
        self.optimizer = load_optimizer(config=cfg.config, net=self.net)
        self.checkpoint.register_modules(optimizer=self.optimizer)

        '''Load scheduler'''
        cfg.info('Loading optimizer scheduler.')
        self.scheduler = load_scheduler(cfg=cfg, optimizer=self.optimizer)
        self.checkpoint.register_modules(scheduler=self.scheduler)

        '''Check existing checkpoint (resume or finetune)'''
        self.checkpoint.parse_checkpoint(device)

        '''BN momentum scheduler'''
        cfg.info('Loading batchnorm scheduler.')
        self.bnm_scheduler = load_bnm_scheduler(cfg=cfg, net=self.net, start_epoch=self.scheduler['generator'].last_epoch)

        '''Load sub trainer for a specific method.'''
        cfg.info('Loading method trainer.')
        self.subtrainer = load_trainer(cfg=cfg, net=self.net, optimizer=self.optimizer, device=device)

        '''Output network size'''
        self.subtrainer.show_net_n_params()

        # put logger where it belongs
        if self.is_master and cfg.config.log.if_wandb:
            cfg.info('Loading wandb.')
            wandb.init(project=cfg.config.method, name=cfg.config.exp_name, config=cfg.config)
            # wandb.watch(self.net)

    def log_wandb(self, loss, phase):
        dict_ = dict()
        for key, value in loss.items():
            dict_[phase + '/' + key] = value
        wandb.log(dict_)

    def train_epoch(self, epoch, stage):
        '''train and val'''
        '''Time meter setup.'''
        dataload_timemeter = AverageMeter()
        batch_timemeter = AverageMeter()

        phase = 'train'
        self.cfg.info('-' * 100)
        self.cfg.info('Switch phase to %s.' % (phase))
        self.cfg.info('-' * 100)

        dataloader = self.dataloaders[phase]
        batch_size = self.cfg.config[phase].batch_size // self.cfg.config.distributed.num_gpus
        loss_recorder = LossRecorder(batch_size)
        # set mode
        self.subtrainer.set_mode(phase)

        if self.cfg.config.distributed.num_gpus > 1:
            dataloader.batch_sampler.sampler.set_epoch(epoch)

        torch.cuda.empty_cache()
        batch_start = time()
        for iter, data in enumerate(dataloader):
            # measure data loading time
            dataload_timemeter.update(time() - batch_start)

            loss, extra_output = self.subtrainer.train_step(data, stage, start_deform=self.cfg.config.start_deform)

            # visualize intermediate results.
            if (iter % self.cfg.config.log.vis_step) == 0:
                self.subtrainer.visualize_step(epoch, phase, iter, data)

            loss_recorder.update_loss(loss)

            '''Display batch info'''
            batch_timemeter.update(time() - batch_start)
            if (iter % self.cfg.config.log.print_step) == 0:
                self.cfg.info('G_LR: {G_lr:s} | {phase:s} | Epoch: [{0}][{1}/{2}] | Loss: {loss:s}\
                               Batch Time {batch_time:.3f} | Data Time {data_time:.3f}'.format(
                               epoch, iter + 1, len(dataloader), phase=phase, loss=str(loss),
                               batch_time=batch_timemeter.avg, data_time=dataload_timemeter.avg,
                               G_lr=str(self.scheduler['generator'].get_last_lr()[:2])))
                if self.is_master and self.cfg.config.log.if_wandb:
                    self.log_wandb(loss, phase)
            batch_start = time()

        # synchronize over all processes
        loss_recorder.synchronize_between_processes()

        '''Display epoch info'''
        self.cfg.info('=' * 100)
        for loss_name, loss_value in loss_recorder.loss_recorder.items():
            self.cfg.info('Currently the last %s loss (%s) is: %f' % (phase, loss_name, loss_value.avg))
        self.cfg.info('=' * 100)

        return loss_recorder.loss_recorder

    def run(self):
        '''Start to train'''
        self.cfg.info('Start to train.')
        # ---------------------------------------------------------------------------------------
        start_epoch = self.scheduler['generator'].last_epoch
        epochs_network = self.cfg.config.train.epochs
        epochs_latents = self.cfg.config.train.epochs_latent
        total_epochs = epochs_network + epochs_latents
        min_eval_loss = self.checkpoint.get('min_loss')
        stage = 'all'
        net_types = self.scheduler.keys()

        self.cfg.info('Start to train network + latents.')
        for epoch in range(start_epoch, total_epochs):
            if epoch == epochs_network:
                self.cfg.info('Network training finished.')
                self.cfg.info('=' * 100)
                self.cfg.info('Start to train latent codes.')
                stage = 'latent_only'
                net_types = ['latent_input']
                for net_type, subnet in self.net.items():
                    if net_type in net_types: continue
                    for param in subnet.parameters():
                        param.requires_grad = False

            self.cfg.info('-' * 100)
            self.cfg.info('Epoch (%d/%s):' % (epoch, total_epochs - 1))
            self.subtrainer.show_lr()
            epoch_start = time()
            eval_loss_recorder = self.train_epoch(epoch, stage)
            eval_loss = self.subtrainer.eval_loss_parser(eval_loss_recorder)
            for net_type in net_types:
                self.scheduler[net_type].step()
                self.bnm_scheduler[net_type].step()
            self.cfg.info('Epoch (%d/%s) Time elapsed: (%f).' % (epoch, total_epochs - 1, time() - epoch_start))

            # save checkpoint
            self.checkpoint.register_modules(epoch=epoch, min_loss=eval_loss)
            if ((epoch % self.cfg.config.log.save_weight_step) == 0) or (epoch == total_epochs - 1):
                self.checkpoint.save('last_{:04d}'.format(epoch))
                self.cfg.info('Saved the latest checkpoint.')
            if epoch == 0 or eval_loss < min_eval_loss:
                self.checkpoint.save('best')
                min_eval_loss = eval_loss
                self.cfg.info('Saved the best checkpoint.')
                self.cfg.info('=' * 100)
                for loss_name, loss_value in eval_loss_recorder.items():
                    self.cfg.info('Currently the best val loss (%s) is: %f' % (loss_name, loss_value.avg))
                self.cfg.info('=' * 100)
        # ---------------------------------------------------------------------------------------
        wandb.finish()
        self.cfg.info('Training finished.')