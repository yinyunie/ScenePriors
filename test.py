#  Copyright (c) 2.2022. Yinyu Nie
#  License: MIT
import torch
import wandb
from time import time
import os
from net_utils.utils import CheckpointIO, LossRecorder, AverageMeter
from models.optimizers import load_optimizer, load_scheduler
from net_utils.utils import load_device, load_model, load_trainer, load_tester, load_dataloader
from typing import Dict

class Test(object):
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
        self.split = cfg.config.test.finetune_split
        n_views_for_finetune = cfg.config.test.n_views_for_finetune
        self.dataloader = load_dataloader(cfg, mode=self.split)
        self.dataloader.dataset.update_split(n_views_for_finetune=n_views_for_finetune)

        '''Load model'''
        cfg.info('Loading model')
        self.net = load_model(cfg, device=device)
        self.checkpoint.register_modules(net=self.net)
        cfg.info(self.net)

        '''Freeze network part'''
        for net_type, subnet in self.net.items():
            if net_type in ['latent_input']: continue
            self.cfg.info('%s is frozen.' % (net_type))
            for param in subnet.parameters():
                param.requires_grad = False

        '''Read network weights (finetune mode)'''
        self.checkpoint.parse_checkpoint(device=device)

        '''Load optimizer'''
        cfg.info('Loading optimizer.')
        self.optimizer = load_optimizer(config=cfg.config, net=self.net)

        '''Load scheduler'''
        cfg.info('Loading optimizer scheduler.')
        self.scheduler = load_scheduler(cfg=cfg, optimizer=self.optimizer)

        '''Load sub trainer for a specific method.'''
        cfg.info('Loading method trainer and tester.')
        self.subtrainer = load_trainer(cfg=cfg, net=self.net, optimizer=self.optimizer, device=device)
        self.subtester = load_tester(cfg=cfg, net=self.net, device=device)

        '''Output network size'''
        self.subtester.show_net_n_params()

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

    def test_epoch(self, matching_by_rooms:Dict=dict(), **kwargs):
        '''test'''
        dataload_timemeter = AverageMeter()
        batch_timemeter = AverageMeter()

        phase = 'test'
        dataloader = self.dataloader
        batch_size = self.cfg.config[self.split].batch_size // self.cfg.config.distributed.num_gpus
        loss_recorder = LossRecorder(batch_size)

        if self.cfg.config.distributed.num_gpus > 1: # optional, since only go through dataloader once.
            dataloader.batch_sampler.sampler.set_epoch(0)

        max_n_preds = self.cfg.max_n_obj
        torch.cuda.empty_cache()
        batch_start = time()
        for iter, data in enumerate(dataloader):
            # measure data loading time
            dataload_timemeter.update(time() - batch_start)
            pred_gt_matching = [matching_by_rooms[idx.item()] for idx in
                                data['room_idx'][:, 0]] if len(matching_by_rooms) else None
            loss, est_data = self.subtester.test_step(data, start_deform=self.cfg.config.start_deform,
                                                      pred_gt_matching=pred_gt_matching)
            # visualize intermediate results.
            if self.cfg.config.generation.dump_results:
                self.subtester.visualize_step(phase, iter, data, est_data)

            loss_recorder.update_loss(loss)

            '''Display batch info'''
            batch_timemeter.update(time() - batch_start)
            if (iter % self.cfg.config.log.print_step) == 0:
                self.cfg.info('{phase:s} | Epoch: [{0}/{1}] | Loss: {loss:s}\
                               Batch Time {batch_time:.3f} | Data Time {data_time:.3f}'.format(
                    iter + 1, len(dataloader), phase=phase, loss=str(loss),
                    batch_time=batch_timemeter.avg, data_time=dataload_timemeter.avg))

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

    def finetune_latents(self, epoch, if_mask_loss=True, matching_by_rooms:Dict=dict(), **kwargs):
        '''Finetune latent codes'''
        # ---------------------------------------------------------------------------------------
        '''Time meter setup.'''
        dataload_timemeter = AverageMeter()
        batch_timemeter = AverageMeter()
        phase = 'train'

        dataloader = self.dataloader
        batch_size = self.cfg.config[self.split].batch_size // self.cfg.config.distributed.num_gpus
        loss_recorder = LossRecorder(batch_size)
        # set mode
        self.subtrainer.set_mode(phase)
        # freeze the network part
        for net_type, subnet in self.net.items():
            if net_type in ['latent_input']: continue
            for child in subnet.children():
                child.train(False)

        if self.cfg.config.distributed.num_gpus > 1:
            dataloader.batch_sampler.sampler.set_epoch(epoch)

        torch.cuda.empty_cache()
        batch_start = time()
        for iter, data in enumerate(dataloader):
            # measure data loading time
            dataload_timemeter.update(time() - batch_start)
            pred_gt_matching = [matching_by_rooms[idx.item()] for idx in
                                data['room_idx'][:, 0]] if if_mask_loss == True else None
            loss, extra_output = self.subtrainer.train_step(data, stage='latent_only',
                                                            start_deform=self.cfg.config.start_deform,
                                                            return_matching=True,
                                                            if_mask_loss=if_mask_loss, pred_gt_matching=pred_gt_matching)

            matching_by_rooms.update({room_idx.item(): matching for room_idx, matching in zip(data['room_idx'][:, 0], extra_output['pred_gt_matching'])})

            # visualize intermediate results.
            if (iter % self.cfg.config.log.vis_step) == 0:
                self.subtrainer.visualize_step(epoch, phase, iter, data)

            loss_recorder.update_loss(loss)

            '''Display batch info'''
            batch_timemeter.update(time() - batch_start)
            if (iter % self.cfg.config.log.print_step) == 0:
                self.cfg.info('Latent_lr: {Latent_lr:s} | {phase:s} | Epoch: [{0}][{1}/{2}] | Loss: {loss:s}\
                               Batch Time {batch_time:.3f} | Data Time {data_time:.3f}'.format(
                    epoch, iter + 1, len(dataloader), phase='finetune', loss=str(loss),
                    batch_time=batch_timemeter.avg, data_time=dataload_timemeter.avg,
                    Latent_lr=str(self.scheduler['latent_input'].get_last_lr()[:2])))
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
        '''Finetune latent codes and output results'''
        '''Start to finetune latent codes'''
        self.cfg.info('Start to finetune latent codes.')
        # ---------------------------------------------------------------------------------------
        start_epoch = 0
        total_epochs = self.cfg.config.test.epochs
        min_eval_loss = self.checkpoint.get('min_loss')
        finetune_start = time()

        matching_by_rooms = {}
        if_mask_loss = False
        for epoch in range(start_epoch, total_epochs):
            self.cfg.info('-' * 100)
            self.cfg.info('Epoch (%d/%s):' % (epoch, total_epochs - 1))
            self.subtrainer.show_lr()
            epoch_start = time()
            if epoch > self.cfg.config.demo.mask_flag:
                if_mask_loss = True
            eval_loss_recorder = self.finetune_latents(epoch, if_mask_loss=if_mask_loss, matching_by_rooms=matching_by_rooms)
            eval_loss = self.subtrainer.eval_loss_parser(eval_loss_recorder)
            self.scheduler['latent_input'].step()
            self.cfg.info('Epoch (%d/%s) Time elapsed: (%f).' % (epoch, total_epochs - 1, time() - epoch_start))

            # save checkpoint
            self.checkpoint.register_modules(epoch=epoch, min_loss=eval_loss)
            if ((epoch % self.cfg.config.log.save_weight_step) == 0) or (epoch == total_epochs - 1):
                self.checkpoint.save('last')
                self.cfg.info('Saved the latest checkpoint.')
            if epoch == 0 or eval_loss < min_eval_loss:
                self.checkpoint.save('best')
                min_eval_loss = eval_loss
                self.cfg.info('Saved the best checkpoint.')
                self.cfg.info('=' * 100)
                for loss_name, loss_value in eval_loss_recorder.items():
                    self.cfg.info('Currently the best val loss (%s) is: %f' % (loss_name, loss_value.avg))
                self.cfg.info('=' * 100)

        self.cfg.info('Finetuning time elapsed: (%f).' % (time() - finetune_start))

        '''Start to test'''
        self.cfg.info('Start to test.')
        # ---------------------------------------------------------------------------------------
        # set mode
        self.subtester.set_mode('test')
        start = time()
        with torch.no_grad():
            self.test_epoch(matching_by_rooms=matching_by_rooms)
        self.cfg.info('Test time elapsed: (%f).' % (time() - start))
        # ---------------------------------------------------------------------------------------
        wandb.finish()
        self.cfg.info('Testing finished.')