#  Copyright (c) 2.2022. Yinyu Nie
#  License: MIT
import torch
import wandb
from time import time
import os
import numpy as np
from net_utils.utils import CheckpointIO, AverageMeter
from models.optimizers import load_optimizer, load_scheduler
from net_utils.utils import load_device, load_model, load_trainer, load_tester
from models.ours.dataloader import THREEDFRONT, ScanNet, collate_fn
import h5py


class Demo(object):
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

        '''Load data'''
        cfg.info('Reading demo samples.')
        self.all_samples = self.read_samples(self.cfg.demo_samples)

        '''Load model'''
        cfg.info('Loading model')
        self.net = load_model(cfg, device=self.device)
        self.checkpoint.register_modules(net=self.net)
        cfg.info(self.net)

        '''Freeze network part'''
        for net_type, subnet in self.net.items():
            if net_type in ['latent_input']: continue
            self.cfg.info('%s is frozen.' % (net_type))
            for param in subnet.parameters():
                param.requires_grad = False

        '''Read network weights (finetune mode)'''
        self.checkpoint.parse_checkpoint(device=self.device)

        '''Load sub trainer for a specific method.'''
        cfg.info('Loading method tester.')
        self.subtester = load_tester(cfg=cfg, net=self.net, device=self.device)

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

    def read_samples(self, sample_files):
        batch_id = self.cfg.config.demo.batch_id
        batch_num = self.cfg.config.demo.batch_num

        sublist = np.array_split(np.arange(len(sample_files)), batch_num)[batch_id]
        sample_files = [sample_files[idx] for idx in sublist]
        samples = []
        for sample_file in sample_files:
            processed_sample = self.read_data(sample_file, self.cfg.config.data.dataset)
            samples.append(processed_sample)
        return samples

    def read_sample(self, sample_name, dataset_name):
        if dataset_name == '3D-Front':
            room_type, img, cam_K, cam_T, insts = THREEDFRONT.parse_hdf5(self.cfg, sample_name)
            image_size = self.cfg.image_size
        elif dataset_name == 'ScanNet':
            room_type, img, cam_K, cam_T, insts, image_size = ScanNet.parse_hdf5(self.cfg, sample_name)
        else:
            raise NotImplementedError
        return room_type, img, cam_K, cam_T, insts, image_size

    def load_unique_inst_marks(self, sample_path, dataset_name):
        '''read data'''
        if dataset_name == '3D-Front':
            room_name = '_'.join(os.path.basename(sample_path).split('_')[:-1])
            unique_inst_marks = self.cfg.unique_inst_mark[room_name]
        elif dataset_name == 'ScanNet':
            with h5py.File(sample_path, "r") as sample_data:
                unique_inst_marks = sample_data['unique_inst_marks'][:].tolist()
        else:
            raise NotImplementedError
        return unique_inst_marks

    def read_data(self, sample_path, dataset_name):
        '''read views data from 3d front'''

        # sample other view
        room_name = '_'.join(sample_path.name.split('_')[:-1])

        candidates = [file for file in sample_path.parent.iterdir() if room_name in file.name and file.name != sample_path.name]

        if len(candidates):
            view_ids = np.random.choice(len(candidates), self.cfg.config.data.n_views - 1,
                                        replace=len(candidates) < self.cfg.config.data.n_views - 1)
            view_files = [sample_path] + [candidates[idx] for idx in view_ids]
        else:
            view_files = [sample_path] * self.cfg.config.data.n_views

        parsed_data = []
        for view_file in view_files:
            room_type, img, cam_K, cam_T, insts, image_size = self.read_sample(view_file, dataset_name)
            sample_name = '.'.join(os.path.basename(view_file).split('.')[:-1])
            parsed_data.append((img, cam_K, cam_T, insts, image_size, sample_name))

        room_type_idx = self.cfg.room_types.index(room_type)

        unique_marks = sorted(list(set(sum([view[3]['inst_marks'] for view in parsed_data], []))))

        # re-organize instances following track ids
        parsed_data = THREEDFRONT.track_insts(parsed_data, unique_marks)

        keywords = ['sample_name', 'cam_K', 'image_size', 'cam_T', 'box2ds_tr', 'inst_marks']

        if self.cfg.config.start_deform:
            keywords.append('masks_tr')
            if self.cfg.config.data.dataset == 'ScanNet':
                keywords.append('render_mask_tr')

        views_data = []
        for parsed_view in parsed_data:
            view_data = THREEDFRONT.get_view_data(self.cfg, parsed_view, self.cfg.config.data.downsample_ratio)
            view_data = {**{k: view_data[k] for k in keywords},
                         **{'room_idx': 0, 'max_len': len(unique_marks), 'room_type_idx': room_type_idx}}
            views_data.append(view_data)

        return collate_fn([views_data])

    def run(self):
        '''Finetune latent codes and output results'''
        '''Start to finetune latent codes'''
        self.cfg.info('Start to finetune latent codes.')
        '''Time meter setup.'''
        sample_timemeter = AverageMeter()
        epoch_timemeter = AverageMeter()
        phase = 'train'
        start_epoch = 0
        total_epochs = self.cfg.config.demo.epochs

        # ---------------------------------------------------------------------------------------
        for sample in self.all_samples:
            sample_start = time()
            torch.cuda.empty_cache()

            self.cfg.info('=' * 100)
            self.cfg.info('Processing: %s.' % (sample['sample_name'][0][0]))
            self.cfg.info('Loading optimizer.')
            '''Load optimizer'''
            optimizer = load_optimizer(config=self.cfg.config, net=self.net)

            '''Load scheduler'''
            self.cfg.info('Loading optimizer scheduler.')
            scheduler = load_scheduler(cfg=self.cfg, optimizer=optimizer)

            '''Load sub trainer for a specific method.'''
            self.cfg.info('Loading method trainer.')
            subtrainer = load_trainer(cfg=self.cfg, net=self.net, optimizer=optimizer, device=self.device)

            '''Freeze network'''
            # set mode
            subtrainer.set_mode(phase)
            # freeze the network part
            for net_type, subnet in self.net.items():
                if net_type in ['latent_input']: continue
                for child in subnet.children():
                    child.train(False)

            '''Start to finetune latent code'''
            self.cfg.info('Start to finetune latent code.')
            # ---------------------------------------------------------------------------------------
            min_eval_loss = self.checkpoint.get('min_loss')
            loss = {'total': min_eval_loss}
            pred_gt_matching = None
            if_mask_loss = False
            losses = []
            for epoch in range(start_epoch, total_epochs):
                epoch_start = time()
                if (epoch % self.cfg.config.log.print_step) == 0:
                    self.cfg.info('-' * 100)
                    self.cfg.info('Epoch (%d/%s):' % (epoch, total_epochs - 1))
                    subtrainer.show_lr()

                if epoch > self.cfg.config.demo.mask_flag:
                    pred_gt_matching = extra_output['pred_gt_matching']
                    if_mask_loss = True

                loss, extra_output = subtrainer.train_step(sample, stage='latent_only',
                                                           start_deform=self.cfg.config.start_deform,
                                                           return_matching=True, pred_gt_matching=pred_gt_matching, if_mask_loss=if_mask_loss)
                losses.append(loss)
                if loss['total'] < min_eval_loss:
                    min_eval_loss = loss['total']

                '''Display epoch info'''
                if (epoch % self.cfg.config.log.print_step) == 0:
                    epoch_timemeter.update(time() - epoch_start)
                    self.cfg.info('Latent_lr: {Latent_lr:s} | {phase:s} | Epoch: [{0}/{1}] | Loss: {loss:s}\
                                   Epoch Time {epoch_time:.3f}'.format(
                        epoch, total_epochs, phase='finetune', loss=str(loss),
                        epoch_time=epoch_timemeter.avg, Latent_lr=str(scheduler['latent_input'].get_last_lr()[:2])))

                    if self.is_master and self.cfg.config.log.if_wandb:
                        self.log_wandb(loss, sample['sample_name'][0][0])

                scheduler['latent_input'].step()

            sample_timemeter.update(time() - sample_start)

            self.cfg.info('-' * 100)
            self.cfg.info('{sample:s}: Best loss is {best_loss:.3f} | Last loss is {last_loss:.3f} | Avg fitting time: {time:.3f}'.format(
                sample=sample['sample_name'][0][0], best_loss=min_eval_loss, last_loss=loss['total'], time=sample_timemeter.avg))

            '''Output visualizations'''
            self.cfg.info('=' * 100)
            self.cfg.info('Export visualizations.')
            # set mode
            self.subtester.set_mode('test')
            with torch.no_grad():
                _, est_data = self.subtester.test_step(sample, start_deform=self.cfg.config.start_deform, pred_gt_matching=extra_output['pred_gt_matching'])
                self.subtester.visualize_step(phase, iter, sample, est_data,
                                              dump_dir=self.cfg.config.demo.output_dir)
        # ---------------------------------------------------------------------------------------
        wandb.finish()
        self.cfg.info('Testing finished.')