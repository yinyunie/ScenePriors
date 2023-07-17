#  tester for P2RNet.
#  Copyright (c) 5.2021. Yinyu Nie
#  License: MIT
import os
import numpy as np
import torch

from models.testing import BaseTester
from .training import Trainer


class Tester(BaseTester, Trainer):
    '''
    Tester object for ISCNet.
    '''

    def __init__(self, cfg, net, device=None):
        super(Tester, self).__init__(cfg, net, device)
        self.latent_input = net['latent_input']
        self.generator = net['generator']

    def get_metric_values(self, est_data, gt_data):
        ''' Performs a evaluation step.
        '''
        pass

    def evaluate_step(self, est_data, data=None):
        eval_metrics = {}
        return eval_metrics

    def interpolate(self, room_idx_1, room_idx_2, interval, room_type_idx, start_deform=False):
        '''network forwarding'''
        room_idx_1 = torch.tensor([room_idx_1]).to(self.device)
        room_idx_2 = torch.tensor([room_idx_2]).to(self.device)

        latent_z = self.latent_input.module.interpolate(room_idx_1, room_idx_2, interval)
        est_data = self.generator.module.generate(latent_z, {'room_type_idx': room_type_idx}, start_deform=start_deform, self_end=True,
                                                  output_render=False)
        return est_data

    def generate(self, room_type_idx, start_deform=False):
        '''network forwarding'''
        latent_z = self.latent_input.module.sample_latent()
        est_data = self.generator.module.generate(latent_z, {'room_type_idx': room_type_idx}, start_deform=start_deform, self_end=True,
                                                  output_render=False)
        return est_data

    def test_step(self, data, start_deform=False, **kwargs):
        '''
        test by epoch
        '''
        '''load input and ground-truth data'''
        data = self.to_device(data)

        '''network forwarding'''
        latent_z = self.latent_input(data)
        est_data = self.generator.module.generate(latent_z, data, start_deform=start_deform, **kwargs)

        '''compute losses'''
        eval_metrics = self.evaluate_step(est_data)

        # for logging
        return eval_metrics, est_data

    def visualize_interp(self, phase, output_filename, est_data):
        if not self.cfg.is_master:
            return
        dump_dir = self.cfg.config.generation.dump_dir
        if not os.path.exists(dump_dir):
            os.makedirs(dump_dir)

        room_type = self.cfg.config[self.cfg.config.mode].room_type

        # prediction
        for batch_id, output in enumerate(est_data):
            box3ds = output['box3ds'].cpu().numpy()
            centers = box3ds[..., :3]
            sizes = box3ds[..., 3:6]
            category_ids = np.argmax(box3ds[..., 6:], axis=-1)
            meshes = output['posed_meshes']
            mesh_vertices = meshes.verts_padded().cpu().numpy()
            mesh_faces = meshes.faces_padded().cpu().numpy()
            dump_file = os.path.join(dump_dir, '%s/%s_%d.npz' % (room_type, output_filename, batch_id))
            if not os.path.exists(os.path.dirname(dump_file)):
                os.mkdir(os.path.dirname(dump_file))
            np.savez(dump_file,
                     category_ids=category_ids[0],
                     centers=centers[0],
                     sizes=sizes[0],
                     mesh_vertices=mesh_vertices,
                     mesh_faces=mesh_faces)

    def visualize_step(self, phase, iter, gt_data, est_data, dump_dir=None):
        ''' Performs a visualization step.
        '''
        if not self.cfg.is_master:
            return
        if dump_dir is None:
            dump_dir = self.cfg.config.generation.dump_dir
        if not os.path.exists(dump_dir):
            os.makedirs(dump_dir)

        # prediction
        for batch_id, output in enumerate(est_data):
            box3ds = output['box3ds'].cpu().numpy()
            centers = box3ds[..., :3]
            sizes = box3ds[..., 3:6]
            category_ids = np.argmax(box3ds[..., 6:], axis=-1)
            meshes = output['posed_meshes']
            points_2d = output['points_2d'][0].cpu().numpy()
            mesh_vertices = meshes.verts_padded().cpu().numpy()
            mesh_faces = meshes.faces_padded().cpu().numpy()
            if self.cfg.config.start_deform:
                silhouettes = output['silhouettes'][0].cpu().numpy()
                seg_obj_ids = output['obj_ids'][0].cpu().numpy()
            else:
                silhouettes = [None]
                seg_obj_ids= [None]
            dump_file = os.path.join(dump_dir, '%s_%s.npz' % (self.cfg.config.data.split_type, gt_data['sample_name'][batch_id][0]))
            np.savez(dump_file,
                     category_ids=category_ids[0],
                     centers=centers[0],
                     sizes=sizes[0],
                     points_2d=points_2d,
                     sample_names=gt_data['sample_name'][batch_id],
                     mesh_vertices=mesh_vertices,
                     mesh_faces=mesh_faces,
                     silhouettes=silhouettes,
                     seg_obj_ids=seg_obj_ids)