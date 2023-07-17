#  Copyright (c) 5.2021. Yinyu Nie
#  License: MIT
import os
import random

import torch
import torch.utils.data
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import cv2
from torchvision import transforms
import numpy as np
import h5py
from models.datasets import Base_Dataset

default_collate = torch.utils.data.dataloader.default_collate


class THREEDFRONT(Base_Dataset):
    def __init__(self, cfg, mode):
        super(THREEDFRONT, self).__init__(cfg, mode)
        self.aug = mode == 'train' and cfg.config.mode == 'train'
        self.downsample_ratio = cfg.config.data.downsample_ratio
        self.permute_objs = lambda insts: self.permute(insts, None, 'random')
        # self.preprocess = transforms.Compose([
        #     transforms.Resize((224, 224)),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    @staticmethod
    def parse_hdf5(cfg, sample_file):
        '''read data'''
        cam_K = cfg.cam_K
        with h5py.File(sample_file, "r") as sample_data:
            # img = Image.fromarray(sample_data['colors'][:])
            # img = self.preprocess(img)
            img = None
            room_type = sample_data['room_type'][0].decode('ascii')
            cam_T = sample_data['cam_T'][:]
            inst_h5py = sample_data['inst_info']
            box2ds = []
            category_ids = []
            inst_marks = []
            masks = []
            for inst_id in inst_h5py:
                box2ds.append(inst_h5py[inst_id]['bbox2d'][:])
                category_ids.append(inst_h5py[inst_id]['category_id'][0])
                inst_marks.append(inst_h5py[inst_id]['inst_mark'][0].decode('ascii'))
                masks.append(inst_h5py[inst_id]['mask'][:])

        insts = {'box2ds': box2ds,
                 'category_ids': category_ids,
                 'inst_marks': inst_marks,
                 'masks': masks}

        room_type = [rm_type for rm_type in cfg.room_types if rm_type in room_type][0]
        return room_type, img, cam_K, cam_T, insts

    def load_unique_inst_marks(self, sample_path):
        '''read data'''
        room_name = '_'.join(os.path.basename(sample_path).split('_')[:-1])
        unique_inst_marks = self.cfg.unique_inst_mark[room_name]
        return unique_inst_marks

    def permute(self, insts, cls_importance=None, method='random'):
        '''
        Permute object ordering
        :param insts: objects
        :param cls_importance: [optional] permute objects by class importance if method == 'weighted';
        :param method: 'random', 'weighted' or 'fixed'
        :return: permuted objects
        '''
        if cls_importance is None:
            assert method == 'random'

        lengths = [len(value) for value in insts.values()]
        assert len(set(lengths)) == 1
        n_objs = len(insts['box2ds'])

        if method == 'random':
            ordering = np.random.permutation(n_objs)
        elif method == 'weighted':
            scores = [cls_importance[str(cls)] for cls in insts['category_ids']]
            ordering = sorted(list(range(n_objs)), key=lambda i: np.random.random() * scores[i], reverse=True)
            ordering = np.array(ordering)
        elif method == 'fixed':
            scores = np.array([cls_importance[str(cls)] for cls in insts['category_ids']])
            # for objects share the same category label, they are equal in ordering. so here a small noise is added.
            scores += 1e-6 * np.random.randn(n_objs)
            ordering = np.argsort(scores)[::-1]
        else:
            raise NotImplementedError

        for key in insts.keys():
            insts[key] = [insts[key][idx] for idx in ordering]

        return insts

    @staticmethod
    def track_insts(parsed_data, unique_marks):
        box2ds_template = -1 * np.ones(shape=(len(unique_marks), 4), dtype=np.int32)
        category_ids_template = np.zeros(shape=(len(unique_marks),), dtype=np.int32)
        masks_template = np.array([None] * len(unique_marks))
        inst_marks_template = np.zeros(shape=(len(unique_marks),), dtype=bool)
        for view_data in parsed_data:
            insts_data = view_data[3]
            ordering = [unique_marks.index(mark) for mark in insts_data['inst_marks']]
            # re-order instances
            # bbox2ds
            empty_box2ds = box2ds_template.copy()
            empty_box2ds[ordering, :] = insts_data['box2ds']
            insts_data['box2ds'] = empty_box2ds
            # category_ids
            empty_category_ids = category_ids_template.copy()
            empty_category_ids[ordering] = insts_data['category_ids']
            insts_data['category_ids'] = empty_category_ids
            # masks
            empty_masks = masks_template.copy()
            empty_masks[ordering] = insts_data['masks']
            insts_data['masks'] = empty_masks
            # inst_marks
            empty_inst_marks = inst_marks_template.copy()
            empty_inst_marks[ordering] = True
            insts_data['inst_marks'] = empty_inst_marks

        return parsed_data

    def read_sample(self, sample_name):
        room_type, img, cam_K, cam_T, insts = self.parse_hdf5(self.cfg, sample_name)
        return room_type, img, cam_K, cam_T, insts, self.cfg.image_size

    @staticmethod
    def get_view_data(cfg, parsed_view, downsample_ratio):
        img, cam_K, cam_T, insts, image_size, sample_name = parsed_view

        inst_marks = insts['inst_marks']
        n_objects = len(insts['box2ds'])
        box2ds = np.array(insts['box2ds'])
        category_ids = insts['category_ids']
        n_classes = len(cfg.label_names)
        category_labels = np.zeros(shape=(n_objects, n_classes))
        category_labels[range(n_objects), category_ids] = 1
        x1y1 = box2ds[..., :2]
        x2y2 = box2ds[..., :2] + box2ds[..., 2:4] - 1
        inst_box2ds = np.concatenate([x1y1, x2y2, category_labels], axis=-1)
        inst_masks = -1 * np.ones((int(image_size[1]), int(image_size[0])), dtype=int)
        render_mask = np.ones_like(inst_masks)

        for inst_id, (box2d, mask) in enumerate(zip(insts['box2ds'], insts['masks'])):
            if not inst_marks[inst_id]:
                continue
            current_block = inst_masks[box2d[1]: box2d[1] + box2d[3], box2d[0]: box2d[0] + box2d[2]]
            current_block[mask == True] = inst_id

        # pad to self.cfg.image_size
        render_image_size = np.array(cfg.image_size)
        if (render_image_size != image_size).any():
            scale_ratio = render_image_size / image_size
            to_size = np.int32(image_size * scale_ratio.min())
            inst_masks = cv2.resize(inst_masks, to_size, interpolation=cv2.INTER_NEAREST_EXACT)
            render_mask = np.ones_like(inst_masks)
            long_axis = np.argmax(scale_ratio)
            padding = render_image_size[long_axis] - to_size[long_axis]
            if long_axis == 0:
                pad_item = ((0, 0), (padding // 2, padding - padding // 2))
            else:
                pad_item = ((padding // 2, padding - padding // 2), (0, 0))
            inst_masks = np.pad(inst_masks, pad_item, 'constant', constant_values=((-1, -1), (-1, -1)))
            render_mask = np.pad(render_mask, pad_item, 'constant', constant_values=((0, 0), (0, 0)))

        resize_w = int(render_image_size[0]) // downsample_ratio
        resize_h = int(render_image_size[1]) // downsample_ratio
        inst_masks = cv2.resize(inst_masks, (resize_w, resize_h), interpolation=cv2.INTER_NEAREST_EXACT)
        render_mask = cv2.resize(render_mask, (resize_w, resize_h), interpolation=cv2.INTER_NEAREST_EXACT)

        '''store gt data'''
        data = {}
        data['sample_name'] = sample_name
        # data['img'] = img
        data['cam_K'] = cam_K.astype(np.float32)
        data['image_size'] = image_size.astype(np.float32)
        data['cam_T'] = cam_T.astype(np.float32)
        data['box2ds_tr'] = inst_box2ds.astype(np.float32)
        data['masks_tr'] = inst_masks.astype(np.int64)
        data['render_mask_tr'] = render_mask.astype(bool)
        data['inst_marks'] = np.array(inst_marks, dtype=bool)
        return data

    def __getitem__(self, idx):
        '''Get each sample'''
        # sample main view
        candidates = self.split[self.room_uids[idx]]
        aug_id = int(self.room_uids[idx].split('_')[-1])

        # augment data
        theta = [0, 0.5 * np.pi, np.pi, 1.5 * np.pi][aug_id]
        rot_mat = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
        trans_mat = np.eye(4)
        trans_mat[:3, :3] = rot_mat

        if self.cfg.config.data.n_views < len(candidates):
            view_ids = np.random.choice(len(candidates), self.cfg.config.data.n_views, replace=False)
        else:
            view_ids = np.random.choice(len(candidates), self.cfg.config.data.n_views-len(candidates), replace=True)
            view_ids = np.append(view_ids, list(range(len(candidates))))
        view_files = [os.path.join(self.cfg.config.root_dir, candidates[idx]) for idx in view_ids]

        parsed_data = []
        for view_file in view_files:
            room_type, img, cam_K, cam_T, insts, image_size = self.read_sample(view_file)
            cam_T = trans_mat.dot(cam_T)
            sample_name = '.'.join(os.path.basename(view_file).split('.')[:-1])
            parsed_data.append((img, cam_K, cam_T, insts, image_size, sample_name))

        room_type_idx = 0

        unique_marks = self.load_unique_inst_marks(view_files[0])

        # shuffle instance ordering, since objects in a room should be permutation-invariant
        if self.aug:
            random.shuffle(unique_marks)

        # re-organize instances following track ids
        parsed_data = self.track_insts(parsed_data, unique_marks)

        keywords = ['sample_name', 'cam_K', 'image_size', 'cam_T', 'box2ds_tr', 'inst_marks']

        if self.cfg.config.start_deform:
            keywords.append('masks_tr')
            if self.cfg.config.data.dataset == 'ScanNet':
                keywords.append('render_mask_tr')

        views_data = []
        for parsed_view in parsed_data:
            view_data = self.get_view_data(self.cfg, parsed_view, self.downsample_ratio)
            view_data = {**{k: view_data[k] for k in keywords},
                         **{'room_idx': idx, 'max_len': len(unique_marks), 'room_type_idx': room_type_idx}}
            views_data.append(view_data)

        return views_data

class ScanNet(THREEDFRONT):
    def __init__(self, cfg, mode):
        super(ScanNet, self).__init__(cfg, mode)
        self.permute_objs = lambda insts: self.permute(insts, None, 'random')
        # self.preprocess = transforms.Compose([
        #     transforms.Resize((224, 224)),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def load_unique_inst_marks(self, sample_path):
        '''read data'''
        with h5py.File(sample_path, "r") as sample_data:
            unique_inst_marks = sample_data['unique_inst_marks'][:].tolist()
        return unique_inst_marks

    @staticmethod
    def parse_hdf5(cfg, sample_file):
        '''read data'''
        with h5py.File(sample_file, "r") as sample_data:
            # img = Image.fromarray(sample_data['colors'][:])
            # img = self.preprocess(img)
            img = None
            room_type = sample_data['room_type'][0].decode('ascii')
            cam_T = sample_data['cam_T'][:]
            cam_K = sample_data['cam_K'][:]
            image_size = sample_data['image_size'][:]
            inst_h5py = sample_data['inst_info']
            box2ds = []
            category_ids = []
            inst_marks = []
            masks = []
            for inst_id in inst_h5py:
                box2ds.append(inst_h5py[inst_id]['bbox2d'][:])
                category_ids.append(inst_h5py[inst_id]['category_id'][0])
                inst_marks.append(inst_h5py[inst_id]['inst_mark'][0])
                masks.append(inst_h5py[inst_id]['mask'][:])

        insts = {'box2ds': box2ds,
                 'category_ids': category_ids,
                 'inst_marks': inst_marks,
                 'masks': masks}
        return room_type, img, cam_K, cam_T, insts, image_size

    def read_sample(self, sample_name):
        room_type, img, cam_K, cam_T, insts, image_size = self.parse_hdf5(self.cfg, sample_name)
        return room_type, img, cam_K, cam_T, insts, image_size

# Init datasets and dataloaders
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def collate_fn(samples):
    padding_keys = ['box2ds_tr']
    room_level_keys = ['inst_marks']
    max_length = max(sample[0]['max_len'] for sample in samples)

    collated_batch = {}
    for key in samples[0][0]:
        collated_batch[key] = []
        for sample in samples:  # n_batch samples
            if key in room_level_keys:
                collated_view_data = default_collate([np.append(view_data[key], np.zeros(
                    (max_length - len(view_data[key]), *view_data[key].shape[1:]), dtype=view_data[key].dtype)) for
                                                      view_data in sample])
            elif key not in padding_keys:
                collated_view_data = default_collate([view_data[key] for view_data in sample])
            else:
                collated_view_data = default_collate([np.vstack([view_data[key], np.zeros(
                    (max_length - len(view_data[key]), *view_data[key].shape[1:]), dtype=view_data[key].dtype)]) for view_data in
                                                      sample])
            collated_batch[key].append(collated_view_data)

    for key in collated_batch:
        if key not in ['sample_name']:
            collated_batch[key] = default_collate(collated_batch[key])

    return collated_batch


# Init datasets and dataloaders
def our_dataloader(cfg, mode='train'):
    if cfg.config.data.dataset == '3D-Front':
        dataset = THREEDFRONT(cfg=cfg, mode=mode)
    elif cfg.config.data.dataset == 'ScanNet':
        dataset = ScanNet(cfg=cfg, mode=mode)
    else:
        raise NotImplementedError

    if cfg.config.distributed.num_gpus > 1:
        sampler = DistributedSampler(dataset, shuffle=(mode == 'train'))
    else:
        if mode == 'train':
            sampler = torch.utils.data.RandomSampler(dataset)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)

    batch_sampler = torch.utils.data.BatchSampler(sampler, batch_size=cfg.config[mode].batch_size // cfg.config.distributed.num_gpus,
                                                  drop_last=True)

    dataloader = DataLoader(dataset=dataset,
                            batch_sampler=batch_sampler,
                            num_workers=cfg.config.device.num_workers,
                            collate_fn=collate_fn,
                            worker_init_fn=my_worker_init_fn)
    return dataloader
