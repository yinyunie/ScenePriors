# 
# Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
# Licensed under the NVIDIA Source Code License.
# See LICENSE at https://github.com/nv-tlabs/ATISS.
# Authors: Despoina Paschalidou, Amlan Kar, Maria Shugrina, Karsten Kreis,
#          Andreas Geiger, Sanja Fidler
# 

import numpy as np
import pickle
import os
import trimesh
from pathlib import Path
import torch
from pytorch3d.loss import chamfer_distance

from .utils import parse_threed_future_models


class ThreedFutureDataset(object):
    def __init__(self, objects):
        assert len(objects) > 0
        self.objects = objects

    def __len__(self):
        return len(self.objects)

    def __str__(self):
        return "Dataset contains objects with {} discrete types".format(
            len(self)
        )

    def __getitem__(self, idx):
        return self.objects[idx]

    def _filter_objects_by_label(self, label, generic_mapping):
        return [oi for oi in self.objects if oi.label in generic_mapping and generic_mapping[oi.label] == label]

    def get_closest_furniture_to_box(self, query_label, query_vertices, generic_mapping):
        objects = self._filter_objects_by_label(query_label, generic_mapping)

        assert len(objects)

        query_vertices = query_vertices/ (query_vertices.max() * 2)

        per_rot_mat = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])

        all_key_vertices = []
        valid_objects = []
        for i, oi in enumerate(objects):
            vertice_path = Path(os.path.join('./temp/3D_Future_vertices', oi.model_jid + '.npy'))
            if vertice_path.exists():
                key_vertices = np.load(str(vertice_path))
            else:
                try:
                    if not vertice_path.parent.exists():
                        vertice_path.parent.mkdir(parents=True)
                    key_vertices = trimesh.load(oi.raw_model_path, force='mesh').sample(5000)
                    np.save(str(vertice_path), key_vertices)
                except:
                    print('Failed to load %s.' % (str(oi.raw_model_path)))
                    continue

            key_vertices = key_vertices * oi.scale
            key_lbdb = key_vertices.min(axis=0)
            key_ubdb = key_vertices.max(axis=0)
            key_centroid = (key_lbdb + key_ubdb) / 2.
            key_vertices = key_vertices - key_centroid

            key_vertices = key_vertices / (key_vertices.max() * 2)

            all_key_vertices.append(key_vertices)
            valid_objects.append(oi)

        all_key_vertices = np.array(all_key_vertices)

        all_key_vertices_w_rot = [all_key_vertices]
        for i in range(3):
            all_key_vertices_w_rot.append(all_key_vertices_w_rot[-1].dot(per_rot_mat))

        # 4 x n_objs
        all_key_vertices_w_rot = torch.from_numpy(np.array(all_key_vertices_w_rot)).cuda().float()
        all_key_vertices_w_rot = all_key_vertices_w_rot.flatten(0, 1)

        query_vertices = torch.from_numpy(query_vertices)[None].expand(all_key_vertices_w_rot.size(0), -1, -1).cuda().float()

        cham_dist, cham_normals = chamfer_distance(query_vertices, all_key_vertices_w_rot, batch_reduction=None,
                                                   point_reduction='mean', norm=1)
        best_idx = cham_dist.argmin().item()
        rot_idx = best_idx // len(valid_objects)
        obj_idx = best_idx % len(valid_objects)

        return valid_objects[obj_idx], np.linalg.matrix_power(per_rot_mat, rot_idx)

    def get_closest_furniture_to_2dbox(self, query_label, query_size):
        objects = self._filter_objects_by_label(query_label)

        mses = {}
        for i, oi in enumerate(objects):
            mses[oi] = (
                (oi.size[0] - query_size[0])**2 +
                (oi.size[2] - query_size[1])**2
            )
        sorted_mses = [k for k, v in sorted(mses.items(), key=lambda x: x[1])]
        return sorted_mses[0]

    @classmethod
    def from_dataset_directory(
        cls, dataset_directory, path_to_model_info, path_to_models, path_to_3d_future_objects
    ):
        objects = parse_threed_future_models(
            dataset_directory, path_to_models, path_to_model_info, path_to_3d_future_objects
        )
        return cls(objects)

    @classmethod
    def from_pickled_dataset(cls, path_to_pickled_dataset):
        with open(path_to_pickled_dataset, "rb") as f:
            dataset = pickle.load(f)
        return dataset
