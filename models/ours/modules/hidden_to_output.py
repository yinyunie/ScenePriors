# 
# Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
# Licensed under the NVIDIA Source Code License.
# See LICENSE at https://github.com/nv-tlabs/ATISS.
# Authors: Despoina Paschalidou, Amlan Kar, Maria Shugrina, Karsten Kreis,
#          Andreas Geiger, Sanja Fidler
# 

import torch
import torch.nn as nn


class Hidden2Output(nn.Module):
    def __init__(self, hidden_size, n_classes, with_extra_fc=False):
        super().__init__()
        self.with_extra_fc = with_extra_fc
        self.n_classes = n_classes
        self.hidden_size = hidden_size

        if with_extra_fc:
            mlp_layers = [
                nn.Linear(hidden_size, 2*hidden_size),
                nn.ReLU(),
                nn.Linear(2*hidden_size, hidden_size),
                nn.ReLU()
            ]
            self.hidden2output = nn.Sequential(*mlp_layers)

    def forward(self, x, sample_params=None):
        raise NotImplementedError()


class DeterminsticOutput(Hidden2Output):
    def __init__(
        self,
        hidden_size,
        n_classes,
        with_extra_fc=False,
    ):
        super(DeterminsticOutput, self).__init__(hidden_size, n_classes, with_extra_fc)
        self.class_layer = nn.Linear(hidden_size, n_classes)

        self.centroid_layer_x = DeterminsticOutput._mlp(hidden_size, 1)
        self.centroid_layer_y = nn.Sequential(DeterminsticOutput._mlp(hidden_size, 1),
                                              nn.ReLU(inplace=True))
        self.centroid_layer_z = DeterminsticOutput._mlp(hidden_size, 1)
        self.size_layer_x = nn.Sequential(DeterminsticOutput._mlp(hidden_size, 1),
                                          nn.Softplus())
        self.size_layer_y = nn.Sequential(DeterminsticOutput._mlp(hidden_size, 1),
                                          nn.Softplus())
        self.size_layer_z = nn.Sequential(DeterminsticOutput._mlp(hidden_size, 1),
                                          nn.Softplus())

    @staticmethod
    def _mlp(hidden_size, output_size):
        mlp_layers = [
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        ]
        return nn.Sequential(*mlp_layers)

    def forward(self, x, *args, **kwargs):
        if self.with_extra_fc:
            x = self.hidden2output(x)

        # Extract the target properties from sample_params and embed them into
        # a higher dimensional space.
        pred_cls = self.class_layer(x)

        # Using the true class label we now want to predict the sizes and translations
        size_x = self.size_layer_x(x)
        size_y = self.size_layer_y(x)
        size_z = self.size_layer_z(x)

        centroid_x = self.centroid_layer_x(x)
        centroid_y = self.centroid_layer_y(x)
        centroid_z = self.centroid_layer_z(x)

        centroid_y = centroid_y + size_y / 2.

        pred_translations = torch.cat([centroid_x, centroid_y, centroid_z], dim=-1)
        pred_sizes = torch.cat([size_x, size_y, size_z], dim=-1)

        return torch.cat([pred_translations, pred_sizes, pred_cls], dim=-1)
