# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from itertools import product

import torch
from fvcore.common.benchmark import benchmark
from tests.test_mesh_normal_consistency import TestMeshNormalConsistency


def bm_mesh_normal_consistency() -> None:
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")

    kwargs_list = []
    num_meshes = [16, 32, 64]
    levels = [2, 3]
    test_cases = product(num_meshes, levels, devices)
    for case in test_cases:
        n, l, d = case
        kwargs_list.append({"num_meshes": n, "level": l, "device": d})

    benchmark(
        TestMeshNormalConsistency.mesh_normal_consistency_with_ico,
        "MESH_NORMAL_CONSISTENCY_ICO",
        kwargs_list,
        warmup_iters=1,
    )


if __name__ == "__main__":
    bm_mesh_normal_consistency()
