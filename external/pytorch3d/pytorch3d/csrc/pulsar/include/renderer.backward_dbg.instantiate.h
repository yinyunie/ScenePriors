/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "./renderer.backward_dbg.device.h"

namespace pulsar {
namespace Renderer {

template void backward_dbg<ISONDEVICE>(
    Renderer* self,
    const float* grad_im,
    const float* image,
    const float* forw_info,
    const float* vert_pos,
    const float* vert_col,
    const float* vert_rad,
    const CamInfo& cam,
    const float& gamma,
    float percent_allowed_difference,
    const uint& max_n_hits,
    const float* vert_opy,
    const size_t& num_balls,
    const uint& mode,
    const bool& dif_pos,
    const bool& dif_col,
    const bool& dif_rad,
    const bool& dif_cam,
    const bool& dif_opy,
    const uint& pos_x,
    const uint& pos_y,
    cudaStream_t stream);

} // namespace Renderer
} // namespace pulsar
