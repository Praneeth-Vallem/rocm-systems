// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/memory_pipeline.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/l1_scalar_cache.h"
#include "rocjitsu/vm/amdgpu/l1_vector_cache.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/lds.h"
#include "rocjitsu/vm/amdgpu/mem_state.h"

#include <cstring>

namespace rocjitsu {
namespace amdgpu {

namespace {

/// Shared complete_access logic for vector/LDS loads (write VGPRs from
/// response data). Used by both GlobalMemPipeline and LocalMemPipeline.
void vector_complete(VectorMemState &d, ComputeUnitCore &cu) {
  if (!d.is_load)
    return;
  uint32_t stride = d.num_elems * d.elem_size;
  for (uint32_t lane = 0; lane < 64; ++lane) {
    if (!(d.lane_mask & (1ULL << lane)))
      continue;
    for (uint32_t i = 0; i < d.num_elems; ++i) {
      uint32_t val = 0;
      std::memcpy(&val, &d.response_data[lane * stride + i * d.elem_size], d.elem_size);
      cu.write_vgpr(d.dst_reg_base + i, lane, val);
    }
  }
}

} // namespace

void ScalarMemPipeline::initiate_access(Instruction &inst, Wavefront & /*wf*/) {
  auto &d = *inst.data_as<ScalarMemState>();
  if (d.is_load) {
    l1_->load(d.addr, d.num_dwords, d.response_data);
  } else {
    for (uint32_t i = 0; i < d.num_dwords; ++i) {
      uint8_t buf[4];
      std::memcpy(buf, &d.store_data[i], 4);
      l2_->write(d.addr + i * 4, buf, 4);
    }
  }
}

void ScalarMemPipeline::complete_access(Instruction &inst, Wavefront &wf) {
  auto &d = *inst.data_as<ScalarMemState>();
  if (!d.is_load)
    return;
  auto &cu = wf.cu();
  for (uint32_t i = 0; i < d.num_dwords; ++i)
    cu.write_sgpr(d.dst_reg_base + i, d.response_data[i]);
}

void GlobalMemPipeline::initiate_access(Instruction &inst, Wavefront & /*wf*/) {
  auto &d = *inst.data_as<VectorMemState>();
  if (d.is_load) {
    d.response_data.resize(64 * d.num_elems * d.elem_size);
    l1_->load(d.per_lane_addr.data(), d.lane_mask, d.elem_size, d.num_elems, d.response_data.data(),
              d.mtype, d.non_temporal);
  } else {
    l1_->store(d.per_lane_addr.data(), d.lane_mask, d.elem_size, d.num_elems, d.store_data.data(),
               d.mtype, d.non_temporal);
  }
}

void GlobalMemPipeline::complete_access(Instruction &inst, Wavefront &wf) {
  vector_complete(*inst.data_as<VectorMemState>(), wf.cu());
}

void LocalMemPipeline::initiate_access(Instruction &inst, Wavefront & /*wf*/) {
  auto &d = *inst.data_as<VectorMemState>();
  if (d.is_load) {
    d.response_data.resize(64 * d.num_elems * d.elem_size);
    lds_->vector_load(d.per_lane_addr.data(), d.lane_mask, d.elem_size, d.num_elems,
                      d.response_data.data());
  } else {
    lds_->vector_store(d.per_lane_addr.data(), d.lane_mask, d.elem_size, d.num_elems,
                       d.store_data.data());
  }
}

void LocalMemPipeline::complete_access(Instruction &inst, Wavefront &wf) {
  vector_complete(*inst.data_as<VectorMemState>(), wf.cu());
}

} // namespace amdgpu
} // namespace rocjitsu
