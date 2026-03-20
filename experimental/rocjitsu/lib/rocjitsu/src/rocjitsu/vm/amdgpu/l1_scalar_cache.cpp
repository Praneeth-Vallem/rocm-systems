// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/l1_scalar_cache.h"

#include "rocjitsu/vm/amdgpu/l2_cache.h"

#include <cstring>

namespace rocjitsu {
namespace amdgpu {

void L1ScalarCache::ensure_line(uint64_t addr) {
  if (cache_.lookup(addr))
    return;

  uint64_t line_addr = CacheStore::line_address(addr);
  simdojo::CacheTag evicted;
  uint8_t evicted_data[CacheStore::LINE_SIZE];
  cache_.allocate(addr, &evicted, evicted_data);

  // K$ is read-only - evicted lines are never dirty, no write-back needed.

  uint8_t line_buf[CacheStore::LINE_SIZE];
  l2_->read(line_addr, line_buf, CacheStore::LINE_SIZE);
  cache_.fill_line(addr, line_buf);
}

void L1ScalarCache::load(uint64_t addr, uint32_t num_dwords, uint32_t *dst) {
  for (uint32_t i = 0; i < num_dwords; ++i) {
    uint64_t ea = addr + i * 4;
    ensure_line(ea);

    uint8_t buf[4]{};
    cache_.read_line(ea, buf, CacheStore::line_offset(ea), 4);
    std::memcpy(&dst[i], buf, 4);
  }
}

} // namespace amdgpu
} // namespace rocjitsu
