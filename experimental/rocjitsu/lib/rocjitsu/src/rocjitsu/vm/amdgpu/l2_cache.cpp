// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/l2_cache.h"

#include <bit>
#include <cassert>
#include <cstring>

namespace rocjitsu {
namespace amdgpu {

void L2CacheController::ensure_line(uint64_t addr) {
  if (cache_.lookup(addr))
    return;

  uint64_t line_addr = CacheStore::line_address(addr);
  simdojo::CacheTag evicted;
  uint8_t evicted_data[LINE_SIZE];
  cache_.allocate(addr, &evicted, evicted_data);

  if (evicted.valid && evicted.dirty) {
    static constexpr uint32_t SET_INDEX_BITS = std::bit_width(NUM_SETS - 1);
    uint64_t evicted_addr = (evicted.tag << (LINE_SIZE_BITS + SET_INDEX_BITS)) |
                            (static_cast<uint64_t>(CacheStore::set_index(addr)) << LINE_SIZE_BITS);
    backing_->write(evicted_addr, evicted_data, LINE_SIZE);
  }

  uint8_t line_buf[LINE_SIZE];
  backing_->read(line_addr, line_buf, LINE_SIZE);
  cache_.fill_line(addr, line_buf);
}

void L2CacheController::read(uint64_t addr, uint8_t *dst, uint32_t size, Mtype mtype) {
  if (mtype == Mtype::UC) {
    // Uncacheable: bypass L2, read directly from HBM.
    backing_->read(addr, dst, size);
    return;
  }

  ensure_line(addr);
  cache_.read_line(addr, dst, CacheStore::line_offset(addr), size);
}

void L2CacheController::write(uint64_t addr, const uint8_t *src, uint32_t size, Mtype mtype) {
  if (mtype == Mtype::UC) {
    // Uncacheable: bypass L2, write directly to HBM.
    backing_->write(addr, src, size);
    return;
  }

  ensure_line(addr);
  cache_.write_line(addr, src, CacheStore::line_offset(addr), size);

  simdojo::CacheTag *tag = nullptr;
  cache_.lookup(addr, &tag);
  assert(tag != nullptr && "ensure_line must guarantee hit");

  if (mtype == Mtype::CC) {
    // Coherently Cacheable: write-through to HBM.
    backing_->write(addr, src, size);
    tag->coherence = simdojo::CoherenceState::SHARED;
  } else {
    // Read-Write / Write-Back: mark dirty, defer write-back to eviction.
    tag->dirty = true;
    tag->coherence = simdojo::CoherenceState::MODIFIED;
  }
}

void L2CacheController::fetch_line(uint64_t addr, uint8_t *line_buf) {
  uint64_t line_addr = CacheStore::line_address(addr);
  ensure_line(line_addr);
  cache_.read_line(line_addr, line_buf, 0, LINE_SIZE);
}

void L2CacheController::writeback_line(uint64_t line_addr, const uint8_t *data, Mtype mtype) {
  // Write the line data into L2.
  simdojo::CacheTag *tag = nullptr;
  if (cache_.lookup(line_addr, &tag)) {
    cache_.write_line(line_addr, data, 0, LINE_SIZE);
  } else {
    simdojo::CacheTag evicted;
    uint8_t evicted_data[LINE_SIZE];
    cache_.allocate(line_addr, &evicted, evicted_data);

    if (evicted.valid && evicted.dirty) {
      static constexpr uint32_t SET_INDEX_BITS = std::bit_width(NUM_SETS - 1);
      uint64_t evicted_addr =
          (evicted.tag << (LINE_SIZE_BITS + SET_INDEX_BITS)) |
          (static_cast<uint64_t>(CacheStore::set_index(line_addr)) << LINE_SIZE_BITS);
      backing_->write(evicted_addr, evicted_data, LINE_SIZE);
    }

    cache_.fill_line(line_addr, data);
    cache_.lookup(line_addr, &tag);
  }

  if (mtype == Mtype::CC) {
    // Write-through to HBM for coherent accesses.
    backing_->write(line_addr, data, LINE_SIZE);
    tag->coherence = simdojo::CoherenceState::SHARED;
  } else {
    tag->dirty = true;
    tag->coherence = simdojo::CoherenceState::MODIFIED;
  }
}

void L2CacheController::flush_all() {
  cache_.for_each_dirty([this](simdojo::CacheTag &tag, uint64_t line_addr, uint8_t *data) {
    backing_->write(line_addr, data, LINE_SIZE);
    tag.dirty = false;
  });
  cache_.invalidate_all();
}

} // namespace amdgpu
} // namespace rocjitsu
