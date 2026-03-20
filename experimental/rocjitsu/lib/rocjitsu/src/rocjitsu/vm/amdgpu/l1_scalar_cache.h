// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_VM_AMDGPU_L1_SCALAR_CACHE_H_
#define ROCJITSU_VM_AMDGPU_L1_SCALAR_CACHE_H_

#include "simdojo/components/cache.h"

#include <cstdint>
#include <cstring>

namespace rocjitsu {
namespace amdgpu {

class L2CacheController;

/// @brief L1 Scalar Cache (K$) controller for SMEM instructions.
///
/// Read-only, 16KB, 64-byte lines, 4-way set-associative with LRU.
/// On hit, returns data directly. On miss, fetches from L2, fills the
/// line, and returns data. The `s_dcache_inv` instruction invalidates
/// all lines.
///
/// CDNA3 K$ geometry: 64B lines, 64 sets, 4-way = 16KB.
class L1ScalarCache {
public:
  static constexpr uint32_t LINE_SIZE_BITS = 6; // 64 bytes
  static constexpr uint32_t NUM_SETS = 64;
  static constexpr uint32_t ASSOCIATIVITY = 4;

  using CacheStore = simdojo::Cache<LINE_SIZE_BITS, NUM_SETS, ASSOCIATIVITY>;

  explicit L1ScalarCache(L2CacheController *l2) : l2_(l2) {}

  /// @brief Scalar load: read num_dwords contiguous dwords from addr.
  ///
  /// Fetches from K$ on hit, or fills from L2 on miss. Handles requests
  /// that span multiple cache lines.
  void load(uint64_t addr, uint32_t num_dwords, uint32_t *dst);

  /// @brief Invalidate all K$ lines (s_dcache_inv).
  void invalidate_all() { cache_.invalidate_all(); }

private:
  /// @brief Ensure the cache line containing addr is present, fetching from L2 if needed.
  void ensure_line(uint64_t addr);

  CacheStore cache_;
  L2CacheController *l2_;
};

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_VM_AMDGPU_L1_SCALAR_CACHE_H_
