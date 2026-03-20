// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_VM_AMDGPU_L1_VECTOR_CACHE_H_
#define ROCJITSU_VM_AMDGPU_L1_VECTOR_CACHE_H_

#include "rocjitsu/vm/amdgpu/mtype.h"
#include "simdojo/components/cache.h"

#include <cstdint>

namespace rocjitsu {
namespace amdgpu {

class L2CacheController;

/// @brief L1 Vector Cache (V$) controller for FLAT/MUBUF/MTBUF instructions.
///
/// @details 32KB, 128-byte lines, 4-way set-associative with LRU. All cacheable
/// stores use write-through to L2 so that partial writes from different CUs
/// sharing the same L2 are properly merged at byte granularity. Memory type
/// (Mtype) determines caching behavior:
///
///   - RW/WB: L1 write-through - allocate on miss, writes go through to L2.
///   - CC: L1 write-through - allocate on miss, writes go through to L2.
///   - UC: Bypass L1 entirely - forward to L2 (which may also bypass).
///   - NT (non-temporal): Bypass L1, L2-only caching.
///
/// CDNA3 V$ geometry: 128B lines, 64 sets, 4-way = 32KB.
class L1VectorCache {
public:
  static constexpr uint32_t LINE_SIZE_BITS = 7; // 128 bytes
  static constexpr uint32_t NUM_SETS = 64;
  static constexpr uint32_t ASSOCIATIVITY = 4;

  using CacheStore = simdojo::Cache<LINE_SIZE_BITS, NUM_SETS, ASSOCIATIVITY>;
  static constexpr uint32_t LINE_SIZE = CacheStore::LINE_SIZE;

  explicit L1VectorCache(L2CacheController *l2) : l2_(l2) {}

  /// @brief Vector load: per-lane load with mtype-aware caching.
  ///
  /// @param addrs Per-lane base addresses (64 entries).
  /// @param lane_mask EXEC mask indicating active lanes.
  /// @param elem_size Bytes per element (1, 2, 4, or 8).
  /// @param num_elems Elements per lane (1-4).
  /// @param dst Output buffer: [lane][elem], stride = num_elems * elem_size per lane.
  /// @param mtype Memory type controlling caching behavior.
  /// @param non_temporal If true, bypass L1 regardless of mtype.
  void load(const uint64_t *addrs, uint64_t lane_mask, uint32_t elem_size, uint32_t num_elems,
            uint8_t *dst, Mtype mtype, bool non_temporal);

  /// @brief Vector store: per-lane store with mtype-aware caching.
  void store(const uint64_t *addrs, uint64_t lane_mask, uint32_t elem_size, uint32_t num_elems,
             const uint8_t *src, Mtype mtype, bool non_temporal);

  /// @brief Invalidate all V$ lines.
  void invalidate_all() { cache_.invalidate_all(); }

  /// @brief Invalidate all V$ lines.
  ///
  /// @details With write-through policy, L1 lines are never dirty, so no
  /// writeback is needed. This simply invalidates the entire cache.
  void flush_all();

private:
  /// @brief Read bytes from cache, handling L1 hit/miss based on mtype.
  void read_bytes(uint64_t addr, uint8_t *dst, uint32_t size, Mtype mtype, bool non_temporal);

  /// @brief Write bytes to cache with mtype-aware policy.
  void write_bytes(uint64_t addr, const uint8_t *src, uint32_t size, Mtype mtype,
                   bool non_temporal);

  /// @brief Ensure the cache line containing addr is present, fetching from L2 if needed.
  void ensure_line(uint64_t addr);

  CacheStore cache_;
  L2CacheController *l2_;
};

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_VM_AMDGPU_L1_VECTOR_CACHE_H_
