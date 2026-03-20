// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file l2_cache.h
/// @brief L2 cache controller and L2 cache component shared per XCD.

#ifndef ROCJITSU_VM_AMDGPU_L2_CACHE_H_
#define ROCJITSU_VM_AMDGPU_L2_CACHE_H_

#include "rocjitsu/vm/amdgpu/mtype.h"
#include "simdojo/components/cache.h"
#include "simdojo/components/memory_interface.h"
#include "simdojo/sim/component.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

namespace rocjitsu {
namespace amdgpu {

/// @brief L2 Cache controller shared per XCD.
///
/// 128-byte lines, 2048 sets, 16-way set-associative = 4MB (default).
/// Write-back to the backing MemoryInterface (HBM controller or memory-side
/// cache, depending on topology).
///
/// Mtype-aware behavior:
///   - UC: Bypass L2, forward directly to backing store.
///   - CC: Allocate in L2, MOESI coherence state tracking for CPU-GPU
///         shared memory. Write-through on CC stores.
///   - RW/WB: Allocate in L2, write-back on eviction.
///   - NT: Allocate in L2 (L1 is bypassed, but L2 still caches).
///
/// Serves as the backing store for both L1 Scalar (K$) and L1 Vector (V$).
class L2CacheController {
public:
  static constexpr uint32_t LINE_SIZE_BITS = 7; // 128 bytes
  static constexpr uint32_t NUM_SETS = 2048;
  static constexpr uint32_t ASSOCIATIVITY = 16;

  using CacheStore = simdojo::Cache<LINE_SIZE_BITS, NUM_SETS, ASSOCIATIVITY>;
  static constexpr uint32_t LINE_SIZE = CacheStore::LINE_SIZE;

  /// @brief Construct with a backing memory interface.
  /// @param backing Backing store (HBM controller or memory-side cache, not owned).
  explicit L2CacheController(simdojo::MemoryInterface *backing) : backing_(backing) {}

  /// @brief Read a cache line worth of data (or partial line).
  ///
  /// Used by L1 controllers to fetch on miss. Always checks L2 first,
  /// then falls through to HBM on L2 miss.
  /// @param addr The starting address (line-aligned or not).
  /// @param dst Destination buffer.
  /// @param size Number of bytes to read.
  /// @param mtype Memory type for caching policy.
  void read(uint64_t addr, uint8_t *dst, uint32_t size, Mtype mtype = Mtype::RW);

  /// @brief Write data to L2 (and possibly through to HBM).
  ///
  /// Used by L1 for write-through (CC) and write-back evictions.
  /// @param addr The starting address.
  /// @param src Source data.
  /// @param size Number of bytes to write.
  /// @param mtype Memory type for caching policy.
  void write(uint64_t addr, const uint8_t *src, uint32_t size, Mtype mtype = Mtype::RW);

  /// @brief Fetch an entire cache line into the given buffer.
  ///
  /// Convenience method for L1 fills. Returns a full LINE_SIZE-byte line
  /// at the line-aligned address containing addr.
  /// @param addr Any address within the desired cache line.
  /// @param[out] line_buf Buffer of at least LINE_SIZE bytes.
  void fetch_line(uint64_t addr, uint8_t *line_buf);

  /// @brief Write back a full cache line from L1 eviction.
  /// @param line_addr Line-aligned address.
  /// @param[in] data Full cache line data (LINE_SIZE bytes).
  /// @param mtype Memory type for caching policy.
  void writeback_line(uint64_t line_addr, const uint8_t *data, Mtype mtype = Mtype::RW);

  /// @brief Invalidate all L2 lines.
  void invalidate_all() { cache_.invalidate_all(); }

  /// @brief Flush all dirty L2 lines to HBM and invalidate.
  void flush_all();

private:
  /// @brief Ensure the L2 line for addr is present, fetching from HBM on miss.
  /// @param addr Any address within the desired cache line.
  void ensure_line(uint64_t addr);

  CacheStore cache_;
  simdojo::MemoryInterface *backing_;
};

/// @brief L2 cache as a simdojo Component, shared per Accelerator Complex Die (XCD).
///
/// Owns the L2CacheController. The backing MemoryInterface (HBM controller
/// or memory-side cache) is provided at construction and not owned.
/// Provides structural ports for the topology graph (IN for CU L1 miss
/// requests, OUT for HBM/fabric traffic). In functional mode, CUs access
/// the L2 via direct method calls on the controller(); ports carry no data.
class L2Cache : public simdojo::Component {
public:
  /// @brief Construct an L2Cache component.
  /// @param name Human-readable name (e.g., "xcd0.l2").
  /// @param backing Backing MemoryInterface (HbmController or MSC, not owned).
  L2Cache(std::string name, simdojo::MemoryInterface *backing)
      : simdojo::Component(std::move(name)), controller_(backing) {
    hbm_out_ = add_port(std::make_unique<simdojo::Port>(
        "hbm_out", 0, this, simdojo::PortDirection::OUT, simdojo::PortProtocol::MEMORY_REQ));
  }

  /// @brief Return the L2 cache controller for direct-call access.
  /// @returns Pointer to the L2 cache controller.
  L2CacheController *controller() { return &controller_; }
  /// @returns Const pointer to the L2 cache controller.
  const L2CacheController *controller() const { return &controller_; }

  /// @brief Create an L2 IN port for a CU connection (one per CU).
  /// @param cu_name Name of the connecting CU (used for port naming).
  /// @returns Pointer to the newly created input port.
  simdojo::Port *create_cu_in_port(const std::string &cu_name) {
    auto port_id = static_cast<simdojo::PortID>(l2_in_ports_.size() + 1);
    auto port = std::make_unique<simdojo::Port>("l2_in_" + cu_name, port_id, this,
                                                simdojo::PortDirection::IN,
                                                simdojo::PortProtocol::MEMORY_REQ);
    auto *raw = add_port(std::move(port));
    l2_in_ports_.push_back(raw);
    return raw;
  }

  /// @brief Return the HBM/fabric output port (for topology wiring).
  /// @returns Pointer to the HBM output port.
  simdojo::Port *hbm_out_port() { return hbm_out_; }

  /// @brief Return all CU-facing IN ports.
  /// @returns Const reference to the vector of L2 input ports.
  const std::vector<simdojo::Port *> &l2_in_ports() const { return l2_in_ports_; }

private:
  L2CacheController controller_;
  simdojo::Port *hbm_out_ = nullptr;
  std::vector<simdojo::Port *> l2_in_ports_;
};

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_VM_AMDGPU_L2_CACHE_H_
