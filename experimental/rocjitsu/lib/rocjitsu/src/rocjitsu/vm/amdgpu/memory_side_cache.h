// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_VM_AMDGPU_MEMORY_SIDE_CACHE_H_
#define ROCJITSU_VM_AMDGPU_MEMORY_SIDE_CACHE_H_

#include "simdojo/components/cache.h"
#include "simdojo/components/memory_interface.h"
#include "simdojo/sim/component.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

namespace rocjitsu {
namespace amdgpu {

/// @brief Memory-side cache controller sitting between L2 and HBM on each IOD.
///
/// @details Write-back, write-allocate cache, and no mtype awareness. All traffic from
/// L2 is already filtered (UC bypasses L2, so it never reaches the MSC).
/// On miss, fetches from the backing MemoryInterface (typically HbmController).
/// On eviction, dirty lines are written back to the backing store.
class MemorySideCacheController : public simdojo::MemoryInterface {
public:
  static constexpr uint32_t LINE_SIZE_BITS = 7; // 128 bytes
  static constexpr uint32_t NUM_SETS = 65536;   // 65536 sets × 16-way × 128B = 128MB
  static constexpr uint32_t ASSOCIATIVITY = 16;

  using CacheStore = simdojo::Cache<LINE_SIZE_BITS, NUM_SETS, ASSOCIATIVITY>;
  static constexpr uint32_t LINE_SIZE = CacheStore::LINE_SIZE;

  explicit MemorySideCacheController(simdojo::MemoryInterface *backing) : backing_(backing) {}

  void read(uint64_t addr, uint8_t *dst, uint32_t size) override;
  void write(uint64_t addr, const uint8_t *src, uint32_t size) override;

  /// @brief Flush all dirty lines to the backing store and invalidate.
  void flush_all();

private:
  void ensure_line(uint64_t addr);

  CacheStore cache_;
  simdojo::MemoryInterface *backing_;
};

/// @brief Memory-side cache as a simdojo Component, one per IOD.
///
/// @details Owns the MemorySideCacheController. Provides structural ports for the
/// topology graph. In functional mode, the parent IOD wires L2's backing
/// MemoryInterface to the controller directly; ports carry no data.
class MemorySideCache : public simdojo::Component {
public:
  MemorySideCache(std::string name, simdojo::MemoryInterface *backing)
      : simdojo::Component(std::move(name)), controller_(backing) {
    hbm_out_ = add_port(std::make_unique<simdojo::Port>(
        "hbm_out", 0, this, simdojo::PortDirection::OUT, simdojo::PortProtocol::MEMORY_REQ));
  }

  MemorySideCacheController *controller() { return &controller_; }

  simdojo::Port *hbm_out_port() { return hbm_out_; }

  simdojo::Port *create_fabric_in_port(const std::string &src_name) {
    auto port_id = static_cast<simdojo::PortID>(fabric_in_ports_.size() + 1);
    auto port = std::make_unique<simdojo::Port>("fabric_in_" + src_name, port_id, this,
                                                simdojo::PortDirection::IN,
                                                simdojo::PortProtocol::MEMORY_REQ);
    auto *raw = add_port(std::move(port));
    fabric_in_ports_.push_back(raw);
    return raw;
  }

private:
  MemorySideCacheController controller_;
  simdojo::Port *hbm_out_ = nullptr;
  std::vector<simdojo::Port *> fabric_in_ports_;
};

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_VM_AMDGPU_MEMORY_SIDE_CACHE_H_
