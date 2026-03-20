// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_VM_AMDGPU_HBM_CONTROLLER_H_
#define ROCJITSU_VM_AMDGPU_HBM_CONTROLLER_H_

#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "simdojo/components/memory_interface.h"

#include <cstdint>
#include <cstring>

namespace rocjitsu {
namespace amdgpu {

/// @brief High Bandwidth Memory (HBM) controller — the lowest level of the memory hierarchy.
///
/// @details Wraps GpuMemory (SparseMemory) and provides the MemoryInterface that L2
/// uses for cache line fills and write-backs. In a timing model this would
/// model HBM channel latency, bandwidth, and bank conflicts. The current
/// functional implementation is synchronous and immediate.
class HbmController : public simdojo::MemoryInterface {
public:
  explicit HbmController(GpuMemory *memory) : memory_(memory) {}

  void read(uint64_t addr, uint8_t *dst, uint32_t size) override {
    for (uint32_t i = 0; i < size; ++i)
      dst[i] = memory_->read8(addr + i);
  }

  void write(uint64_t addr, const uint8_t *src, uint32_t size) override {
    for (uint32_t i = 0; i < size; ++i)
      memory_->write8(addr + i, src[i]);
  }

  /// @brief Read a 32-bit dword (little-endian).
  uint32_t read32(uint64_t addr) const { return memory_->read32(addr); }

  /// @brief Write a 32-bit dword (little-endian).
  void write32(uint64_t addr, uint32_t val) { memory_->write32(addr, val); }

  /// @brief Direct access to the underlying GpuMemory.
  GpuMemory *memory() const { return memory_; }

private:
  GpuMemory *memory_;
};

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_VM_AMDGPU_HBM_CONTROLLER_H_
