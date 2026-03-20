// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_VM_AMDGPU_IOD_H_
#define ROCJITSU_VM_AMDGPU_IOD_H_

#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/hbm_controller.h"
#include "rocjitsu/vm/amdgpu/memory_side_cache.h"

#include "simdojo/sim/component.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace rocjitsu {
namespace amdgpu {

/// @brief I/O Die — contains memory-side cache, memory controllers, and fabric routing.
///
/// @details Each IOD serves a subset of XCDs and owns a portion of the memory-side cache
/// and HBM stacks. The number of IODs and their XCD assignments are driven by config.
class Iod : public simdojo::CompositeComponent {
public:
  struct Config {
    uint32_t num_hbm_stacks; ///< Number of HBM stacks on this IOD.
  };

  Iod(std::string name, const Config &config, GpuMemory *memory);

  void initialize() override;

  /// @brief Create an IN port for an XCD's L2 hbm_out connection.
  simdojo::Port *create_xcd_in_port(const std::string &xcd_name);

  /// @brief Peer interconnect OUT port (for cross-IOD traffic).
  simdojo::Port *peer_out_port() { return peer_out_; }

  /// @brief Peer interconnect IN port (for cross-IOD traffic).
  simdojo::Port *peer_in_port() { return peer_in_; }

  /// @brief Return the memory-side cache controller (for MemoryInterface wiring).
  MemorySideCacheController *msc_controller() { return msc_->controller(); }

  /// @brief Return the HBM controller for this IOD.
  HbmController *hbm_controller() { return hbm_; }

  /// @brief Return the HBM output ports (structural, for topology visibility).
  const std::vector<simdojo::Port *> &hbm_out_ports() const { return hbm_out_ports_; }

private:
  std::unique_ptr<HbmController> hbm_owned_; ///< Owned HbmController.
  MemorySideCache *msc_ = nullptr;
  HbmController *hbm_ = nullptr;
  std::vector<simdojo::Port *> xcd_in_ports_;
  simdojo::Port *peer_out_ = nullptr;
  simdojo::Port *peer_in_ = nullptr;
  std::vector<simdojo::Port *> hbm_out_ports_;
};

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_VM_AMDGPU_IOD_H_
