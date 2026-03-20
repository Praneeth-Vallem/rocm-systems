// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/xcd.h"

#include "simdojo/sim/simulation.h"
#include "simdojo/sim/topology.h"

#include <cassert>
#include <memory>
#include <string>

namespace rocjitsu {
namespace amdgpu {

Xcd::Xcd(std::string name, const Config &config, rj_code_arch_t arch, GpuMemory *memory,
         simdojo::MemoryInterface *l2_backing, simdojo::ExecMode exec_mode)
    : simdojo::CompositeComponent(std::move(name)), exec_mode_(exec_mode) {
  auto xcd_name = this->name();

  // Create shared L2 cache for this XCD, backed by the provided MemoryInterface
  // (HbmController directly, or IOD's MemorySideCacheController).
  auto l2 = std::make_unique<L2Cache>(xcd_name + ".l2", l2_backing);
  l2_cache_ = l2.get();
  add_child(std::move(l2));

  // Create command processor for this XCD.
  auto cp = std::make_unique<CommandProcessor>(xcd_name + ".cp");
  cp_ = cp.get();

  // Create shader engines, each with its own CU array sharing the XCD's L2.
  for (uint32_t i = 0; i < config.num_shader_engines; ++i) {
    auto se =
        std::make_unique<ShaderEngine>(xcd_name + ".se" + std::to_string(i), config.shader_engine,
                                       arch, memory, l2_cache_->controller(), exec_mode);
    // Register all CUs in this SE with the XCD's command processor.
    // This also sets up on_idle callbacks from CUs to CP::check_all_idle().
    for (uint32_t c = 0; c < se->num_compute_units(); ++c)
      cp_->add_compute_unit(se->compute_unit(c));
    shader_engines_.push_back(se.get());
    add_child(std::move(se));
  }

  add_child(std::move(cp));
}

void Xcd::initialize() {
  auto &topo = engine()->topology();

  // Wire CP dispatch ports → CU dispatch_in ports.
  auto &dispatch_ports = cp_->dispatch_ports();
  auto &cus = cp_->compute_units();
  assert(dispatch_ports.size() == cus.size() && "dispatch ports and CU count must match");
  for (size_t i = 0; i < cus.size(); ++i) {
    auto *link = topo.add_link(dispatch_ports[i], cus[i]->dispatch_in_port(), /*latency=*/1);
    link->set_exec_mode(exec_mode_);
  }

  // Wire each CU's L2 request port → a dedicated L2 IN port (one per CU).
  for (size_t i = 0; i < cus.size(); ++i) {
    auto *l2_in = l2_cache_->create_cu_in_port(cus[i]->name());
    auto *link = topo.add_link(cus[i]->l2_req_port(), l2_in, /*latency=*/1);
    link->set_exec_mode(exec_mode_);
  }

  // L2→HBM/fabric port wiring is done by SoC::initialize() since the
  // destination depends on the IOD topology.
}

} // namespace amdgpu
} // namespace rocjitsu
