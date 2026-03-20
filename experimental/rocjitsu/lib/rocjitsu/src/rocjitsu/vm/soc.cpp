// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/soc.h"

#include "simdojo/sim/simulation.h"
#include "simdojo/sim/topology.h"

#include <memory>
#include <string>

namespace rocjitsu {

SoC::SoC(std::string name, const Config &config)
    : simdojo::CompositeComponent(std::move(name)), exec_mode_(config.exec_mode) {
  auto soc_name = this->name();

  // GPU memory is shared across all XCDs.
  auto mem = std::make_unique<amdgpu::GpuMemory>("vram");
  memory_ = mem.get();
  add_child(std::move(mem));

  if (config.num_iods > 0) {
    // Create IODs, each with its own memory-side cache and HBM controller.
    // HBM stacks are split evenly across IODs.
    uint32_t xcds_per_iod = config.num_xcds / config.num_iods;
    for (uint32_t j = 0; j < config.num_iods; ++j) {
      amdgpu::Iod::Config iod_config{};
      iod_config.num_hbm_stacks = 4; // Structural only in functional mode.
      auto iod_ptr =
          std::make_unique<amdgpu::Iod>(soc_name + ".iod" + std::to_string(j), iod_config, memory_);
      iods_.push_back(iod_ptr.get());
      add_child(std::move(iod_ptr));
    }

    // Create XCDs, each assigned to its parent IOD.
    // L2 backing = IOD's memory-side cache controller.
    for (uint32_t i = 0; i < config.num_xcds; ++i) {
      uint32_t iod_idx = i / xcds_per_iod;
      if (iod_idx >= config.num_iods)
        iod_idx = config.num_iods - 1; // Handle uneven division.
      auto *l2_backing = iods_[iod_idx]->msc_controller();
      auto xcd_ptr =
          std::make_unique<amdgpu::Xcd>(soc_name + ".xcd" + std::to_string(i), config.xcd,
                                        config.arch, memory_, l2_backing, config.exec_mode);
      xcds_.push_back(xcd_ptr.get());
      add_child(std::move(xcd_ptr));
    }
  } else {
    // No IOD modeling: XCDs connect directly to a standalone HBM controller.
    hbm_standalone_ = std::make_unique<amdgpu::HbmController>(memory_);
    for (uint32_t i = 0; i < config.num_xcds; ++i) {
      auto xcd_ptr = std::make_unique<amdgpu::Xcd>(soc_name + ".xcd" + std::to_string(i),
                                                   config.xcd, config.arch, memory_,
                                                   hbm_standalone_.get(), config.exec_mode);
      xcds_.push_back(xcd_ptr.get());
      add_child(std::move(xcd_ptr));
    }
  }
}

void SoC::flush_all() {
  // Flush all per-CU L1 caches (invalidate, since L1 is write-through).
  for (auto *x : xcds_) {
    for (uint32_t si = 0; si < x->num_shader_engines(); ++si) {
      auto *se = x->shader_engine(si);
      for (uint32_t ci = 0; ci < se->num_compute_units(); ++ci)
        se->compute_unit(ci)->flush_l1();
    }
  }

  // Flush each XCD's L2 once (L2 is shared across all CUs in an XCD).
  for (auto *x : xcds_)
    x->l2_cache()->controller()->flush_all();

  // Flush all IOD memory-side caches (MSC → HBM).
  for (auto *i : iods_)
    i->msc_controller()->flush_all();
}

void SoC::initialize() {
  if (iods_.empty())
    return;

  auto &topo = engine()->topology();
  uint32_t xcds_per_iod = static_cast<uint32_t>(xcds_.size()) / static_cast<uint32_t>(iods_.size());

  // Wire each XCD's L2 hbm_out port → its parent IOD's xcd_in port.
  for (uint32_t i = 0; i < xcds_.size(); ++i) {
    uint32_t iod_idx = i / xcds_per_iod;
    if (iod_idx >= iods_.size())
      iod_idx = static_cast<uint32_t>(iods_.size()) - 1;
    auto *xcd_out = xcds_[i]->l2_cache()->hbm_out_port();
    auto *iod_in = iods_[iod_idx]->create_xcd_in_port(xcds_[i]->name());
    auto *link = topo.add_link(xcd_out, iod_in, /*latency=*/1);
    link->set_exec_mode(exec_mode_);
  }

  // Wire IOD peer interconnect links (bidirectional between all IOD pairs).
  for (uint32_t a = 0; a < iods_.size(); ++a) {
    for (uint32_t b = a + 1; b < iods_.size(); ++b) {
      auto *link_ab =
          topo.add_link(iods_[a]->peer_out_port(), iods_[b]->peer_in_port(), /*latency=*/1);
      link_ab->set_exec_mode(exec_mode_);
      auto *link_ba =
          topo.add_link(iods_[b]->peer_out_port(), iods_[a]->peer_in_port(), /*latency=*/1);
      link_ba->set_exec_mode(exec_mode_);
    }
  }
}

} // namespace rocjitsu
