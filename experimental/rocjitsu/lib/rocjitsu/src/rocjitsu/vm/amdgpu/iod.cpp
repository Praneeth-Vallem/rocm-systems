// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/iod.h"

#include <memory>
#include <string>

namespace rocjitsu {
namespace amdgpu {

Iod::Iod(std::string name, const Config &config, GpuMemory *memory)
    : simdojo::CompositeComponent(std::move(name)) {
  auto iod_name = this->name();

  // HbmController is a non-Component controller owned by the IOD directly.
  // We allocate it on the heap so its lifetime matches the IOD.
  auto hbm = std::make_unique<HbmController>(memory);
  hbm_ = hbm.get();

  // The MemorySideCache is backed by the HbmController.
  auto msc = std::make_unique<MemorySideCache>(iod_name + ".msc", hbm_);
  msc_ = msc.get();
  add_child(std::move(msc));

  // Peer interconnect ports (structural, for cross-IOD topology).
  peer_out_ = add_port(std::make_unique<simdojo::Port>(
      "peer_out", 0, this, simdojo::PortDirection::OUT, simdojo::PortProtocol::MEMORY_REQ));
  peer_in_ = add_port(std::make_unique<simdojo::Port>(
      "peer_in", 1, this, simdojo::PortDirection::IN, simdojo::PortProtocol::MEMORY_REQ));

  // HBM output ports — one per HBM stack (structural).
  for (uint32_t i = 0; i < config.num_hbm_stacks; ++i) {
    auto port_id = static_cast<simdojo::PortID>(2 + i);
    auto port = std::make_unique<simdojo::Port>("hbm_out_" + std::to_string(i), port_id, this,
                                                simdojo::PortDirection::OUT,
                                                simdojo::PortProtocol::MEMORY_REQ);
    hbm_out_ports_.push_back(add_port(std::move(port)));
  }

  // Transfer HbmController ownership to a member so it outlives children.
  hbm_owned_ = std::move(hbm);
}

simdojo::Port *Iod::create_xcd_in_port(const std::string &xcd_name) {
  auto port_id = static_cast<simdojo::PortID>(2 + hbm_out_ports_.size() + xcd_in_ports_.size());
  auto port = std::make_unique<simdojo::Port>("xcd_in_" + xcd_name, port_id, this,
                                              simdojo::PortDirection::IN,
                                              simdojo::PortProtocol::MEMORY_REQ);
  auto *raw = add_port(std::move(port));
  xcd_in_ports_.push_back(raw);
  return raw;
}

void Iod::initialize() {
  // No internal port wiring needed in functional mode, L2 calls the
  // MSC controller directly via MemoryInterface. Port wiring between
  // XCD→IOD and IOD→GpuMemory is done by SoC::initialize().
}

} // namespace amdgpu
} // namespace rocjitsu
