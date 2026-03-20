// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/config/config_loader.h"

#include "flatbuffers/idl.h"
#include "simdojo/sim/exec_mode.h"
#include "simulation_config_generated.h"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace rocjitsu {
namespace config {

namespace {

/// @brief Read an entire file into a string.
std::string read_file(const std::string &path) {
  std::ifstream f(path);
  if (!f.is_open())
    throw std::runtime_error("Cannot open file: " + path);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

} // namespace

rj_code_arch_t parse_arch(const std::string &arch_str) {
  if (arch_str == "cdna1")
    return ROCJITSU_CODE_ARCH_CDNA1;
  if (arch_str == "cdna2")
    return ROCJITSU_CODE_ARCH_CDNA2;
  if (arch_str == "cdna3")
    return ROCJITSU_CODE_ARCH_CDNA3;
  if (arch_str == "cdna4")
    return ROCJITSU_CODE_ARCH_CDNA4;
  if (arch_str == "rv32i")
    return ROCJITSU_CODE_ARCH_RV32I;
  if (arch_str == "rv64i")
    return ROCJITSU_CODE_ARCH_RV64I;
  return ROCJITSU_CODE_ARCH_INVALID;
}

const char *arch_to_string(rj_code_arch_t arch) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA1:
    return "cdna1";
  case ROCJITSU_CODE_ARCH_CDNA2:
    return "cdna2";
  case ROCJITSU_CODE_ARCH_CDNA3:
    return "cdna3";
  case ROCJITSU_CODE_ARCH_CDNA4:
    return "cdna4";
  case ROCJITSU_CODE_ARCH_RV32I:
    return "rv32i";
  case ROCJITSU_CODE_ARCH_RV64I:
    return "rv64i";
  default:
    return "invalid";
  }
}

namespace {

/// @brief Build engine config from top-level SimulationConfig fields.
simdojo::SimulationEngine::Config
engine_config_from_fb(const rocjitsu::fb::SimulationConfig *fb_config) {
  simdojo::SimulationEngine::Config cfg{};
  cfg.max_ticks = fb_config->max_ticks();
  cfg.num_threads = fb_config->num_threads();
  cfg.verbose = fb_config->verbose();
  return cfg;
}

/// @brief Build a VirtualMachine::Config from the vm section.
VirtualMachine::Config vm_config_from_fb(const rocjitsu::fb::SimulationConfig *fb_config) {
  VirtualMachine::Config vm_config{};
  vm_config.soc.arch = ROCJITSU_CODE_ARCH_INVALID;

  auto *vm = fb_config->vm();
  if (!vm)
    throw std::runtime_error("Missing 'vm' section in configuration");

  if (vm->arch())
    vm_config.soc.arch = parse_arch(vm->arch()->str());
  if (vm_config.soc.arch == ROCJITSU_CODE_ARCH_INVALID)
    throw std::runtime_error("Missing or invalid 'arch' in configuration");

  if (auto *gpu = vm->gpu()) {
    vm_config.soc.num_xcds = gpu->num_xcds();
    vm_config.soc.num_iods = gpu->num_iods();
    if (auto *xcd = gpu->xcd()) {
      vm_config.soc.xcd.num_shader_engines = xcd->num_shader_engines();
      if (auto *se = xcd->shader_engine()) {
        vm_config.soc.xcd.shader_engine.num_compute_units = se->num_compute_units();
        if (auto *cu = se->compute_unit()) {
          auto &cu_cfg = vm_config.soc.xcd.shader_engine.compute_unit;
          cu_cfg.num_wf_slots = cu->num_wf_slots();
          cu_cfg.sgprs_per_wf = cu->sgprs_per_wf();
          cu_cfg.vgprs_per_wf = cu->vgprs_per_wf();
          cu_cfg.lds_size_kb = cu->lds_size_kb();
        }
      }
    }
  }

  if (vm_config.soc.num_xcds == 0)
    throw std::runtime_error("'num_xcds' must be > 0");
  if (vm_config.soc.xcd.num_shader_engines == 0)
    throw std::runtime_error("'num_shader_engines' must be > 0");
  if (vm_config.soc.xcd.shader_engine.num_compute_units == 0)
    throw std::runtime_error("'num_compute_units' must be > 0");

  return vm_config;
}

/// @brief Parse exec_mode string to ExecMode enum.
simdojo::ExecMode parse_exec_mode(const rocjitsu::fb::SimulationConfig *fb_config) {
  if (fb_config->exec_mode()) {
    std::string mode_str = fb_config->exec_mode()->str();
    if (mode_str == "clocked")
      return simdojo::ExecMode::CLOCKED;
  }
  return simdojo::ExecMode::FUNCTIONAL;
}

/// @brief Build a LoadedConfig from a parsed FlatBuffer config.
LoadedConfig build_from_fb(const rocjitsu::fb::SimulationConfig *fb_config) {
  auto engine_config = engine_config_from_fb(fb_config);
  auto vm_config = vm_config_from_fb(fb_config);
  vm_config.soc.exec_mode = parse_exec_mode(fb_config);

  auto vm = std::make_unique<VirtualMachine>(vm_config);

  // Load programs and enqueue dispatch packets from the vm config.
  auto *vm_fb = fb_config->vm();
  if (vm_fb && vm_fb->programs()) {
    for (auto *prog : *vm_fb->programs()) {
      // Load binary into memory.
      if (prog->binary_path()) {
        std::string bin_path = prog->binary_path()->str();
        std::ifstream bin(bin_path, std::ios::binary | std::ios::ate);
        if (!bin.is_open())
          throw std::runtime_error("Cannot open binary: " + bin_path);
        auto pos = bin.tellg();
        if (pos < 0)
          throw std::runtime_error("Cannot determine size of binary: " + bin_path);
        auto size = static_cast<size_t>(pos);
        bin.seekg(0, std::ios::beg);
        std::vector<uint8_t> data(size);
        if (!bin.read(reinterpret_cast<char *>(data.data()), static_cast<std::streamsize>(size)))
          throw std::runtime_error("Failed to read binary: " + bin_path);
        vm->memory()->load_image(data.data(), data.size(), prog->base_addr());
      }

      // Enqueue dispatch packets on the first XCD's command processor.
      if (prog->dispatches() && !vm->soc()->xcds().empty()) {
        auto *cp = vm->soc()->xcds()[0]->command_processor();
        for (auto *d : *prog->dispatches()) {
          amdgpu::DispatchPacket pkt;
          pkt.kernel_entry_pc = d->kernel_entry_pc();
          pkt.workgroup_count = d->workgroup_count();
          pkt.wfs_per_workgroup = d->wfs_per_workgroup();
          pkt.sgprs_per_wf = d->sgprs_per_wf();
          pkt.vgprs_per_wf = d->vgprs_per_wf();
          cp->enqueue(pkt);
        }
      }
    }
  }

  return {engine_config, std::move(vm)};
}

/// @brief Parse a JSON string against the schema and return a FlatBuffer config.
const rocjitsu::fb::SimulationConfig *
parse_json(const std::string &json, const std::string &schema_path, flatbuffers::Parser &parser) {
  std::string schema_text = read_file(schema_path);
  parser.opts.skip_unexpected_fields_in_json = true;
  if (!parser.Parse(schema_text.c_str()))
    throw std::runtime_error("Failed to parse schema: " + std::string(parser.error_));
  if (!parser.Parse(json.c_str()))
    throw std::runtime_error("Failed to parse JSON config: " + std::string(parser.error_));
  return flatbuffers::GetRoot<rocjitsu::fb::SimulationConfig>(parser.builder_.GetBufferPointer());
}

} // namespace

LoadedConfig load_config(const std::string &json_path, const std::string &schema_path) {
  std::string json_text = read_file(json_path);
  flatbuffers::Parser parser;
  auto *fb_config = parse_json(json_text, schema_path, parser);
  return build_from_fb(fb_config);
}

LoadedConfig load_config_from_string(const std::string &json, const std::string &schema_path) {
  flatbuffers::Parser parser;
  auto *fb_config = parse_json(json, schema_path, parser);
  return build_from_fb(fb_config);
}

} // namespace config
} // namespace rocjitsu
