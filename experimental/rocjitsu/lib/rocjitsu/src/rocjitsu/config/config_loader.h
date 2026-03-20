// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file config_loader.h
/// @brief Configuration loading from JSON via FlatBuffers.

#ifndef ROCJITSU_CONFIG_CONFIG_LOADER_H_
#define ROCJITSU_CONFIG_CONFIG_LOADER_H_

#include "rocjitsu/vm/virtual_machine.h"

#include "simdojo/sim/simulation.h"

#include <memory>
#include <string>

namespace rocjitsu {
namespace config {

/// @brief Result of loading a simulation configuration.
///
/// Contains the engine configuration and the VirtualMachine model. The caller
/// is responsible for wiring the VM into a SimulationEngine topology.
struct LoadedConfig {
  simdojo::SimulationEngine::Config engine_config; ///< Engine parameters.
  std::unique_ptr<VirtualMachine> vm;              ///< The virtual machine model.
};

/// @brief Parse an architecture name string to an rj_code_arch_t enum value.
/// @param arch_str Architecture name (e.g., "cdna3", "rv32i").
/// @returns The corresponding rj_code_arch_t, or ROCJITSU_CODE_ARCH_INVALID if unknown.
rj_code_arch_t parse_arch(const std::string &arch_str);

/// @brief Convert an rj_code_arch_t enum to its string name.
/// @param arch Architecture enum value.
/// @returns Architecture name string, or "invalid" for unknown values.
const char *arch_to_string(rj_code_arch_t arch);

/// @brief Load simulation config from a JSON file using FlatBuffers.
/// @param json_path Path to the JSON config file.
/// @param schema_path Path to the simulation_config.fbs schema file.
/// @returns LoadedConfig with engine parameters and a VirtualMachine, ready to wire.
/// @throws std::runtime_error on file I/O, parse errors, or invalid arch.
LoadedConfig load_config(const std::string &json_path, const std::string &schema_path);

/// @brief Load simulation config from a JSON string using FlatBuffers.
/// @param json JSON configuration string.
/// @param schema_path Path to the simulation_config.fbs schema file.
/// @returns LoadedConfig with engine parameters and a VirtualMachine, ready to wire.
/// @throws std::runtime_error on parse errors or invalid arch.
LoadedConfig load_config_from_string(const std::string &json, const std::string &schema_path);

} // namespace config
} // namespace rocjitsu

#endif // ROCJITSU_CONFIG_CONFIG_LOADER_H_
