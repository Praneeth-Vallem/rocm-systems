// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/config/checkpoint.h"
#include "rocjitsu/config/config_loader.h"
#include "rocjitsu/vm/rj_vm.h"
#include "rocjitsu/vm/virtual_machine.h"

#include "simdojo/sim/simulation.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <string>

namespace {

const std::string kSchemaDir = SCHEMA_DIR;
const std::string kConfigDir = CONFIG_DIR;
const std::string kSchemaPath = kSchemaDir + "/simulation_config.fbs";

using namespace rocjitsu;

TEST(ConfigLoaderTest, LoadCdna4Config) {
  std::string json = kConfigDir + "/amdgpu_cdna4.json";
  auto loaded = config::load_config(json, kSchemaPath);
  auto *vm = loaded.vm.get();

  // CDNA4 config: 8 XCDs, 4 SEs per XCD, 8 CUs per SE, 2 IODs.
  EXPECT_EQ(vm->soc()->num_xcds(), 8u);
  EXPECT_EQ(vm->soc()->num_iods(), 2u);
  auto *xcd = vm->soc()->xcd(0);
  EXPECT_EQ(xcd->num_shader_engines(), 4u);
  EXPECT_EQ(xcd->shader_engine(0)->num_compute_units(), 8u);
}

TEST(ConfigLoaderTest, BuildFromJsonString) {
  const char *json = R"({
    "max_ticks": 5000,
    "num_threads": 1,
    "vm": {
      "arch": "cdna3",
      "gpu": {
        "num_xcds": 1,
        "xcd": {
          "num_shader_engines": 2,
          "shader_engine": {
            "num_compute_units": 3,
            "compute_unit": {
              "num_wf_slots": 20,
              "sgprs_per_wf": 104,
              "vgprs_per_wf": 256,
              "lds_size_kb": 64
            }
          }
        }
      }
    }
  })";

  auto loaded = config::load_config_from_string(json, kSchemaPath);
  auto *vm = loaded.vm.get();

  // 1 XCD, 2 SEs, each with 3 CUs.
  auto *xcd = vm->soc()->xcd(0);
  EXPECT_EQ(xcd->num_shader_engines(), 2u);
  EXPECT_EQ(xcd->shader_engine(0)->num_compute_units(), 3u);
  EXPECT_EQ(xcd->shader_engine(1)->num_compute_units(), 3u);
}

TEST(ConfigLoaderTest, DispatchDistributesAcrossCUs) {
  const char *json = R"({"max_ticks":10000,"num_threads":1,
    "vm":{"arch":"cdna3","gpu":{"num_xcds":1,"xcd":{"num_shader_engines":1,"shader_engine":{"num_compute_units":2,
    "compute_unit":{"num_wf_slots":10,"sgprs_per_wf":104,"vgprs_per_wf":256,"lds_size_kb":64}}}}}})";

  auto loaded = config::load_config_from_string(json, kSchemaPath);
  auto *vm = loaded.vm.get();

  simdojo::SimulationEngine engine(loaded.engine_config);
  engine.topology().set_root(std::move(loaded.vm));
  engine.build();

  // Load an invalid instruction at 0x100 so wavefronts halt immediately.
  vm->memory()->write32(0x100, 0xFFFFFFFF);

  auto *xcd = vm->soc()->xcd(0);
  amdgpu::DispatchPacket pkt;
  pkt.kernel_entry_pc = 0x100;
  pkt.workgroup_count = 2;
  pkt.wfs_per_workgroup = 1;
  pkt.sgprs_per_wf = 104;
  pkt.vgprs_per_wf = 256;
  xcd->command_processor()->enqueue(pkt);

  engine.step();

  // After event-driven execution, wavefronts have halted (one per CU).
  // Verify round-robin distribution: 2 workgroups → 1 wavefront per CU.
  EXPECT_EQ(xcd->command_processor()->dispatched_count(), 1u);
  auto *se = vm->soc()->xcd(0)->shader_engine(0);
  EXPECT_EQ(se->compute_unit(0)->num_wfs(), 1u);
  EXPECT_EQ(se->compute_unit(1)->num_wfs(), 1u);
}

TEST(CheckpointTest, SaveAndRestoreMemory) {
  const char *json = R"({"max_ticks":10000,"num_threads":1,
    "vm":{"arch":"cdna3","gpu":{"num_xcds":1,"xcd":{"num_shader_engines":1,"shader_engine":{"num_compute_units":1,
    "compute_unit":{"num_wf_slots":10,"sgprs_per_wf":104,"vgprs_per_wf":256,"lds_size_kb":64}}}}}})";

  auto loaded = config::load_config_from_string(json, kSchemaPath);
  auto *vm = loaded.vm.get();

  vm->memory()->write32(0x1000, 0xDEADBEEF);
  vm->memory()->write64(0x2000, 0x0123456789ABCDEFULL);

  const char *path = "/tmp/rocjitsu_test_checkpoint.bin";
  config::save_checkpoint(path, *vm, 42, loaded.engine_config);
  ASSERT_TRUE(std::filesystem::exists(path));

  auto restored = config::restore_checkpoint(path);
  EXPECT_EQ(restored.vm->memory()->read32(0x1000), 0xDEADBEEFu);
  EXPECT_EQ(restored.vm->memory()->read64(0x2000), 0x0123456789ABCDEFULL);

  std::filesystem::remove(path);
}

TEST(CApiTest, CreateAndDestroyFromString) {
  const char *json = R"({"max_ticks":10000,"num_threads":1,
    "vm":{"arch":"cdna3","gpu":{"num_xcds":1,"xcd":{"num_shader_engines":1,"shader_engine":{"num_compute_units":1,
    "compute_unit":{"num_wf_slots":10,"sgprs_per_wf":104,"vgprs_per_wf":256,"lds_size_kb":64}}}}}})";
  rj_vm_t *handle = nullptr;
  EXPECT_EQ(rj_vm_create_from_string(json, kSchemaPath.c_str(), &handle), ROCJITSU_STATUS_SUCCESS);
  ASSERT_NE(handle, nullptr);
  rj_vm_destroy(handle);
}

TEST(CApiTest, InvalidArguments) {
  rj_vm_t *handle = nullptr;
  EXPECT_EQ(rj_vm_create_from_string(nullptr, kSchemaPath.c_str(), &handle),
            ROCJITSU_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(rj_vm_step(nullptr, nullptr), ROCJITSU_STATUS_INVALID_ARGUMENT);
}

} // namespace
