// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "app.h"
#include "simulation_panel.h"
#include "topology_editor.h"

#include "rocjitsu/config/checkpoint.h"
#include "rocjitsu/config/config_loader.h"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include <GLFW/glfw3.h>

#include <algorithm>
#include <cstdio>
#include <string>

namespace rocjitsu {
namespace gui {

// File-scope GUI sub-components.
static TopologyEditor s_topo_editor;
static SimulationPanel s_sim_panel;

App::App() = default;

App::~App() {
  if (window_) {
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window_);
    glfwTerminate();
  }
}

void App::reset_simulation() {
  engine_.reset();
  vm_ = nullptr;
  sim_running_ = false;
  sim_tick_ = 0;
  s_sim_panel.clear_history();
}

bool App::init(int width, int height, const char *title) {
  if (!glfwInit()) {
    std::fprintf(stderr, "Failed to initialize GLFW\n");
    return false;
  }

  // OpenGL 3.3 core profile.
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  window_ = glfwCreateWindow(width, height, title, nullptr, nullptr);
  if (!window_) {
    std::fprintf(stderr, "Failed to create GLFW window\n");
    glfwTerminate();
    return false;
  }
  glfwMakeContextCurrent(window_);
  glfwSwapInterval(1); // VSync.

  // Initialize Dear ImGui.
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO &io = ImGui::GetIO();
  (void)io;

  ImGui::StyleColorsDark();

  ImGui_ImplGlfw_InitForOpenGL(window_, true);
  ImGui_ImplOpenGL3_Init("#version 330");

  // Initialize sub-components.
  s_topo_editor.init();

  return true;
}

void App::load_config(const std::string &json_path, const std::string &schema_path) {
  reset_simulation();
  schema_path_ = schema_path;

  try {
    auto loaded = config::load_config(json_path, schema_path);
    // GUI drives stepping from the render thread - force single-threaded.
    loaded.engine_config.num_threads = 1;
    engine_config_ = loaded.engine_config;
    vm_ = loaded.vm.get();
    engine_ = std::make_unique<simdojo::SimulationEngine>(loaded.engine_config);
    engine_->topology().set_root(std::move(loaded.vm));
    engine_->build();
    s_topo_editor.load_from_simulation(vm_);
  } catch (const std::exception &e) {
    std::fprintf(stderr, "Failed to load config: %s\n", e.what());
    vm_ = nullptr;
  }
}

void App::build_default_simulation() {
  reset_simulation();

  // Build JSON config from editor state.
  const auto &cu_cfg = s_topo_editor.cu_config();
  std::string json = R"({"max_ticks":100000,"num_threads":1,"vm":{)"
                     R"("arch":"cdna3","gpu":{"num_xcds":1,"xcd":{"num_shader_engines":)" +
                     std::to_string(s_topo_editor.num_shader_engines()) +
                     R"(,"shader_engine":{"num_compute_units":)" +
                     std::to_string(s_topo_editor.num_cus_per_se()) +
                     R"(,"compute_unit":{"num_wf_slots":)" + std::to_string(cu_cfg.num_wf_slots) +
                     R"(,"sgprs_per_wf":)" + std::to_string(cu_cfg.sgprs_per_wf) +
                     R"(,"vgprs_per_wf":)" + std::to_string(cu_cfg.vgprs_per_wf) +
                     R"(,"lds_size_kb":)" + std::to_string(cu_cfg.lds_size_kb) + R"(}}}}})";

  if (schema_path_.empty()) {
    std::fprintf(stderr, "No schema path set - cannot build simulation\n");
    return;
  }

  try {
    auto loaded = config::load_config_from_string(json, schema_path_);
    engine_config_ = loaded.engine_config;
    vm_ = loaded.vm.get();
    engine_ = std::make_unique<simdojo::SimulationEngine>(loaded.engine_config);
    engine_->topology().set_root(std::move(loaded.vm));
    engine_->build();
  } catch (const std::exception &e) {
    std::fprintf(stderr, "Failed to build simulation from editor config: %s\n", e.what());
    vm_ = nullptr;
  }
}

void App::run() {
  while (!glfwWindowShouldClose(window_)) {
    glfwPollEvents();

    // Auto-step the simulation if running.
    if (sim_running_ && vm_) {
      for (int i = 0; i < steps_per_frame_; ++i) {
        bool any_active = engine_->step();
        sim_tick_++;
        s_sim_panel.record_tick(vm_, sim_tick_);
        if (!any_active) {
          sim_running_ = false;
          break;
        }
      }
    }

    // Begin ImGui frame.
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    render_frame();

    // Render.
    ImGui::Render();
    int display_w, display_h;
    glfwGetFramebufferSize(window_, &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    glfwSwapBuffers(window_);
  }
}

void App::render_frame() {
  render_menu_bar();

  // Topology editor panel.
  ImGui::Begin("Topology Editor");
  s_topo_editor.render();
  ImGui::End();

  // Simulation controls panel.
  ImGui::Begin("Simulation");
  bool step_requested = s_sim_panel.render_controls(sim_running_, sim_tick_, steps_per_frame_);

  // Handle Build button - (re)constructs the simulation from editor state.
  if (ImGui::Button("Build Simulation")) {
    build_default_simulation();
  }

  // Handle single-step request.
  if (step_requested && vm_ && !sim_running_) {
    engine_->step();
    sim_tick_++;
    s_sim_panel.record_tick(vm_, sim_tick_);
  }
  ImGui::End();

  // Wavefront state inspector.
  if (vm_) {
    ImGui::Begin("Wavefront State");
    s_sim_panel.render_wf_state(vm_);
    ImGui::End();

    ImGui::Begin("Timeline");
    s_sim_panel.render_timeline(vm_, sim_tick_);
    ImGui::End();
  }

  // Properties panel for selected component.
  ImGui::Begin("Properties");
  auto &cu_cfg = s_topo_editor.cu_config();
  int wf_slots = static_cast<int>(cu_cfg.num_wf_slots);
  int sgprs = static_cast<int>(cu_cfg.sgprs_per_wf);
  int vgprs = static_cast<int>(cu_cfg.vgprs_per_wf);
  int lds = static_cast<int>(cu_cfg.lds_size_kb);

  ImGui::Text("Compute Unit Configuration");
  ImGui::Separator();
  if (ImGui::SliderInt("WF Slots", &wf_slots, 1, 40))
    cu_cfg.num_wf_slots = static_cast<uint32_t>(std::clamp(wf_slots, 1, 40));
  if (ImGui::SliderInt("SGPRs/WF", &sgprs, 16, 128))
    cu_cfg.sgprs_per_wf = static_cast<uint32_t>(std::clamp(sgprs, 16, 128));
  if (ImGui::SliderInt("VGPRs/WF", &vgprs, 16, 512))
    cu_cfg.vgprs_per_wf = static_cast<uint32_t>(std::clamp(vgprs, 16, 512));
  if (ImGui::SliderInt("LDS (KB)", &lds, 16, 128))
    cu_cfg.lds_size_kb = static_cast<uint32_t>(std::clamp(lds, 16, 128));
  ImGui::End();
}

void App::render_menu_bar() {
  if (ImGui::BeginMainMenuBar()) {
    if (ImGui::BeginMenu("File")) {
      if (ImGui::MenuItem("Load Config...")) {
        // TODO: file dialog integration.
      }
      if (ImGui::MenuItem("Save Checkpoint...")) {
        if (vm_) {
          try {
            config::save_checkpoint("checkpoint.bin", *vm_, sim_tick_, engine_config_);
          } catch (const std::exception &e) {
            std::fprintf(stderr, "Failed to save checkpoint: %s\n", e.what());
          }
        }
      }
      if (ImGui::MenuItem("Restore Checkpoint...")) {
        // TODO: file dialog integration.
      }
      ImGui::Separator();
      if (ImGui::MenuItem("Quit")) {
        glfwSetWindowShouldClose(window_, GLFW_TRUE);
      }
      ImGui::EndMenu();
    }
    ImGui::EndMainMenuBar();
  }
}

} // namespace gui
} // namespace rocjitsu
