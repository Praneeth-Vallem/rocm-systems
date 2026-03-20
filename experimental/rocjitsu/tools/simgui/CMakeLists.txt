# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

add_executable(simgui
    main.cpp
    app.cpp
    topology_editor.cpp
    simulation_panel.cpp
)

target_include_directories(simgui PRIVATE
    ${PROJECT_SOURCE_DIR}/lib/rocjitsu/include
    ${PROJECT_SOURCE_DIR}/lib/rocjitsu/src
    ${GENERATED_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_options(simgui PRIVATE -Wall -Wextra -Wpedantic -Werror)

target_link_libraries(simgui PRIVATE
    rocjitsu_code
    rocjitsu_isa
    rocjitsu_isa_cdna3
    rocjitsu_isa_cdna4
    rocjitsu_isa_risc_v
    simdojo
    rocjitsu_vm
    rocjitsu_vm_amdgpu
    rocjitsu_vm_risc_v
    rocjitsu_config
    util
    flatbuffers
    Threads::Threads
    imgui_lib
    imnodes_lib
    implot_lib
)

add_dependencies(simgui flatbuffers_schemas)
