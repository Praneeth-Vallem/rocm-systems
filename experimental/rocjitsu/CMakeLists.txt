# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.28)

project(rocm_jit_suite)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

find_package(Threads REQUIRED)

option(ROCJITSU_BUILD_GUI "Build the simulation GUI (requires OpenGL)" ON)

# Third-party dependencies (fetched into third_party/).
set(FETCHCONTENT_BASE_DIR ${PROJECT_SOURCE_DIR}/third_party)
include(FetchContent)

# Google Test
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.2
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# FlatBuffers - serialization for config, checkpoint, and wire format.
FetchContent_Declare(
    flatbuffers
    GIT_REPOSITORY https://github.com/google/flatbuffers.git
    GIT_TAG v24.3.25
)
set(FLATBUFFERS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(FLATBUFFERS_INSTALL OFF CACHE BOOL "" FORCE)
set(FLATBUFFERS_BUILD_FLATLIB ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest flatbuffers)

# Compile FlatBuffers schemas into C++ headers.
set(GENERATED_DIR ${CMAKE_BINARY_DIR}/generated)
file(MAKE_DIRECTORY ${GENERATED_DIR})

set(SCHEMA_FILES
    ${CMAKE_SOURCE_DIR}/schemas/simulation_config.fbs
    ${CMAKE_SOURCE_DIR}/schemas/checkpoint.fbs
)

set(GENERATED_HEADERS)
foreach(schema ${SCHEMA_FILES})
    get_filename_component(schema_name ${schema} NAME_WE)
    set(output_header "${GENERATED_DIR}/${schema_name}_generated.h")
    add_custom_command(
        OUTPUT ${output_header}
        COMMAND $<TARGET_FILE:flatc> --cpp -I ${CMAKE_SOURCE_DIR}/schemas -o ${GENERATED_DIR} ${schema}
        DEPENDS flatc ${schema}
        COMMENT "Compiling FlatBuffers schema: ${schema_name}.fbs"
    )
    list(APPEND GENERATED_HEADERS ${output_header})
endforeach()

add_custom_target(flatbuffers_schemas DEPENDS ${GENERATED_HEADERS})

# GUI dependencies (conditional).
if(ROCJITSU_BUILD_GUI)
    find_package(OpenGL QUIET)
    if(NOT OpenGL_FOUND)
        message(WARNING "OpenGL not found - disabling GUI build")
        set(ROCJITSU_BUILD_GUI OFF)
    endif()
endif()

if(ROCJITSU_BUILD_GUI)
    # GLFW - cross-platform windowing.
    FetchContent_Declare(
        glfw
        GIT_REPOSITORY https://github.com/glfw/glfw.git
        GIT_TAG 3.4
    )
    set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_WAYLAND OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(glfw)

    # Dear ImGui - immediate-mode GUI.
    FetchContent_Declare(
        imgui
        GIT_REPOSITORY https://github.com/ocornut/imgui.git
        GIT_TAG v1.91.8
    )
    FetchContent_MakeAvailable(imgui)

    add_library(imgui_lib STATIC
        ${imgui_SOURCE_DIR}/imgui.cpp
        ${imgui_SOURCE_DIR}/imgui_demo.cpp
        ${imgui_SOURCE_DIR}/imgui_draw.cpp
        ${imgui_SOURCE_DIR}/imgui_tables.cpp
        ${imgui_SOURCE_DIR}/imgui_widgets.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
    )
    target_include_directories(imgui_lib PUBLIC
        ${imgui_SOURCE_DIR}
        ${imgui_SOURCE_DIR}/backends
    )
    target_compile_definitions(imgui_lib PUBLIC IMGUI_DEFINE_MATH_OPERATORS)
    target_link_libraries(imgui_lib PUBLIC glfw OpenGL::GL)

    # imnodes - node editor for topology visualization.
    FetchContent_Declare(
        imnodes
        GIT_REPOSITORY https://github.com/Nelarius/imnodes.git
        GIT_TAG v0.5
    )
    FetchContent_MakeAvailable(imnodes)

    add_library(imnodes_lib STATIC ${imnodes_SOURCE_DIR}/imnodes.cpp)
    target_include_directories(imnodes_lib PUBLIC ${imnodes_SOURCE_DIR})
    target_link_libraries(imnodes_lib PUBLIC imgui_lib)

    # ImPlot - plotting for timeline visualization.
    FetchContent_Declare(
        implot
        GIT_REPOSITORY https://github.com/epezent/implot.git
        GIT_TAG v0.16
    )
    FetchContent_MakeAvailable(implot)

    add_library(implot_lib STATIC
        ${implot_SOURCE_DIR}/implot.cpp
        ${implot_SOURCE_DIR}/implot_items.cpp
    )
    target_include_directories(implot_lib PUBLIC ${implot_SOURCE_DIR})
    target_link_libraries(implot_lib PUBLIC imgui_lib)
endif()

# Libraries
add_subdirectory(lib/util)
add_subdirectory(lib/simdojo)
add_subdirectory(lib/rocjitsu)

# Main shared library - assembles all object libraries.
add_library(rocjitsu_shared SHARED)
set_target_properties(rocjitsu_shared PROPERTIES OUTPUT_NAME rocjitsu)
target_link_libraries(rocjitsu_shared PRIVATE
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
    flatbuffers
    Threads::Threads
)
target_include_directories(rocjitsu_shared PRIVATE ${GENERATED_DIR})
add_dependencies(rocjitsu_shared flatbuffers_schemas)

# Tools
if(ROCJITSU_BUILD_GUI)
    add_subdirectory(tools/simgui)
endif()

# Device kernels (compiled from HIP sources for simulator testing).
add_subdirectory(tests/kernels)

# Tests
enable_testing()

add_executable(rocjitsu_tests
    tests/rvsim_test.cpp
    tests/amdgpu_vm_test.cpp
    tests/config_test.cpp
    tests/matmul_test.cpp
    tests/vector_add_test.cpp
)
target_include_directories(rocjitsu_tests PRIVATE
    ${PROJECT_SOURCE_DIR}/lib/rocjitsu/include
    ${PROJECT_SOURCE_DIR}/lib/rocjitsu/src
    ${GENERATED_DIR})
set(ROCJITSU_OBJECT_LIBS
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
)

target_link_libraries(rocjitsu_tests PRIVATE
    ${ROCJITSU_OBJECT_LIBS} util flatbuffers Threads::Threads GTest::gtest_main)
target_compile_options(rocjitsu_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
target_compile_definitions(rocjitsu_tests PRIVATE
    SCHEMA_DIR="${CMAKE_SOURCE_DIR}/schemas"
    CONFIG_DIR="${CMAKE_SOURCE_DIR}/configs"
)
add_dependencies(rocjitsu_tests flatbuffers_schemas)

if(HAS_DEVICE_KERNELS)
    target_compile_definitions(rocjitsu_tests PRIVATE
        HAS_DEVICE_KERNELS=1
        KERNEL_DIR="${KERNEL_OUTPUT_DIR}"
    )
    add_dependencies(rocjitsu_tests device_kernels)
endif()

include(GoogleTest)
gtest_discover_tests(rocjitsu_tests)
