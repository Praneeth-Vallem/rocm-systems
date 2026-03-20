# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Define an OBJECT library with standard include dirs and flags for
# rocjitsu sub-components. ROCJITSU_INCLUDE_DIR and ROCJITSU_SRC_DIR
# must be set before including this module.
function(add_object_library name)
    add_library(${name} OBJECT ${ARGN})
    set_target_properties(${name} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_include_directories(${name} PRIVATE
        ${ROCJITSU_INCLUDE_DIR}
        ${ROCJITSU_SRC_DIR})
    target_link_libraries(${name} PRIVATE util simdojo)
    target_compile_options(${name} PRIVATE -Wall -Wextra -Wpedantic -Werror -fvisibility=hidden)
endfunction()
