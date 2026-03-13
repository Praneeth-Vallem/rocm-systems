// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <rocprofiler-sdk/marker/api_args.h>
#include <rocprofiler-sdk/marker/api_id.h>
#include <rocprofiler-sdk/marker/table_id.h>

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Marker-tracing lifecycle and process state.
 *
 * These declarations provide a minimal ABI surface for distinguishing the
 * original process from forked child processes and for guarding against
 * duplicate marker initialization/finalization in persistent worker children.
 */
typedef enum rocprofiler_marker_lifecycle_state_t
{
    ROCPROFILER_MARKER_LIFECYCLE_STATE_UNINITIALIZED = 0,
    ROCPROFILER_MARKER_LIFECYCLE_STATE_INITIALIZING  = 1,
    ROCPROFILER_MARKER_LIFECYCLE_STATE_INITIALIZED   = 2,
    ROCPROFILER_MARKER_LIFECYCLE_STATE_FINALIZING    = 3,
    ROCPROFILER_MARKER_LIFECYCLE_STATE_FINALIZED     = 4
} rocprofiler_marker_lifecycle_state_t;

typedef struct rocprofiler_marker_process_state_t
{
    /**
     * @brief Process ID that originally initialized marker tracing state.
     */
    uint64_t parent_pid;

    /**
     * @brief Current process ID observed by the marker runtime.
     */
    uint64_t current_pid;

    /**
     * @brief True if the current process is a forked child of the original
     * marker-tracing parent process.
     */
    bool is_forked_child;

    /**
     * @brief True if marker tracing initialization has already been performed
     * in the current process.
     */
    bool init_seen_in_process;

    /**
     * @brief True if marker tracing finalization has already been performed
     * in the current process.
     */
    bool fini_seen_in_process;

    /**
     * @brief Current marker-tracing lifecycle state.
     */
    rocprofiler_marker_lifecycle_state_t lifecycle_state;
} rocprofiler_marker_process_state_t;

#ifdef __cplusplus
}
#endif