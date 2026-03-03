// MIT License
//
// Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "trace_decoder_api.h"

#ifndef ROCPROF_TRACE_DECODER_COMGR_DISABLED

#    include "cxx/code_printing.hpp"

#    include <atomic>
#    include <cstring>
#    include <memory>
#    include <mutex>
#    include <unordered_map>

#    define PUBLIC_API __attribute__((visibility("default")))

namespace
{
using AddressTable = rocprof_trace_decoder::codeobj::CodeobjAddressTranslate;
using Instruction = rocprof_trace_decoder::codeobj::Instruction;

struct DecoderInstance
{
    std::mutex mtx{};
    AddressTable table{};
};

using DecoderMap = std::unordered_map<uint64_t, std::shared_ptr<DecoderInstance>>;

std::mutex& get_map_mutex()
{
    static std::mutex mtx;
    return mtx;
}

DecoderMap& get_map()
{
    static DecoderMap map;
    return map;
}

std::shared_ptr<DecoderInstance> get_instance(rocprof_trace_decoder_handle_t handle)
{
    std::lock_guard<std::mutex> lock(get_map_mutex());
    auto& map = get_map();
    auto it = map.find(handle.handle);
    if (it != map.end()) return it->second;
    return nullptr;
}

// ISA callback implementation that uses the address table
rocprofiler_thread_trace_decoder_status_t comgr_isa_callback(
    char* isa_instruction,
    uint64_t* isa_memory_size,
    uint64_t* isa_size,
    rocprofiler_thread_trace_decoder_pc_t pc,
    void* userdata
)
{
    auto* instance = static_cast<DecoderInstance*>(userdata);
    if (!instance) return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR;

    try
    {
        std::unique_ptr<Instruction> instruction;
        {
            std::lock_guard<std::mutex> lock(instance->mtx);
            instruction = instance->table.get(pc.code_object_id, pc.address);
        }

        if (!instruction) return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_ARGUMENT;

        {
            size_t tmp_isa_size = *isa_size;
            *isa_size = instruction->inst.size();

            if (*isa_size > tmp_isa_size) return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_OUT_OF_RESOURCES;
        }

        std::memcpy(isa_instruction, instruction->inst.data(), *isa_size);
        *isa_memory_size = instruction->size;
    }
    catch (...)
    {
        return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR;
    }
    return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS;
}

struct decode_userdata_t
{
    DecoderInstance* instance{nullptr};
    rocprof_trace_decoder_se_data_callback_t se_cb{nullptr};
    rocprof_trace_decoder_trace_callback_t trace_cb{nullptr};
    void* userdata{nullptr};
};

uint64_t decode_se_callback(uint8_t** buffer, uint64_t* buffer_size, void* userdata)
{
    auto* ctx = static_cast<decode_userdata_t*>(userdata);
    return ctx->se_cb(buffer, buffer_size, ctx->userdata);
}

rocprofiler_thread_trace_decoder_status_t decode_trace_callback(
    rocprofiler_thread_trace_decoder_record_type_t record_type_id,
    void* trace_events,
    uint64_t trace_size,
    void* userdata
)
{
    auto* ctx = static_cast<decode_userdata_t*>(userdata);
    return ctx->trace_cb(record_type_id, trace_events, trace_size, ctx->userdata);
}

} // namespace

extern "C"
{
PUBLIC_API rocprofiler_thread_trace_decoder_status_t
rocprof_trace_decoder_create_handle(rocprof_trace_decoder_handle_t* handle)
{
    if (!handle) return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_ARGUMENT;

    static std::atomic<uint64_t> counter{1};
    handle->handle = counter.fetch_add(1);

    auto instance = std::make_shared<DecoderInstance>();

    std::lock_guard<std::mutex> lock(get_map_mutex());
    get_map()[handle->handle] = std::move(instance);

    return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS;
}

PUBLIC_API rocprofiler_thread_trace_decoder_status_t
rocprof_trace_decoder_destroy_handle(rocprof_trace_decoder_handle_t handle)
{
    std::lock_guard<std::mutex> lock(get_map_mutex());
    auto& map = get_map();
    if (map.erase(handle.handle) == 0) return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_ARGUMENT;
    return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS;
}

PUBLIC_API rocprofiler_thread_trace_decoder_status_t rocprof_trace_decoder_codeobj_load(
    rocprof_trace_decoder_handle_t handle,
    uint64_t load_id,
    uint64_t load_addr,
    uint64_t load_size,
    const void* data,
    uint64_t data_size
)
{
    auto instance = get_instance(handle);
    if (!instance) return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_ARGUMENT;

    try
    {
        std::lock_guard<std::mutex> lock(instance->mtx);
        instance->table.addDecoder(data, data_size, load_id, load_addr, load_size);
    }
    catch (...)
    {
        return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR;
    }
    return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS;
}

PUBLIC_API rocprofiler_thread_trace_decoder_status_t
rocprof_trace_decoder_codeobj_unload(rocprof_trace_decoder_handle_t handle, uint64_t load_id)
{
    auto instance = get_instance(handle);
    if (!instance) return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_ARGUMENT;

    try
    {
        std::lock_guard<std::mutex> lock(instance->mtx);
        bool result = instance->table.removeDecoder(load_id);
        if (result) return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS;
    }
    catch (...)
    {}

    return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR;
}

PUBLIC_API rocprofiler_thread_trace_decoder_status_t rocprof_trace_decoder_decode(
    rocprof_trace_decoder_handle_t handle,
    rocprof_trace_decoder_se_data_callback_t se_data_callback,
    rocprof_trace_decoder_trace_callback_t trace_callback,
    void* userdata
)
{
    auto instance = get_instance(handle);
    if (!instance) return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_ARGUMENT;

    decode_userdata_t ctx{};
    ctx.instance = instance.get();
    ctx.se_cb = se_data_callback;
    ctx.trace_cb = trace_callback;
    ctx.userdata = userdata;

    try
    {
        return rocprof_trace_decoder_parse_data(decode_se_callback, decode_trace_callback, comgr_isa_callback, &ctx);
    }
    catch (...)
    {
        return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_SHADER_DATA;
    }
}

} // extern "C"

#else // ROCPROF_TRACE_DECODER_COMGR_DISABLED — stub implementations

#    define PUBLIC_API __attribute__((visibility("default")))

extern "C"
{
PUBLIC_API rocprofiler_thread_trace_decoder_status_t
rocprof_trace_decoder_create_handle(rocprof_trace_decoder_handle_t*)
{
    return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_NOT_IMPLEMENTED;
}

PUBLIC_API rocprofiler_thread_trace_decoder_status_t rocprof_trace_decoder_destroy_handle(rocprof_trace_decoder_handle_t
)
{
    return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_NOT_IMPLEMENTED;
}

PUBLIC_API rocprofiler_thread_trace_decoder_status_t
rocprof_trace_decoder_codeobj_load(rocprof_trace_decoder_handle_t, uint64_t, uint64_t, uint64_t, const void*, uint64_t)
{
    return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_NOT_IMPLEMENTED;
}

PUBLIC_API rocprofiler_thread_trace_decoder_status_t
rocprof_trace_decoder_codeobj_unload(rocprof_trace_decoder_handle_t, uint64_t)
{
    return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_NOT_IMPLEMENTED;
}

PUBLIC_API rocprofiler_thread_trace_decoder_status_t
rocprof_trace_decoder_decode(rocprof_trace_decoder_handle_t, rocprof_trace_decoder_se_data_callback_t, rocprof_trace_decoder_trace_callback_t, void*)
{
    return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_NOT_IMPLEMENTED;
}

} // extern "C"

#endif // ROCPROF_TRACE_DECODER_COMGR_DISABLED
