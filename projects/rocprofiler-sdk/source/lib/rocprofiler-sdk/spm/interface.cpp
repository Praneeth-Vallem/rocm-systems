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

#include "lib/rocprofiler-sdk/spm/interface.hpp"
#include "lib/common/logging.hpp"

#include <fmt/format.h>

#include <dlfcn.h>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

namespace rocprofiler
{
namespace spm
{

std::optional<spm_interface>
construct_spm_interface(void* handle)
{
   
    if(!handle) handle = dlopen("libhsa-amd-aqlprofile64.so.1", RTLD_NOLOAD | RTLD_LAZY);

    if(!handle)
    {
        ROCP_CI_LOG(WARNING) << fmt::format("aqlprofile cannot be opened");
        return  std::nullopt;
    }
    auto interface = spm_interface(); 
    interface.spm_create_packets = (spm_interface::spm_create_packets_fn_t*) dlsym(handle, "aqlprofile_spm_create_packets");
    interface.spm_delete_packets = (spm_interface::spm_delete_packets_fn_t*) dlsym(handle, "aqlprofile_spm_delete_packets");
    interface.spm_start      = (spm_interface::spm_start_fn_t*) dlsym(handle, "aqlprofile_spm_start");
    interface.spm_stop      = (spm_interface::spm_stop_fn_t*) dlsym(handle, "aqlprofile_spm_stop");
    interface.spm_decode_stream_v1     = (spm_interface::spm_decode_stream_v1_fn_t*) dlsym(handle, "aqlprofile_spm_decode_stream_v1");
    interface.spm_decode_query     = (spm_interface::spm_decode_query_fn_t*) dlsym(handle, "aqlprofile_spm_decode_query");
    interface.spm_is_event_supported  = (spm_interface::spm_is_event_supported_fn_t*) dlsym(handle, "aqlprofile_spm_is_event_supported");
    return interface;
}

}  // namespace spm
}  // namespace rocprofiler
