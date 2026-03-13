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

#ifdef buffered_api_tracing_client_EXPORTS
#    define CLIENT_API __attribute__((visibility("default")))
#else
#    define CLIENT_API
#endif

#include <cstdint>
#include <set>
#include <utility>
#include <vector>

namespace client
{
struct tracing_category_config
{
    uint32_t category       = 0;
    uint32_t operation_count = 0;
};

struct buffered_tracing_config
{
    std::set<uint32_t> enabled_categories = {};
    std::vector<tracing_category_config> category_operations = {};

    void
    enable_category(uint32_t category, uint32_t operation_count = 0)
    {
        enabled_categories.insert(category);
        if(operation_count > 0) set_operation_count(category, operation_count);
    }

    void
    set_operation_count(uint32_t category, uint32_t operation_count)
    {
        for(auto& itr : category_operations)
        {
            if(itr.category == category)
            {
                itr.operation_count = operation_count;
                return;
            }
        }
        category_operations.emplace_back(tracing_category_config{category, operation_count});
    }

    bool
    is_category_enabled(uint32_t category) const
    {
        return (enabled_categories.count(category) > 0);
    }

    uint32_t
    get_operation_count(uint32_t category) const
    {
        for(const auto& itr : category_operations)
        {
            if(itr.category == category) return itr.operation_count;
        }
        return 0;
    }

    bool
    is_valid_record(uint32_t category, uint32_t operation) const
    {
        if(!is_category_enabled(category)) return false;

        auto operation_count = get_operation_count(category);
        if(operation_count == 0) return true;

        return (operation < operation_count);
    }
};

void
setup() CLIENT_API;

void
shutdown() CLIENT_API;

void
start() CLIENT_API;

void
stop() CLIENT_API;

void
identify(uint64_t corr_id) CLIENT_API;

buffered_tracing_config&
get_buffered_tracing_config() CLIENT_API;

bool
is_category_enabled(uint32_t category) CLIENT_API;

bool
is_valid_buffered_record(uint32_t category, uint32_t operation) CLIENT_API;
}  // namespace client