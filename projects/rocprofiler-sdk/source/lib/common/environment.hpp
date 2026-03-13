// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#pragma once

#include "lib/common/logging.hpp"

#include <unistd.h>
#include <initializer_list>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace rocprofiler
{
namespace common
{
namespace impl
{
struct sfinae
{};

std::string get_env(std::string_view, std::string_view);

std::string
get_env(std::string_view, const char*);

bool
get_env(std::string_view, bool);

template <typename Tp>
Tp get_env(std::string_view,
           Tp,
           std::enable_if_t<std::is_integral<Tp>::value || std::is_floating_point<Tp>::value,
                            sfinae> = {});

int
set_env(std::string_view, bool, int override = 0);

template <typename Tp>
int
set_env(std::string_view, Tp, int override = 0);

// additional environment and executable-discovery interfaces for ROCm/hipcc setup validation
bool
has_env(std::string_view);

std::string
find_executable(std::string_view exe_name);

std::string
find_executable(std::string_view exe_name, std::string_view search_path);

std::string
find_hipcc();

std::string
find_rocm_path();

bool
is_executable(std::string_view);

bool
validate_rocm_setup(std::string* hipcc_path = nullptr,
                    std::string* rocm_path  = nullptr,
                    std::string* error_msg  = nullptr);
}  // namespace impl

template <typename Tp>
inline auto
get_env(std::string_view env_id, Tp&& _default)
{
    if constexpr(std::is_enum<Tp>::value)
    {
        using Up = std::underlying_type_t<Tp>;
        // cast to underlying type -> get_env -> cast to enum type
        return static_cast<Tp>(impl::get_env(env_id, static_cast<Up>(_default)));
    }
    else
    {
        return impl::get_env(env_id, std::forward<Tp>(_default));
    }
}

template <typename Tp>
inline auto
set_env(std::string_view env_id, Tp&& value, int override = 0)
{
    return impl::set_env(env_id, std::forward<Tp>(value), override);
}

inline bool
has_env(std::string_view env_id)
{
    return impl::has_env(env_id);
}

inline std::string
find_executable(std::string_view exe_name)
{
    return impl::find_executable(exe_name);
}

inline std::string
find_executable(std::string_view exe_name, std::string_view search_path)
{
    return impl::find_executable(exe_name, search_path);
}

inline std::string
find_hipcc()
{
    return impl::find_hipcc();
}

inline std::string
find_rocm_path()
{
    return impl::find_rocm_path();
}

inline bool
is_executable(std::string_view file_path)
{
    return impl::is_executable(file_path);
}

inline bool
validate_rocm_setup(std::string* hipcc_path = nullptr,
                    std::string* rocm_path  = nullptr,
                    std::string* error_msg  = nullptr)
{
    return impl::validate_rocm_setup(hipcc_path, rocm_path, error_msg);
}

struct env_config
{
    std::string env_name  = {};
    std::string env_value = {};
    int         overwrite = 0;

    auto operator()(bool _verbose = false) const
    {
        if(env_name.empty())
            return -1;
        else if(_verbose)
        {
            ROCP_INFO << "[rocprofiler][set_env] setenv(\"" << env_name << "\", \"" << env_value
                      << "\", " << overwrite << ")\n";
        }
        return (env_value.empty() && overwrite > 0)
                   ? unsetenv(env_name.c_str())
                   : setenv(env_name.c_str(), env_value.c_str(), overwrite);
    }
};

struct env_store
{
    template <template <typename, typename...> class ContainerT, typename... TailT>
    explicit env_store(ContainerT<env_config, TailT...>&& _container);
    explicit env_store(std::initializer_list<env_config>&& _container);

    ~env_store();
    env_store(const env_store&)     = default;
    env_store(env_store&&) noexcept = default;
    env_store& operator=(const env_store&) = default;
    env_store& operator=(env_store&&) noexcept = default;

    bool push();
    bool pop(bool unset_if_empty = true);
    bool is_pushed() const { return m_pushed; }

private:
    bool                    m_pushed   = false;
    std::vector<env_config> m_original = {};
    std::vector<env_config> m_modified = {};
};

template <template <typename, typename...> class ContainerT, typename... TailT>
env_store::env_store(ContainerT<env_config, TailT...>&& _container)
{
    for(const auto& itr : _container)
    {
        m_original.emplace_back(env_config{itr.env_name, get_env(itr.env_name, ""), 1});
        m_modified.emplace_back(env_config{itr.env_name, itr.env_value, 1});
    }
}
}  // namespace common
}  // namespace rocprofiler