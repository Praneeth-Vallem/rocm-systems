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

#include "lib/common/environment.hpp"
#include "lib/common/demangle.hpp"
#include "lib/common/logging.hpp"

#include <fmt/format.h>

#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <string_view>
#include <unistd.h>

namespace rocprofiler
{
namespace common
{
namespace impl
{
std::string
get_env(std::string_view env_id, std::string_view _default)
{
    if(env_id.empty()) return std::string{_default};
    char* env_var = ::std::getenv(env_id.data());
    if(env_var) return std::string{env_var};
    return std::string{_default};
}

std::string
get_env(std::string_view env_id, const char* _default)
{
    return get_env(env_id, std::string_view{_default});
}

bool
get_env(std::string_view env_id, bool _default)
{
    if(env_id.empty()) return _default;
    char* env_var = ::std::getenv(env_id.data());
    if(env_var)
    {
        if(std::string_view{env_var}.empty())
        {
            ROCP_FATAL << fmt::format("No boolean value provided for {}", env_id);
        }

        if(std::string_view{env_var}.find_first_not_of("0123456789") == std::string_view::npos)
        {
            return static_cast<bool>(std::stoi(env_var));
        }

        for(size_t i = 0; i < std::string_view{env_var}.length(); ++i)
            env_var[i] = static_cast<char>(tolower(env_var[i]));

        for(const auto& itr : {"off", "false", "no", "n", "f", "0"})
            if(std::string_view{env_var} == itr) return false;

        return true;
    }
    return _default;
}

template <typename Tp>
Tp
get_env(std::string_view env_id,
        Tp               _default,
        std::enable_if_t<std::is_integral<Tp>::value || std::is_floating_point<Tp>::value, sfinae>)
{
    static_assert(!std::is_same<Tp, bool>::value, "unexpected! should be using bool overload");
    static_assert(
        sizeof(Tp) <= sizeof(uint64_t),
        "change use of stol/stoul if instantiating for type larger than a 64-bit integer");

    if(env_id.empty()) return _default;
    char* env_var = ::std::getenv(env_id.data());
    if(env_var)
    {
        try
        {
            if constexpr(std::is_integral<Tp>::value)
            {
                // use stol/stoul
                if constexpr(std::is_signed<Tp>::value)
                    return static_cast<Tp>(std::stol(env_var));
                else
                    return static_cast<Tp>(std::stoul(env_var));
            }
            else if constexpr(std::is_floating_point<Tp>::value)
            {
                return static_cast<Tp>(std::stod(env_var));
            }
        } catch(std::exception& _e)
        {
            ROCP_ERROR << "[rocprofiler][get_env] Exception thrown converting getenv(\"" << env_id
                       << "\") = " << env_var << " to " << cxx_demangle(typeid(Tp).name())
                       << " :: " << _e.what() << ". Using default value of " << _default << "\n";
        }
        return _default;
    }
    return _default;
}

int
set_env(std::string_view env_id, bool value, int override)
{
    return ::setenv(env_id.data(), (value) ? "1" : "0", override);
}

template <typename Tp>
int
set_env(std::string_view env_id,
        Tp               value,  // NOLINT(performance-unnecessary-value-param)
        int              override)
{
    auto str_value = std::stringstream{};
    str_value << value;
    return ::setenv(env_id.data(), str_value.str().c_str(), override);
}

#define SPECIALIZE_GET_ENV(TYPE)                                                                   \
    template TYPE get_env<TYPE>(                                                                   \
        std::string_view,                                                                          \
        TYPE,                                                                                      \
        std::enable_if_t<std::is_integral<TYPE>::value || std::is_floating_point<TYPE>::value,    \
                         sfinae>);

#define SPECIALIZE_SET_ENV(TYPE) template int set_env<TYPE>(std::string_view, TYPE, int);

SPECIALIZE_GET_ENV(int8_t)
SPECIALIZE_GET_ENV(int16_t)
SPECIALIZE_GET_ENV(int32_t)
SPECIALIZE_GET_ENV(int64_t)
SPECIALIZE_GET_ENV(uint8_t)
SPECIALIZE_GET_ENV(uint16_t)
SPECIALIZE_GET_ENV(uint32_t)
SPECIALIZE_GET_ENV(uint64_t)
SPECIALIZE_GET_ENV(float)
SPECIALIZE_GET_ENV(double)

SPECIALIZE_SET_ENV(const char*)
SPECIALIZE_SET_ENV(std::string)
SPECIALIZE_SET_ENV(std::string_view)
SPECIALIZE_SET_ENV(float)
SPECIALIZE_SET_ENV(double)
SPECIALIZE_SET_ENV(int8_t)
SPECIALIZE_SET_ENV(int16_t)
SPECIALIZE_SET_ENV(int32_t)
SPECIALIZE_SET_ENV(int64_t)
SPECIALIZE_SET_ENV(uint8_t)
SPECIALIZE_SET_ENV(uint16_t)
SPECIALIZE_SET_ENV(uint32_t)
SPECIALIZE_SET_ENV(uint64_t)
}  // namespace impl

namespace
{
void
configure_child_process_environment()
{
    // Record the originating process ID for child/fork detection if not already set.
    auto parent_pid = impl::get_env<uint64_t>("ROCPROFILER_PARENT_PID", 0);
    auto current_pid =
        static_cast<uint64_t>(::getpid());  // NOLINT(performance-no-int-to-ptr, hicpp-signed-bitwise)

    if(parent_pid == 0)
    {
        (void) impl::set_env("ROCPROFILER_PARENT_PID", current_pid, 0);
        return;
    }

    // If the PID changed, we are executing in a child process that inherited environment
    // and potentially unsafe profiler/marker interception state from a fork.
    if(parent_pid != current_pid)
    {
        (void) impl::set_env("ROCPROFILER_IS_CHILD_PROCESS", true, 1);

        // Marker tracing / ROCTX interception is not safe to inherit across fork for some
        // DataLoader worker configurations. Provide an environment-driven hook to disable it
        // in child processes by default, while allowing explicit opt-in to child profiling.
        const auto allow_child_marker_trace =
            impl::get_env("ROCPROFILER_ENABLE_CHILD_MARKER_TRACE",
                          impl::get_env("ROCPROFILER_CHILD_MARKER_TRACE", false));

        if(!allow_child_marker_trace)
        {
            (void) impl::set_env("ROCPROFILER_DISABLE_ROCTX_INTERCEPT", true, 1);
            (void) impl::set_env("ROCPROFILER_MARKER_TRACE", false, 1);
            (void) impl::set_env("ROCPROFILER_CHILD_DISABLE_ROCTX_INTERCEPT", true, 1);
        }
        else
        {
            // Signal downstream initialization to safely reinitialize marker/ROCTX state
            // instead of using inherited state.
            (void) impl::set_env("ROCPROFILER_REINITIALIZE_ROCTX_INTERCEPT", true, 1);
        }
    }
}
}  // namespace

env_store::env_store(std::initializer_list<env_config>&& _container)
{
    configure_child_process_environment();

    for(const auto& itr : _container)
    {
        if(itr.env_name.empty()) continue;

        auto env_name = std::string{itr.env_name};
        auto env_val  = impl::get_env(itr.env_name, itr.default_value);

        if(itr.description.empty())
        {
            emplace(env_name, env_val);
        }
        else
        {
            emplace(env_name, env_variable{env_name, env_val, itr.description});
        }
    }
}
}  // namespace common
}  // namespace rocprofiler