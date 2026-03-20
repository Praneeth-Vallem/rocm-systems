// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file refcount.h
/// @brief Thread-safe reference-counting mixin for opaque C API handles.

#ifndef ROCJITSU_REFCOUNT_H_
#define ROCJITSU_REFCOUNT_H_

#include <atomic>
#include <cstdint>

namespace rocjitsu {

/// @brief Reference-counting mixin for opaque C API handle types.
///
/// @details Provides retain/release semantics with atomic operations for thread safety.
/// Objects start with refcount 0. `retain` increments, `release` decrements.
/// When `destroy` is called, the object is marked for destruction. The backing
/// memory is freed when refcount reaches 0 after a destroy has been requested.
class RefCounted {
public:
  /// @brief Increment the reference count.
  void retain() { ref_count_.fetch_add(1, std::memory_order_relaxed); }

  /// @brief Decrement the reference count.
  /// @retval true The object has been destroyed and the reference count reached 0; caller should
  /// free.
  /// @retval false The object still has references or has not been destroyed.
  bool release() {
    uint32_t prev = ref_count_.load(std::memory_order_relaxed);
    if (prev > 0)
      ref_count_.fetch_sub(1, std::memory_order_acq_rel);
    return destroyed_.load(std::memory_order_acquire) &&
           ref_count_.load(std::memory_order_acquire) == 0;
  }

  /// @brief Mark the object for destruction.
  /// @retval true The reference count is already 0; caller should free immediately.
  /// @retval false Outstanding references remain; the last release() will trigger freeing.
  bool destroy() {
    destroyed_.store(true, std::memory_order_release);
    return ref_count_.load(std::memory_order_acquire) == 0;
  }

  /// @brief Current reference count.
  /// @returns The reference count value (relaxed load).
  uint32_t ref_count() const { return ref_count_.load(std::memory_order_relaxed); }

  /// @brief Whether destroy() has been called.
  /// @retval true The object has been marked for destruction.
  /// @retval false The object is still alive.
  bool is_destroyed() const { return destroyed_.load(std::memory_order_relaxed); }

private:
  std::atomic<uint32_t> ref_count_{0};
  std::atomic<bool> destroyed_{false};
};

} // namespace rocjitsu

#endif // ROCJITSU_REFCOUNT_H_
