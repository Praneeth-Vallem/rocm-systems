// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file wall_clock_sync.h
/// @brief Wall-clock-coupled simulation time synchronization.

#ifndef SIMDOJO_SIM_WALL_CLOCK_SYNC_H_
#define SIMDOJO_SIM_WALL_CLOCK_SYNC_H_

#include "simdojo/sim/sim_types.h"

#include <atomic>
#include <chrono>
#include <thread>

namespace simdojo {

/// @brief Wall-clock-coupled simulation time synchronization.
///
/// @details Maintains a linear mapping between wall-clock (real) time and
/// simulation time, enabling three capabilities:
///
/// 1. **Timestamp translation** - converts "now" in wall time to a simulation
///    tick, so async events injected from external threads (driver doorbells,
///    host CPU submissions) get realistic timestamps instead of tick 0.
///
/// 2. **Throttling** - prevents the simulation from running ahead of real
///    time. After processing events, if sim time is ahead of the wall-clock
///    projection, the engine sleeps until they converge.
///
/// 3. **Idle advancement** - during quiescent periods (queue empty, primaries
///    active), the engine uses timed waits instead of unbounded blocking.
///    On timeout, sim time advances proportionally to elapsed wall time,
///    keeping the clock moving even when no events are pending.
///
/// The mapping is: `sim_tick = sim_anchor + (wall_now - wall_anchor) * ratio * TICKS_PER_NS`
///
/// where TICKS_PER_NS = 1000 (1 ns = 1000 ps = 1000 ticks at ps resolution).
///
/// Disabled by default (ratio = 0). When disabled, the engine runs as fast
/// as possible with no wall-clock coupling.
///
/// Integration points in SimulationEngine:
/// - `run()`: calls `anchor()` after startup, before entering worker loop
/// - `worker_loop` (single-threaded): calls `throttle()` after draining
///   events; uses `idle_wait_duration()` for timed CV waits during quiescence
/// - `worker_loop` (multi-threaded): calls `throttle()` after each epoch;
///   partition 0 uses `idle_wait_duration()` for GVT wait timeouts
/// - `schedule_event_now()`: uses `sim_tick_now()` for wall-time-translated
///   timestamps on externally injected events
class WallClockSync {
public:
  /// @brief Construct with a target simulation-to-real-time ratio.
  /// @param ratio Sim-time per wall-time. 1.0 = real-time (1 sim ns = 1 wall
  ///   ns). 2.0 = sim runs at 2x real speed. 0.0 = disabled.
  explicit WallClockSync(double ratio = 0.0) : ratio_(ratio) {}

  /// @brief Whether wall-clock sync is active.
  /// @retval true Wall-clock sync is enabled (ratio > 0).
  /// @retval false Disabled; simulation runs as fast as possible.
  bool enabled() const { return ratio_ > 0.0; }

  /// @brief Set the anchor point mapping wall-clock now to a simulation tick.
  ///
  /// @details Call once at simulation start (after startup, before the main
  /// loop). All subsequent time translations are relative to this anchor.
  /// @param sim_tick The simulation tick at anchor time (typically 0).
  void anchor(Tick sim_tick = 0) {
    lock();
    wall_anchor_ = clock::now();
    sim_anchor_ = sim_tick;
    unlock();
  }

  /// @brief Re-anchor after a quiescent period to avoid a time jump.
  ///
  /// @details When the engine wakes from a blocked wait (async event arrived
  /// after idle period), the wall clock has advanced but the sim clock has
  /// not. Re-anchoring prevents the sim from trying to "catch up" to the
  /// wall time that elapsed while idle.
  /// @param sim_tick The current simulation tick at re-anchor time.
  void reanchor(Tick sim_tick) {
    lock();
    wall_anchor_ = clock::now();
    sim_anchor_ = sim_tick;
    unlock();
  }

  /// @brief Convert the current wall-clock time to a simulation tick.
  ///
  /// @details Thread-safe. Returns the simulation tick that corresponds to
  /// "right now" according to the wall-clock mapping. Used by
  /// `schedule_event_now()` to assign realistic timestamps to externally
  /// injected events.
  /// @returns Simulation tick corresponding to current wall time.
  Tick sim_tick_now() const {
    if (!enabled())
      return 0;
    lock();
    auto anchor_wall = wall_anchor_;
    auto anchor_sim = sim_anchor_;
    unlock();
    auto elapsed = clock::now() - anchor_wall;
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count();
    if (ns <= 0)
      return anchor_sim;
    return anchor_sim + static_cast<Tick>(static_cast<double>(ns) * ratio_ * TICKS_PER_NS);
  }

  /// @brief If simulation time is ahead of projected wall time, sleep.
  ///
  /// @details Compares `current_sim_tick` against the wall-clock projection.
  /// If sim is ahead, sleeps until wall time catches up. If sim is behind
  /// or equal, returns immediately. No-op when disabled.
  /// @param current_sim_tick The simulation's current tick.
  void throttle(Tick current_sim_tick) const {
    if (!enabled())
      return;
    Tick projected = sim_tick_now();
    if (current_sim_tick <= projected)
      return;
    // Sim is ahead - compute how long to wait.
    double excess_ticks = static_cast<double>(current_sim_tick - projected);
    double wait_ns = excess_ticks / (ratio_ * TICKS_PER_NS);
    auto wait = std::chrono::nanoseconds(static_cast<int64_t>(wait_ns));
    std::this_thread::sleep_for(wait);
  }

  /// @brief Compute the maximum duration to block during a quiescent wait.
  ///
  /// @details Returns a bounded duration for `advance_cv.wait_for()` during
  /// quiescent periods. When the timeout fires, the engine re-checks for
  /// async events and advances the sim clock proportionally to elapsed wall
  /// time. Returns 0ms when disabled (use unbounded wait).
  /// @returns Wait duration, or 0ms if wall-clock sync is disabled.
  std::chrono::milliseconds idle_wait_duration() const {
    if (!enabled())
      return std::chrono::milliseconds(0);
    // Wake periodically to advance sim time. 10ms is responsive enough
    // for interactive use without excessive wakeups.
    return std::chrono::milliseconds(10);
  }

  /// @brief Return the configured ratio.
  /// @returns Sim-time per wall-time ratio (0.0 if disabled).
  double ratio() const { return ratio_; }

private:
  using clock = std::chrono::steady_clock;

  void lock() const {
    while (spinlock_.test_and_set(std::memory_order_acquire)) {
    }
  }
  void unlock() const { spinlock_.clear(std::memory_order_release); }

  static constexpr double TICKS_PER_NS = 1000.0; ///< 1 ns = 1000 ps = 1000 ticks.

  double ratio_ = 0.0;                                   ///< Sim-time per wall-time (0 = disabled).
  mutable std::atomic_flag spinlock_ = ATOMIC_FLAG_INIT; ///< Protects anchor state.
  clock::time_point wall_anchor_{};                      ///< Wall-clock time at anchor.
  Tick sim_anchor_ = 0;                                  ///< Simulation tick at anchor.
};

} // namespace simdojo

#endif // SIMDOJO_SIM_WALL_CLOCK_SYNC_H_
