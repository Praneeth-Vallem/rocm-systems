// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file simulation.h
/// @brief SimulationEngine and PartitionContext for single- and multi-threaded PDES execution.

#ifndef SIMDOJO_SIM_SIMULATION_H_
#define SIMDOJO_SIM_SIMULATION_H_

#include "simdojo/sim/event_queue.h"
#include "simdojo/sim/topology.h"
#include "simdojo/sim/wall_clock_sync.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace simdojo {

/// @brief Per-partition simulation context.
///
/// @details Each worker thread manages exactly one partition. Holds the
/// partition's event queue, incoming message queues, and synchronization
/// state. In single-threaded mode only the event queue and async wakeup
/// fields are used; the vector clock and LBTS fields are active only in
/// multi-threaded mode for causal LBTS synchronization.
struct alignas(64) PartitionContext {
  PartitionContext(PartitionID pid, uint32_t num_partitions)
      : partition_id(pid), vclock(num_partitions, 0) {
    incoming.reserve(num_partitions);
    for (uint32_t i = 0; i < num_partitions; ++i)
      incoming.push_back(std::make_unique<CrossPartitionQueue>());
  }

  // Non-copyable, non-movable (mutex and condition_variable are not movable).
  PartitionContext(const PartitionContext &) = delete;
  PartitionContext &operator=(const PartitionContext &) = delete;
  PartitionContext(PartitionContext &&) = delete;
  PartitionContext &operator=(PartitionContext &&) = delete;

  PartitionID partition_id;           ///< This partition's ID.
  EventQueue event_queue;             ///< Thread-local event priority queue.
  Tick local_min_outgoing = TICK_MAX; ///< Min timestamp of outgoing cross-partition events.

  /// @brief Incoming cross-partition events.
  /// One CrossPartitionQueue per source partition.
  std::vector<std::unique_ptr<CrossPartitionQueue>> incoming;

  /// @brief Vector clock for causal LBTS.
  /// vclock[i] = latest known safe time for partition i. Updated by merging
  /// incoming vector clocks from neighbors during drain.
  std::vector<Tick> vclock;

  /// @brief Locally computed safe time from the vector clock.
  /// In multi-threaded mode, this is derived from neighbor entries in vclock
  /// plus link latencies. In single-threaded mode, computed directly.
  Tick local_lbts = 0;

  /// @brief Precomputed minimum link latencies from each neighbor partition.
  /// Populated during setup_partitions() from boundary links.
  std::vector<std::pair<PartitionID, Tick>> neighbor_latencies;

  std::mutex advance_mutex;
  /// @brief Per-partition condition variable for neighbor-driven wakeup.
  /// Workers block here when no progress is possible, woken by neighbor
  /// advances, async events, or shutdown.
  std::condition_variable advance_cv;
  std::atomic<bool> advance_pending{false};

  uint64_t events_processed = 0;        ///< Total events processed by this partition.
  uint64_t timestamp_advances_sent = 0; ///< Total timestamp advances sent for LBTS.
  uint64_t epochs_completed = 0;        ///< Number of LBTS epochs completed.

  /// @brief Drain all incoming queues into the local event queue.
  void drain_incoming();

  /// @brief Drain all incoming queues, merging sender vector clocks.
  void drain_and_merge_incoming();

  /// @brief Compute the local safe time from this partition's vector clock.
  /// Returns min over neighbors j of (vclock[j] + latency(j→self)),
  /// also considering the local event queue next time and local min outgoing.
  Tick compute_local_lbts() const;
};

/// @brief Callback invoked by the service loop once per LBTS epoch.
using ServiceCallback = std::function<void(class SimulationEngine &)>;

/// @brief The main simulation engine.
///
/// @details Owns the simulation topology and drives the full lifecycle: build,
/// run/step, and shutdown.
///
/// **Single-threaded mode** (`num_threads == 1`): drains events directly from
/// the priority queue in timestamp order. No LBTS computation, no vector
/// clocks, no epoch structure. When the queue empties, drains async buffers
/// and either terminates or blocks on a condition variable for external events.
///
/// **Multi-threaded mode** (`num_threads > 1`): runs a conservative
/// Chandy-Misra-Bryant Parallel Discrete Event Simulation (PDES) loop with
/// causal Lower Bound on Time Stamp (LBTS) synchronization. Each partition
/// independently computes its local safe time from neighbor vector clocks,
/// enabling partitions to advance at their own pace without global barriers.
///
/// Components participate in the simulation by scheduling events during
/// initialize() or startup(). Untimed (functional) components use the
/// Functional\<Base\> CRTP mixin, which self-schedules a timer callback and
/// re-enqueues after each step(). If no events are ever scheduled and no
/// primaries are registered, the simulation terminates via quiescence. If
/// primaries are registered, the engine blocks until async events arrive or
/// all primaries signal completion.
///
/// Typical usage:
/// @code
///   SimulationEngine engine(config);
///   engine.topology().set_root(std::move(my_model));
///   engine.build();
///   auto exit = engine.run();
/// @endcode
class SimulationEngine {
public:
  /// @brief Configuration parameters for the simulation engine.
  struct Config {
    Tick max_ticks = 0;         ///< Simulation stops at this tick (0 = unlimited).
    uint32_t num_threads = 1;   ///< Number of worker threads (one per partition).
    uint32_t gvt_interval = 16; ///< Epochs between Global Virtual Time (GVT) computations.
    double wall_clock_ratio =
        0.0;              ///< Sim-time per wall-time ratio (0 = disabled, 1.0 = real-time).
    bool verbose = false; ///< Print progress to debug output.
  };

  /// @brief Construct with config; caller populates topology(), then calls build().
  /// @param config Engine configuration parameters.
  explicit SimulationEngine(Config config);

  ~SimulationEngine();

  SimulationEngine(const SimulationEngine &) = delete;
  SimulationEngine &operator=(const SimulationEngine &) = delete;
  SimulationEngine(SimulationEngine &&) = delete;
  SimulationEngine &operator=(SimulationEngine &&) = delete;

  /// @brief Partition the topology and prepare the engine for execution.
  ///
  /// @details Required after the Config-only constructor once the topology is
  /// populated. Also used to rebuild after shutdown().
  void build();

  /// @brief Tear down engine state (finalize components, join workers).
  ///
  /// After shutdown(), the engine can be rebuilt with a new build() call.
  /// Called automatically by the destructor if still built.
  void shutdown();

  /// @brief Return whether the engine has been built and is ready for execution.
  /// @retval true Engine is built and ready for run() or step().
  /// @retval false Engine has not been built or has been shut down.
  bool is_built() const { return built_; }

  /// @brief Access the topology for model setup (add components, links, clock domains).
  ///
  /// @details Mutable access is allowed during setup and initialize(). After the
  /// simulation starts running, the topology is frozen and only const access is
  /// permitted (enforced by assertion in debug builds).
  Topology &topology() {
    assert(!running_ && "topology is read-only while the simulation is running");
    return topology_;
  }
  const Topology &topology() const { return topology_; }

  /// @brief Run the simulation to completion.
  ///
  /// @details Initializes and starts all components, then enters the PDES
  /// epoch loop. Finalizes all components before returning.
  /// @returns An ExitStatus describing why the simulation stopped.
  ExitStatus run();

  /// @brief Advance the simulation by one tick (single-threaded only).
  ///
  /// @details On the first call, initializes and starts all components.
  /// Each subsequent call processes all events at the next timestamp, then
  /// returns. If the queue is empty but primaries are registered, returns
  /// true so the caller can poll for async events.
  /// @retval true Simulation can continue.
  /// @retval false Simulation is done (query last_exit() for details).
  bool step();

  /// @brief Request the simulation to stop.
  ///
  /// @details Thread-safe. Can be called from any thread (worker, external, signal
  /// handler via flag).
  /// @param reason Human-readable reason for stopping.
  /// @param code Exit code (0 = success).
  void request_exit(std::string reason, int code = 0);

  /// @brief Register the calling component as a primary (work-producing).
  ///
  /// @details Primary components participate in the end-of-simulation consensus
  /// protocol. The simulation ends gracefully when all registered primaries
  /// have called primary_ok_to_end(). Must be called during initialize()
  /// or startup(), before the main simulation loop starts.
  void register_as_primary();

  /// @brief Signal that this primary component is done producing work.
  ///
  /// @details Thread-safe. When all registered primaries have signaled OK, the
  /// engine sets exit reason COMPLETED at the next epoch boundary.
  void primary_ok_to_end();

  /// @brief Re-arm this primary component (it has new work).
  ///
  /// @details Thread-safe. Reverses a previous primary_ok_to_end() call.
  void primary_do_not_end();

  /// @brief Return the exit event from the last completed run() or step().
  /// @returns Const reference to the stored exit event.
  const ExitStatus &last_exit() const { return exit_event_; }

  /// @brief Enqueue an event into the target component's partition queue.
  ///
  /// @details Must only be called from the thread that owns the target partition.
  /// For cross-partition delivery, use send_cross_partition() instead.
  /// @param event Reusable event descriptor.
  /// @param timestamp Simulation tick at which the event fires.
  /// @param message Optional message payload (ownership transferred).
  void schedule_event(Event *event, Tick timestamp, std::unique_ptr<Message> message = nullptr);

  /// @brief Enqueue an event from any thread (thread-safe).
  ///
  /// @details The event is buffered and drained into the target partition's queue
  /// at the next safe point (epoch boundary or start of step).
  /// @param event Reusable event descriptor.
  /// @param timestamp Simulation tick at which the event fires.
  /// @param message Optional message payload (ownership transferred).
  void schedule_event_async(Event *event, Tick timestamp,
                            std::unique_ptr<Message> message = nullptr);

  /// @brief Enqueue an event from any thread at the current simulation time.
  ///
  /// @details Thread-safe. When wall-clock sync is enabled, translates the
  /// current wall time to a simulation tick. When disabled, uses the engine's
  /// current simulation time. Preferred over schedule_event_async() for
  /// external stimulus (driver doorbells, host submissions).
  /// @param event Reusable event descriptor.
  /// @param message Optional message payload (ownership transferred).
  void schedule_event_now(Event *event, std::unique_ptr<Message> message = nullptr);

  /// @brief Deposit an event into another partition's cross-partition inbox.
  /// @param src_partition Source partition ID (selects the incoming queue).
  /// @param dst_partition Destination partition ID.
  /// @param event Reusable event descriptor.
  /// @param timestamp Simulation tick at which the event fires.
  /// @param message Optional message payload (ownership transferred).
  void send_cross_partition(PartitionID src_partition, PartitionID dst_partition, Event *event,
                            Tick timestamp, std::unique_ptr<Message> message = nullptr);

  /// @brief Register a callback invoked during periodic GVT computation.
  /// Callbacks run from partition 0's worker thread during GVT sweeps.
  /// @param callback Function called with a reference to this engine.
  void register_service_callback(ServiceCallback callback) {
    service_callbacks_.push_back(std::move(callback));
  }

  /// @brief Return the number of partition contexts.
  /// @returns Number of partitions.
  uint32_t num_contexts() const { return static_cast<uint32_t>(contexts_.size()); }

  /// @brief Access a partition context by partition ID (const).
  /// @param pid The partition to look up.
  /// @returns Const reference to the PartitionContext.
  const PartitionContext &context(PartitionID pid) const {
    assert(pid < contexts_.size() && "partition ID out of range");
    return *contexts_[pid];
  }

  /// @brief Access a partition context by partition ID.
  /// @param pid The partition to look up.
  /// @returns Mutable reference to the PartitionContext.
  PartitionContext &context(PartitionID pid) {
    assert(pid < contexts_.size() && "partition ID out of range");
    return *contexts_[pid];
  }

  /// @brief Return the current simulation time.
  /// @returns The latest processed simulation tick.
  Tick global_time() const { return current_time_.load(std::memory_order_acquire); }

  /// @brief Access the wall-clock synchronization state.
  /// @returns Const reference to the WallClockSync.
  const WallClockSync &wall_clock_sync() const { return wall_clock_sync_; }

private:
  /// @brief Worker loop executed by each partition thread.
  void worker_loop(PartitionID partition_id);

  /// @brief Process a single heap entry: execute its event handler if present.
  /// @param ctx The partition context that owns the event queue.
  /// @param entry The heap entry to process.
  void process_event(PartitionContext &ctx, EventQueueEntry &entry);

  /// @brief Compute GVT (Global Virtual Time) from all partition local LBTS values.
  /// Updates current_time_, runs service callbacks, and checks termination.
  /// Called periodically by partition 0 in multi-threaded mode.
  void compute_gvt();

  /// @brief Send timestamp advances on all outgoing boundary links for a partition.
  /// @param ctx The source partition's context.
  /// @param part The source partition descriptor.
  void send_timestamp_advances(PartitionContext &ctx, const Partition &part);

  /// @brief Call initialize() on all components across all partitions.
  void initialize_components();

  /// @brief Call startup() on all components across all partitions.
  void startup_components();

  /// @brief Call finalize() on all components across all partitions.
  void finalize_components();

  /// @brief Drain all async event buffers into their partition queues.
  void drain_async_events();

  /// @brief Set the exit event (internal helper).
  void set_exit(ExitReason reason, Tick tick, std::string message, int code = 0);

  /// @brief Check if all registered primaries have signaled OK to end.
  /// @retval true All primaries are done (and at least one is registered).
  /// @retval false No primaries registered, or some are still active.
  bool all_primaries_done() const;

  /// @brief Check end-of-sim conditions (primaries, max_ticks, quiescence).
  /// Sets exit event and done_ flag if simulation should stop.
  /// @param lbts The current or newly computed LBTS value.
  /// @retval true Simulation should stop.
  /// @retval false Simulation continues.
  bool check_termination(Tick lbts);

  /// @brief Return the minimum latency among cross-partition links (lookahead).
  /// @returns Minimum cross-partition link latency, or TICK_MAX if none.
  Tick min_cross_partition_latency() const { return min_cross_latency_; }

  /// @brief Set up partition contexts, async queues, and engine pointers.
  void setup_partitions();

  Topology topology_; ///< The simulation topology.
  Config config_;     ///< Engine configuration.
  /// @brief Reusable event for timestamp advance entries (no handler, never executed).
  Event timestamp_advance_event_{nullptr, EventType::TIMESTAMP_ADVANCE};
  /// @brief Reusable event for the max-ticks exit sentinel.
  /// Scheduled at config_.max_ticks during init; when processed, triggers sim exit.
  Event max_ticks_event_{nullptr, EventType::SIM_EXIT};
  std::vector<std::unique_ptr<PartitionContext>>
      contexts_;                                   ///< Per-partition state (one per thread).
  std::vector<std::jthread> workers_;              ///< Worker threads (multi-threaded mode).
  std::vector<ServiceCallback> service_callbacks_; ///< Per-epoch service callbacks.
  std::atomic<Tick> current_time_{0};         ///< Current simulation time (latest processed tick).
  Tick min_cross_latency_ = TICK_MAX;         ///< Minimum cross-partition link latency.
  WallClockSync wall_clock_sync_;             ///< Wall-clock coupling (disabled by default).
  std::atomic<bool> done_{false};             ///< Signals simulation completion.
  std::atomic<uint32_t> primary_count_{0};    ///< Total registered primary components.
  std::atomic<uint32_t> primary_ok_count_{0}; ///< Primaries that signaled OK to end.
  ExitStatus exit_event_;                     ///< Exit information from the last run/step.
  bool built_ = false;                        ///< Whether build() or ctor setup has completed.
  bool running_ = false;          ///< True while simulation is running (topology frozen).
  bool step_initialized_ = false; ///< Whether step() has called initialize.

  /// @brief Per-partition async event buffer for cross-thread insertion.
  struct AsyncQueue {
    std::mutex mutex;
    std::vector<EventQueueEntry> events;
  };
  std::vector<std::unique_ptr<AsyncQueue>> async_queues_; ///< One per partition.

  std::mutex exit_mutex_; ///< Protects exit_event_ writes (first-writer-wins).

  /// @brief Wake all partition worker threads (for shutdown/exit).
  void notify_all_partitions();
};

} // namespace simdojo

#endif // SIMDOJO_SIM_SIMULATION_H_
