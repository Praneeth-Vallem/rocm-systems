// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "simdojo/sim/simulation.h"

#include "util/debug_print.h"

#include <algorithm>
#include <cassert>
#include <unordered_map>

namespace simdojo {

void PartitionContext::drain_incoming() {
  for (auto &queue : incoming)
    queue->drain_into(event_queue);
}

void PartitionContext::drain_and_merge_incoming() {
  for (auto &queue : incoming)
    queue->drain_into(event_queue, vclock);
  advance_pending.store(false, std::memory_order_release);
}

Tick PartitionContext::compute_local_lbts() const {
  Tick lbts = TICK_MAX;

  // Minimum over neighbor partitions: vclock[neighbor] + link latency from neighbor.
  for (auto &[neighbor_id, latency] : neighbor_latencies) {
    Tick neighbor_time = vclock[neighbor_id];
    if (neighbor_time == TICK_MAX || neighbor_time > TICK_MAX - latency)
      continue; // Saturate at TICK_MAX (idle neighbor doesn't constrain us).
    lbts = std::min(lbts, neighbor_time + latency);
  }

  // Also constrained by our own pending events and outgoing messages.
  lbts = std::min(lbts, event_queue.next_event_time());
  lbts = std::min(lbts, local_min_outgoing);

  return lbts;
}

SimulationEngine::SimulationEngine(Config config)
    : config_(config), wall_clock_sync_(config.wall_clock_ratio) {}

SimulationEngine::~SimulationEngine() {
  if (built_)
    shutdown();
}

void SimulationEngine::build() {
  assert(!built_ && "build() called twice without shutdown()");
  topology_.partition(config_.num_threads);
  setup_partitions();
}

void SimulationEngine::shutdown() {
  if (!built_)
    return;

  // Signal done and wake any blocked workers.
  done_.store(true, std::memory_order_release);
  notify_all_partitions();

  // jthreads auto-join on destruction; clear to join now.
  workers_.clear();

  // Finalize components if step() was used (run() finalizes on its own).
  if (step_initialized_)
    finalize_components();

  // Reset engine state for potential rebuild.
  running_ = false;
  contexts_.clear();
  async_queues_.clear();
  service_callbacks_.clear();
  min_cross_latency_ = TICK_MAX;
  step_initialized_ = false;
  built_ = false;
}

void SimulationEngine::setup_partitions() {
  uint32_t num_threads = config_.num_threads;
  assert(num_threads > 0);
  assert(topology_.partitions().size() == num_threads);

  contexts_.reserve(num_threads);
  for (uint32_t i = 0; i < num_threads; ++i)
    contexts_.push_back(std::make_unique<PartitionContext>(i, num_threads));

  async_queues_.reserve(num_threads);
  for (uint32_t i = 0; i < num_threads; ++i)
    async_queues_.push_back(std::make_unique<AsyncQueue>());

  // Compute minimum cross-partition link latency (lookahead).
  for (auto &link : topology_.links()) {
    if (link->is_cross_partition()) {
      assert(link->latency() > 0 && "cross-partition links require positive latency for LBTS");
      assert(dynamic_cast<QueuedLink *>(link.get()) == nullptr &&
             "QueuedLinks must not cross partition boundaries (they bypass the LBTS protocol)");
      min_cross_latency_ = std::min(min_cross_latency_, link->latency());
    }
  }

  // Precompute per-partition neighbor latencies from boundary links.
  for (uint32_t pid = 0; pid < num_threads; ++pid) {
    const Partition &part = topology_.partitions()[pid];
    // Build a map: neighbor_partition_id -> min latency from that neighbor to us.
    std::unordered_map<PartitionID, Tick> neighbor_min_latency;
    for (auto *link : part.boundary_links) {
      PartitionID src_part = link->src()->owner()->partition_id();
      PartitionID dst_part = link->dst()->owner()->partition_id();
      // We want inbound links: links where dst is in our partition.
      if (dst_part == pid && src_part != pid) {
        auto it = neighbor_min_latency.find(src_part);
        if (it == neighbor_min_latency.end())
          neighbor_min_latency[src_part] = link->latency();
        else
          it->second = std::min(it->second, link->latency());
      }
    }
    auto &ctx = *contexts_[pid];
    ctx.neighbor_latencies.reserve(neighbor_min_latency.size());
    for (auto &[nid, lat] : neighbor_min_latency)
      ctx.neighbor_latencies.emplace_back(nid, lat);
  }

  // Set engine pointer on all components.
  for (auto &part : topology_.partitions()) {
    for (auto *comp : part.components)
      comp->set_engine(this);
  }

  built_ = true;
}

ExitStatus SimulationEngine::run() {
  done_.store(false, std::memory_order_release);
  primary_count_.store(0, std::memory_order_release);
  primary_ok_count_.store(0, std::memory_order_release);
  exit_event_ = {};
  current_time_.store(0, std::memory_order_release);

  initialize_components();
  startup_components();
  running_ = true;

  wall_clock_sync_.anchor(0);

  // Schedule the max-ticks exit sentinel so that all real events at
  // max_ticks fire before the exit (SIM_EXIT has the lowest priority).
  if (config_.max_ticks > 0) {
    max_ticks_event_.set_handler([this](Tick ts, Message *) {
      set_exit(ExitReason::COMPLETED, ts, "max ticks reached");
      done_.store(true, std::memory_order_release);
      notify_all_partitions();
    });
    contexts_[0]->event_queue.push(
        EventQueueEntry{config_.max_ticks, 0, &max_ticks_event_, nullptr});
  }

  uint32_t num_threads = config_.num_threads;
  if (num_threads == 1) {
    worker_loop(0);
  } else {
    for (uint32_t i = 0; i < num_threads; ++i)
      workers_.emplace_back([this, i]() { worker_loop(i); });
    // Workers are self-coordinating via causal LBTS. Wait for all to finish.
    workers_.clear();
  }

  running_ = false;
  finalize_components();
  return exit_event_;
}

bool SimulationEngine::step() {
  assert(config_.num_threads == 1 && "step() requires single-threaded mode");

  if (!step_initialized_) {
    done_.store(false, std::memory_order_release);
    primary_count_.store(0, std::memory_order_release);
    primary_ok_count_.store(0, std::memory_order_release);
    exit_event_ = {};

    initialize_components();
    startup_components();

    if (config_.max_ticks > 0) {
      max_ticks_event_.set_handler([this](Tick ts, Message *) {
        set_exit(ExitReason::COMPLETED, ts, "max ticks reached");
        done_.store(true, std::memory_order_release);
      });
      contexts_[0]->event_queue.push(
          EventQueueEntry{config_.max_ticks, 0, &max_ticks_event_, nullptr});
    }
    step_initialized_ = true;
    running_ = true;
  }

  if (done_.load(std::memory_order_acquire))
    return false;

  drain_async_events();

  PartitionContext &ctx = *contexts_[0];

  if (ctx.event_queue.empty()) {
    if (check_termination(TICK_MAX)) {
      running_ = false;
      finalize_components();
      step_initialized_ = false;
      return false;
    }
    return true;
  }

  // Process all events at the minimum timestamp (one tick step).
  Tick step_tick = ctx.event_queue.next_event_time();
  while (!ctx.event_queue.empty() && ctx.event_queue.next_event_time() == step_tick) {
    auto entry = ctx.event_queue.pop();
    process_event(ctx, entry);
    if (done_.load(std::memory_order_acquire)) {
      current_time_.store(step_tick, std::memory_order_release);
      running_ = false;
      finalize_components();
      step_initialized_ = false;
      return false;
    }
  }

  current_time_.store(step_tick, std::memory_order_release);

  for (auto &cb : service_callbacks_)
    cb(*this);

  return true;
}

void SimulationEngine::worker_loop(PartitionID partition_id) {
  PartitionContext &ctx = *contexts_[partition_id];
  const Partition &part = topology_.partitions()[partition_id];
  uint32_t num_threads = config_.num_threads;

  while (!done_.load(std::memory_order_acquire)) {
    if (num_threads == 1) {
      // Single-threaded: drain all events in timestamp order.
      while (!ctx.event_queue.empty()) {
        auto entry = ctx.event_queue.pop();
        process_event(ctx, entry);
        if (done_.load(std::memory_order_acquire))
          return;
      }

      // Queue drained - update global time for external observers.
      current_time_.store(ctx.event_queue.current_tick(), std::memory_order_release);

      // Throttle: if sim is ahead of wall clock, sleep until convergence.
      wall_clock_sync_.throttle(ctx.event_queue.current_tick());

      for (auto &cb : service_callbacks_)
        cb(*this);

      drain_async_events();

      // If async events added new work, continue processing.
      if (!ctx.event_queue.empty())
        continue;

      // Quiescent - check termination.
      if (check_termination(TICK_MAX))
        break;

      // Primaries still active - wait for an async event.
      // With wall-clock sync, use a timed wait so we can advance sim time
      // proportionally to elapsed wall time during idle periods.
      {
        std::unique_lock<std::mutex> lock(ctx.advance_mutex);
        auto timeout = wall_clock_sync_.idle_wait_duration();
        if (timeout.count() > 0) {
          ctx.advance_cv.wait_for(lock, timeout, [&ctx, this] {
            return ctx.advance_pending.load(std::memory_order_acquire) ||
                   done_.load(std::memory_order_acquire);
          });
        } else {
          ctx.advance_cv.wait(lock, [&ctx, this] {
            return ctx.advance_pending.load(std::memory_order_acquire) ||
                   done_.load(std::memory_order_acquire);
          });
        }
      }
      if (done_.load(std::memory_order_acquire))
        break;
      ctx.advance_pending.store(false, std::memory_order_release);

      // After waking from idle, re-anchor to prevent a time jump.
      wall_clock_sync_.reanchor(ctx.event_queue.current_tick());

      drain_async_events();
    } else {
      // Multi-threaded causal LBTS path.

      // 1. Process all events with timestamp <= local LBTS.
      while (!ctx.event_queue.empty() && ctx.event_queue.next_event_time() <= ctx.local_lbts) {
        auto entry = ctx.event_queue.pop();
        process_event(ctx, entry);
        if (done_.load(std::memory_order_acquire))
          return;
      }

      // 2. Update own vector clock entry to reflect earliest pending activity.
      Tick local_next = std::min(ctx.event_queue.next_event_time(), ctx.local_min_outgoing);
      ctx.vclock[partition_id] = local_next;

      // 3. Send vector clock advances to neighbors.
      if (min_cross_latency_ != TICK_MAX)
        send_timestamp_advances(ctx, part);

      // 4. Drain incoming queues and merge vector clocks.
      ctx.drain_and_merge_incoming();

      // Also drain async events (external thread injection).
      {
        auto &aq = *async_queues_[partition_id];
        std::lock_guard<std::mutex> lock(aq.mutex);
        for (auto &e : aq.events)
          ctx.event_queue.push(std::move(e));
        aq.events.clear();
      }

      // 5. Recompute local LBTS from merged vector clock.
      Tick old_lbts = ctx.local_lbts;
      ctx.local_lbts = ctx.compute_local_lbts();
      ctx.local_min_outgoing = TICK_MAX;
      ctx.epochs_completed++;

      // 6. If no progress possible, block until a neighbor sends an advance.
      if (ctx.local_lbts <= old_lbts &&
          (ctx.event_queue.empty() || ctx.event_queue.next_event_time() > ctx.local_lbts)) {
        std::unique_lock<std::mutex> lock(ctx.advance_mutex);
        ctx.advance_cv.wait(lock, [&ctx, this] {
          return ctx.advance_pending.load(std::memory_order_acquire) ||
                 done_.load(std::memory_order_acquire);
        });

        if (done_.load(std::memory_order_acquire))
          return;

        // Drain again after wakeup and recompute.
        ctx.drain_and_merge_incoming();

        // Drain async events for this partition.
        {
          auto &aq = *async_queues_[partition_id];
          std::lock_guard<std::mutex> lock_aq(aq.mutex);
          for (auto &e : aq.events)
            ctx.event_queue.push(std::move(e));
          aq.events.clear();
        }

        ctx.local_lbts = ctx.compute_local_lbts();
      }

      // 7. Periodic GVT computation (partition 0 only).
      if (partition_id == 0 && ctx.epochs_completed % config_.gvt_interval == 0)
        compute_gvt();
    }
  }
}

void SimulationEngine::compute_gvt() {
  Tick gvt = TICK_MAX;
  for (auto &ctx : contexts_)
    gvt = std::min(gvt, ctx->local_lbts);
  current_time_.store(gvt, std::memory_order_release);

  if (check_termination(gvt)) {
    notify_all_partitions();
    return;
  }

  for (auto &cb : service_callbacks_)
    cb(*this);

  if (config_.verbose)
    util::debug::print("GVT advanced to ", gvt);
}

void SimulationEngine::process_event(PartitionContext &ctx, EventQueueEntry &entry) {
  ctx.event_queue.set_current_tick(entry.timestamp);

  if (entry.event->has_handler()) {
    entry.event->execute(entry.timestamp, entry.message.get());
    ctx.events_processed++;
  }
}

void SimulationEngine::schedule_event(Event *event, Tick timestamp,
                                      std::unique_ptr<Message> message) {
  Component *target = event->target();
  assert(target != nullptr && "schedule_event: event has no target component");
  PartitionID pid = target->partition_id();
  assert(pid < contexts_.size() && "schedule_event: target partition ID out of range");
  contexts_[pid]->event_queue.push(EventQueueEntry{timestamp, 0, event, std::move(message)});
}

void SimulationEngine::send_cross_partition(PartitionID src_partition, PartitionID dst_partition,
                                            Event *event, Tick timestamp,
                                            std::unique_ptr<Message> message) {
  assert(src_partition < contexts_.size());
  assert(dst_partition < contexts_.size());

  PartitionContext &src_ctx = *contexts_[src_partition];
  PartitionContext &dst_ctx = *contexts_[dst_partition];

  // Deposit into the destination partition's incoming queue for this source.
  dst_ctx.incoming[src_partition]->push(EventQueueEntry{timestamp, 0, event, std::move(message)},
                                        src_ctx.vclock);

  // Update the source partition's min_outgoing.
  if (timestamp < src_ctx.local_min_outgoing)
    src_ctx.local_min_outgoing = timestamp;

  // Wake the destination partition if it's blocking.
  dst_ctx.advance_pending.store(true, std::memory_order_release);
  dst_ctx.advance_cv.notify_one();
}

void SimulationEngine::send_timestamp_advances(PartitionContext &ctx, const Partition &part) {
  // Use the partition's earliest pending activity as the base for advances.
  // If this partition is idle (TICK_MAX), the advance declares a very high
  // lower bound, allowing neighbors to advance freely past it.
  Tick local_next = std::min(ctx.event_queue.next_event_time(), ctx.local_min_outgoing);
  Tick base_time = std::max(ctx.local_lbts, local_next);

  for (auto *link : part.boundary_links) {
    if (link->src()->owner()->partition_id() != ctx.partition_id)
      continue;

    Tick advance_time;
    if (base_time == TICK_MAX || base_time > TICK_MAX - link->latency())
      advance_time = TICK_MAX;
    else
      advance_time = base_time + link->latency();
    PortID src_pid = link->src()->port_id();
    PortID dst_pid = link->dst()->port_id();

    auto msg = std::make_unique<TimestampAdvanceMessage>(advance_time, src_pid, dst_pid);
    PartitionID dst_part = link->dst()->owner()->partition_id();
    PartitionContext &dst_ctx = *contexts_[dst_part];
    dst_ctx.incoming[ctx.partition_id]->push(
        EventQueueEntry{advance_time, 0, &timestamp_advance_event_, std::move(msg)}, ctx.vclock);
    ctx.timestamp_advances_sent++;

    // Wake the destination partition.
    dst_ctx.advance_pending.store(true, std::memory_order_release);
    dst_ctx.advance_cv.notify_one();
  }
}

void SimulationEngine::register_as_primary() {
  primary_count_.fetch_add(1, std::memory_order_release);
}

void SimulationEngine::primary_ok_to_end() {
  primary_ok_count_.fetch_add(1, std::memory_order_release);
}

void SimulationEngine::primary_do_not_end() {
  uint32_t prev = primary_ok_count_.load(std::memory_order_acquire);
  while (prev > 0) {
    if (primary_ok_count_.compare_exchange_weak(prev, prev - 1, std::memory_order_release,
                                                std::memory_order_acquire))
      return;
    // compare_exchange_weak reloads prev on failure; the loop terminates when
    // either the CAS succeeds or prev reaches 0.
  }
  // prev == 0: no primary had signaled OK, so there is nothing to undo.
  // This is not an error — the caller may re-arm before any primary_ok_to_end().
}

bool SimulationEngine::all_primaries_done() const {
  uint32_t total = primary_count_.load(std::memory_order_acquire);
  if (total == 0)
    return false;
  return primary_ok_count_.load(std::memory_order_acquire) >= total;
}

bool SimulationEngine::check_termination(Tick lbts) {
  if (all_primaries_done()) {
    set_exit(ExitReason::COMPLETED, lbts, "all primaries completed");
    done_.store(true, std::memory_order_release);
    return true;
  }
  if (lbts == TICK_MAX) {
    // If primaries are registered but haven't all signaled OK, they are
    // promising future work (via async events). Don't terminate - the
    // worker loop will block on advance_cv and wake when async events arrive.
    if (primary_count_.load(std::memory_order_acquire) > 0)
      return false;
    set_exit(ExitReason::COMPLETED, current_time_.load(std::memory_order_acquire),
             "all partitions quiescent");
    done_.store(true, std::memory_order_release);
    return true;
  }
  return false;
}

void SimulationEngine::request_exit(std::string reason, int code) {
  Tick tick = current_time_.load(std::memory_order_acquire);
  {
    std::lock_guard<std::mutex> lock(exit_mutex_);
    if (!done_.load(std::memory_order_acquire))
      exit_event_ = ExitStatus(ExitReason::EXIT_REQUEST, tick, std::move(reason), code);
  }
  done_.store(true, std::memory_order_release);
  notify_all_partitions();
}

void SimulationEngine::schedule_event_async(Event *event, Tick timestamp,
                                            std::unique_ptr<Message> message) {
  Component *target = event->target();
  assert(target != nullptr && "schedule_event_async: event has no target component");
  PartitionID pid = target->partition_id();
  assert(pid < async_queues_.size() && "schedule_event_async: target partition ID out of range");
  auto &aq = *async_queues_[pid];
  {
    std::lock_guard<std::mutex> lock(aq.mutex);
    aq.events.push_back(EventQueueEntry{timestamp, 0, event, std::move(message)});
  }

  // Wake the target partition so idle workers pick up the new event.
  if (pid < contexts_.size()) {
    PartitionContext &pctx = *contexts_[pid];
    pctx.advance_pending.store(true, std::memory_order_release);
    pctx.advance_cv.notify_one();
  }
}

void SimulationEngine::schedule_event_now(Event *event, std::unique_ptr<Message> message) {
  Tick timestamp = wall_clock_sync_.enabled() ? wall_clock_sync_.sim_tick_now()
                                              : current_time_.load(std::memory_order_acquire);
  schedule_event_async(event, timestamp, std::move(message));
}

void SimulationEngine::drain_async_events() {
  for (uint32_t i = 0; i < async_queues_.size(); ++i) {
    auto &aq = *async_queues_[i];
    std::lock_guard<std::mutex> lock(aq.mutex);
    for (auto &e : aq.events)
      contexts_[i]->event_queue.push(std::move(e));
    aq.events.clear();
  }
}

void SimulationEngine::set_exit(ExitReason reason, Tick tick, std::string message, int code) {
  std::lock_guard<std::mutex> lock(exit_mutex_);
  // First writer wins - don't overwrite if an exit reason was already set.
  if (exit_event_.message.empty())
    exit_event_ = ExitStatus(reason, tick, std::move(message), code);
}

void SimulationEngine::notify_all_partitions() {
  for (auto &ctx : contexts_) {
    ctx->advance_pending.store(true, std::memory_order_release);
    ctx->advance_cv.notify_all();
  }
}

void SimulationEngine::initialize_components() {
  for (auto &part : topology_.partitions()) {
    for (auto *comp : part.components)
      comp->initialize();
  }
}

void SimulationEngine::startup_components() {
  for (auto &part : topology_.partitions()) {
    for (auto *comp : part.components)
      comp->startup();
  }
}

void SimulationEngine::finalize_components() {
  for (auto &part : topology_.partitions()) {
    for (auto *comp : part.components)
      comp->finalize();
  }
}

} // namespace simdojo
