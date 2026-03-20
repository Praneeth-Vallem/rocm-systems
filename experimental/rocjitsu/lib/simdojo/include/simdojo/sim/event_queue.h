// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file event_queue.h
/// @brief Event descriptors, priority queues, and cross-partition queues for the simulation engine.

#ifndef SIMDOJO_SIM_EVENT_QUEUE_H_
#define SIMDOJO_SIM_EVENT_QUEUE_H_

#include "simdojo/sim/message.h"
#include "simdojo/sim/sim_types.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

namespace simdojo {

/// @brief Type of simulation event.
///
/// @details Numeric order determines priority at the same tick: smaller value = higher
/// priority. SIM_EXIT is last so that all real events at max_ticks fire first.
enum class EventType : uint8_t {
  BARRIER_SYNC,      ///< Synchronization barrier pseudo-event.
  TIMESTAMP_ADVANCE, ///< LBTS timestamp advance (declares a lower bound on future messages).
  TIMER_CALLBACK,    ///< A scheduled timer/tick callback.
  MESSAGE_ARRIVAL,   ///< A message arrived at a port.
  SIM_EXIT,          ///< Simulation exit sentinel (lowest priority).
};

class Component;

/// @brief Callback type for event handling.
using EventHandler = std::function<void(Tick, Message *)>;

/// @brief A reusable simulation event descriptor.
///
/// @details Events are long-lived objects owned by Ports, Components,
/// or the SimulationEngine. They hold a target component, event type, and
/// handler callback. Per-firing state (timestamp, message payload) is stored
/// in EventQueueEntry, not in the Event itself. The same Event object can
/// appear in the event queue multiple times at different timestamps.
class Event {
public:
  /// @brief Construct an event.
  /// @param target Component that will process this event.
  /// @param type Category of event (determines processing priority).
  /// @param handler Optional callback invoked when the event executes.
  Event(Component *target, EventType type, EventHandler handler = nullptr)
      : target_(target), type_(type), handler_(std::move(handler)) {}
  virtual ~Event() = default;

  /// @brief Execute the event's handler with the given firing context.
  /// @param timestamp The simulation tick at which this firing occurs.
  /// @param message Optional message payload for this firing.
  virtual void execute(Tick timestamp, Message *message) {
    assert(handler_ && "execute() called on event with no handler");
    handler_(timestamp, message);
  }

  /// @brief Set the event handler callback.
  /// @param h New handler to assign.
  void set_handler(EventHandler h) { handler_ = std::move(h); }

  /// @brief Check whether a handler is registered.
  /// @retval true A handler is assigned.
  /// @retval false No handler is assigned.
  bool has_handler() const { return handler_ != nullptr; }

  /// @brief Return the target component.
  /// @returns Pointer to the component that processes this event.
  Component *target() const { return target_; }

  /// @brief Return the event type.
  /// @returns The EventType category.
  EventType type() const { return type_; }

private:
  Component *const target_; ///< Component that processes this event.
  EventType type_;          ///< Event category / priority class.
  EventHandler handler_;    ///< Callback invoked on execute().
};

/// @brief A single entry in the event priority queue.
///
/// @details Captures the per-firing state: timestamp, message payload,
/// and a pointer to the reusable Event descriptor.
struct EventQueueEntry {
  Tick timestamp = 0;               ///< Simulation tick when this entry fires.
  uint64_t sequence = 0;            ///< Tie-breaking sequence number.
  Event *event = nullptr;           ///< Reusable event descriptor.
  std::unique_ptr<Message> message; ///< Optional message payload for this firing.

  /// @brief Min-heap comparator: smallest timestamp first, then event type
  /// priority, then sequence number.
  struct Greater {
    bool operator()(const EventQueueEntry &a, const EventQueueEntry &b) const {
      if (a.timestamp != b.timestamp)
        return a.timestamp > b.timestamp;
      if (a.event->type() != b.event->type())
        return a.event->type() > b.event->type();
      return a.sequence > b.sequence;
    }
  };
};

/// @brief Per-partition priority queue of simulation events.
///
/// @details Each partition (and therefore each worker thread) has exactly one
/// EventQueue. Entries are ordered by timestamp via a min-heap backed by
/// a flat vector. The queue stores EventQueueEntry values; Events are externally
/// owned and reusable across multiple firings.
class EventQueue {
public:
  EventQueue() = default;

  /// @brief Enqueue a heap entry. Assigns a sequence number for tie-breaking.
  /// @param entry The entry to enqueue (ownership of message transferred).
  void push(EventQueueEntry entry) {
    entry.sequence = next_sequence_++;
    entries_.push_back(std::move(entry));
    std::push_heap(entries_.begin(), entries_.end(), EventQueueEntry::Greater{});
  }

  /// @brief Dequeue and return the earliest entry.
  /// @returns The EventQueueEntry with the smallest timestamp.
  EventQueueEntry pop() {
    assert(!entries_.empty());
    std::pop_heap(entries_.begin(), entries_.end(), EventQueueEntry::Greater{});
    auto entry = std::move(entries_.back());
    entries_.pop_back();
    return entry;
  }

  /// @brief Peek at the earliest entry's timestamp without removing it.
  /// @returns Timestamp of the next entry, or TICK_MAX if empty.
  Tick next_event_time() const { return entries_.empty() ? TICK_MAX : entries_.front().timestamp; }

  /// @brief Check whether the queue is empty.
  /// @retval true No entries are enqueued.
  /// @retval false At least one entry is enqueued.
  bool empty() const { return entries_.empty(); }

  /// @brief Return the number of enqueued entries.
  /// @returns Current queue size.
  size_t size() const { return entries_.size(); }

  /// @brief Return the last processed tick (set by the engine).
  /// @returns The current tick value.
  Tick current_tick() const { return current_tick_; }

  /// @brief Update the current tick (called by the engine after processing).
  /// @param t New current tick value.
  void set_current_tick(Tick t) { current_tick_ = t; }

private:
  std::vector<EventQueueEntry> entries_; ///< Min-heap of entries by timestamp.
  uint64_t next_sequence_ = 0;           ///< Monotonic counter for deterministic tie-breaking.
  Tick current_tick_ = 0;                ///< Last processed simulation tick.
};

/// @brief Concurrent queue for cross-partition event delivery.
///
/// @details Each destination partition has one CrossPartitionQueue per source
/// partition. With causal LBTS, push and drain can be concurrent, so a mutex
/// protects the shared state. Drain uses a swap pattern to minimize time under
/// the lock.
///
/// The queue also carries the sender's vector clock as a side-channel: the
/// sender updates it on each push, and the receiver merges it during drain.
class CrossPartitionQueue {
public:
  CrossPartitionQueue() = default;

  // Non-copyable, non-movable.
  CrossPartitionQueue(const CrossPartitionQueue &) = delete;
  CrossPartitionQueue &operator=(const CrossPartitionQueue &) = delete;

  /// @brief Push an entry and update the sender's vector clock snapshot.
  /// @param entry The entry to push (ownership of message transferred).
  /// @param sender_vclock The sender partition's current vector clock.
  void push(EventQueueEntry entry, const std::vector<Tick> &sender_vclock) {
    std::lock_guard<std::mutex> guard(mutex_);
    entries_.push_back(std::move(entry));
    sender_vclock_ = sender_vclock;
  }

  /// @brief Push an entry without updating the vector clock.
  ///
  /// Used by code paths that don't participate in causal LBTS (e.g.,
  /// single-threaded mode, or legacy callers).
  /// @param entry The entry to push (ownership of message transferred).
  void push(EventQueueEntry entry) {
    std::lock_guard<std::mutex> guard(mutex_);
    entries_.push_back(std::move(entry));
  }

  /// @brief Drain all entries into a local EventQueue, merging vector clocks.
  ///
  /// Swaps the internal buffer under the lock, then processes entries and
  /// merges the sender's vector clock outside the lock.
  /// @param local_queue The partition's thread-local EventQueue.
  /// @param receiver_vclock The receiver's vector clock (merged in-place).
  /// @returns Number of entries drained.
  size_t drain_into(EventQueue &local_queue, std::vector<Tick> &receiver_vclock) {
    std::vector<EventQueueEntry> batch;
    std::vector<Tick> vclock_snapshot;
    {
      std::lock_guard<std::mutex> guard(mutex_);
      batch.swap(entries_);
      vclock_snapshot.swap(sender_vclock_);
    }
    if (!vclock_snapshot.empty()) {
      assert(vclock_snapshot.size() <= receiver_vclock.size());
      for (uint32_t i = 0; i < vclock_snapshot.size(); ++i)
        receiver_vclock[i] = std::max(receiver_vclock[i], vclock_snapshot[i]);
    }
    for (auto &e : batch)
      local_queue.push(std::move(e));
    return batch.size();
  }

  /// @brief Drain all entries without vector clock merging (single-threaded).
  /// @param local_queue The partition's thread-local EventQueue.
  /// @returns Number of entries drained.
  size_t drain_into(EventQueue &local_queue) {
    std::lock_guard<std::mutex> guard(mutex_);
    size_t count = entries_.size();
    for (auto &e : entries_)
      local_queue.push(std::move(e));
    entries_.clear();
    return count;
  }

  /// @brief Check whether the queue is empty (thread-safe).
  /// @retval true No entries are pending.
  /// @retval false At least one entry is pending.
  bool empty() const {
    std::lock_guard<std::mutex> guard(mutex_);
    return entries_.empty();
  }

  /// @brief Return the number of pending entries (thread-safe).
  /// @returns Current queue size.
  size_t size() const {
    std::lock_guard<std::mutex> guard(mutex_);
    return entries_.size();
  }

private:
  mutable std::mutex mutex_;             ///< Protects entries_ and sender_vclock_.
  std::vector<EventQueueEntry> entries_; ///< Buffered entries awaiting drain.
  std::vector<Tick> sender_vclock_;      ///< Sender's latest vector clock snapshot.
};

} // namespace simdojo

#endif // SIMDOJO_SIM_EVENT_QUEUE_H_
