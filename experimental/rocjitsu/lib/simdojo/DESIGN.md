# Simulation Framework Design

A simulation framework for modeling hardware systems as hierarchical compound
graphs [1]. Runs a conservative Chandy-Misra-Bryant Parallel Discrete Event
Simulation (PDES) [2][3] with Lower Bound on Time Stamp (LBTS)
synchronization. Untimed (functional) components participate in the same
event loop via the `Functional<Base>` CRTP mixin.

## File Overview

| File | Purpose |
|------|---------|
| `sim_types.h` | Tick, ComponentID, LinkID, PortID, PartitionID, ExitReason, ExitStatus |
| `event_queue.h` | Event, EventQueueEntry, EventQueue, CrossPartitionQueue |
| `message.h` | MessageHeader, Message, TimestampAdvanceMessage, MessageQueue |
| `component.h` | Node, Component, CompositeComponent, Link, Port, QueuedLink |
| `component.cpp` | Implementation for Component, CompositeComponent, Link |
| `clock_domain.h` | ClockDomain (frequency, period, phase offset) |
| `clocked.h` | Clocked\<Base\> Curiously Recurring Template Pattern (CRTP) mixin for clock-driven components |
| `topology.h` | Topology, Partition, AdjacencyGraph, Partitioner |
| `topology.cpp` | Topology wiring, graph construction, multilevel Fiduccia-Mattheyses (FM) partitioning |
| `functional.h` | Functional\<Base\> CRTP mixin for untimed components |
| `exec_mode.h` | ExecMode enum (FUNCTIONAL, CLOCKED) and ExecBase\<Mode, Base\> alias |
| `wall_clock_sync.h` | WallClockSync - wall-clock-coupled simulation time synchronization |
| `simulation.h` | PartitionContext, SimulationEngine |
| `simulation.cpp` | Engine run loop, event dispatch, LBTS computation (multi-threaded) |
| `components/memory_interface.h` | MemoryInterface - abstract byte-addressed memory controller |
| `components/cache.h` | Cache\<NumSets, Assoc, LineSizeBits\> - set-associative cache template |
| `components/sparse_memory.h` | SparseMemory - page-allocated flat address space |
| `components/register_file.h` | RegisterFile - typed register storage with read/write interface |
| `components/vector_reg.h` | VectorReg\<N, T\> - SIMD-width vector register type |
| *(uses `util/debug_print.h`)* | Conditional debug output (`util::debug` namespace) |

---

## Simulation Time

All timestamps, periods, latencies, and phase offsets are expressed in
`Tick` (`uint64_t`). The tick resolution is fixed globally at 1 picosecond
(`TICKS_PER_SECOND = 1'000'000'000'000`). A 1 GHz clock has a period of 1000
ticks (1 ns); a 2 GHz clock has 500 ticks.

Frequency is a separate dimension measured in Hz (`uint64_t`), not ticks.
The two are mixed in expressions like `TICKS_PER_SECOND / frequency_hz` to
produce a `Tick` period, which is dimensionally correct.

---

## Compound Graph

The simulation topology is a **compound graph** [4] - a graph that combines
two kinds of relationships:

- **Inclusion edges** (parent-child): form the component tree, modeling
  physical containment (e.g., a board contains processors, which contain
  functional units)
- **Adjacency edges** (links): model communication channels between
  components, each with a propagation latency

This mirrors how hardware is structured: a hierarchy of blocks connected by
buses, crossbars, and point-to-point links.

### Components

**Component** is a leaf simulation entity that processes events. It owns
ports and interacts with the simulation engine. Event handlers are
registered directly on ports via `Port::set_handler()`. Key virtual
methods:

- `initialize()` / `startup()` / `finalize()` - lifecycle hooks called by
  the engine
- `step()` - execute one logical step; return true to continue
- `run()` - default loops `while (step()) {}`; override for custom
  lifecycle (e.g., blocking on a condition variable)
- `is_composite()` - returns false for leaves, true for composites

**CompositeComponent** extends Component with a child list. Critically, a
composite can also handle events itself - it is not just a passive
container. This lets you model a block that both contains sub-blocks
and directly handles some events (e.g., routing, arbitration) without
introducing artificial wrapper components.

`CompositeComponent` overrides `run()` and `step()` with default
delegation behavior: `run()` calls `run()` on each child, and `step()`
calls `step()` on each child, returning true if any child is still active.
Subclasses (e.g., VirtualMachine) override these for custom lifecycle
logic. This means a `CompositeComponent` can serve as the topology root
directly - the engine calls `root->run()` and the composite delegates to
its children automatically.

`collect_components()` flattens the entire subtree including composites,
so the engine initializes and dispatches to all components uniformly.

### Ports and Links

**Ports** are named connection points on a component. Each port owns a
reusable `Event` for message arrivals. Sending is done via `Port::send()`;
receiving is handled by calling `Port::set_handler()` to register a
callback that the engine invokes when a message arrives. Each port has a
direction (`PortDirection::IN` or `PortDirection::OUT`) and an optional
protocol tag (`PortProtocol`). `send()` is only valid on `OUT` ports.

**Links** are **unidirectional** connections between two ports with a
propagation latency. Links have an execution mode: `CLOCKED` (default)
routes messages through the engine event queue with latency; `FUNCTIONAL`
invokes the destination handler synchronously with zero overhead. Bidirectional communication requires two links, one
per direction. This matches hardware reality (request and response paths
are separate) and keeps timing, flow control, and routing simple.

```
Component0           Component1
  P0 ----link A----> P1       (request path, 10 tick latency)
  P3 <---link B----- P2       (response path, 5 tick latency)
```

`Topology::add_bidirectional_link(a_out, b_in, b_out, a_in, fwd, rev)`
is a convenience that creates two unidirectional links in one call. It
takes four ports (two per component: one OUT, one IN) and two latencies.

**QueuedLink** is a variant that buffers messages in a bounded priority
queue instead of routing them through the engine immediately. The
receiving component explicitly drains messages when ready - useful for
pull-based consumption patterns like a clocked component that processes
its input queue each cycle.

---

## Clock Domains

A `ClockDomain` defines a shared clock source with a frequency, period, and
phase offset. Derived domains can be created with a frequency divider,
modeling clock hierarchies (e.g., a 1 GHz core clock deriving a 500 MHz
memory clock).

The `Clocked<Base>` CRTP mixin adds clock-driven behavior to any component
type. Subclasses override `advance(Tick now)`, which is called on each
rising edge. Return true to continue clocking, false to halt. The mixin
handles self-scheduling automatically.

```cpp
class Pipeline : public Clocked<Component> {
public:
  Pipeline(const ClockDomain &clk) : Clocked("pipeline", clk) {}
  bool advance(Tick now) override { /* process one cycle */ return true; }
};
```

`Clocked` is templated on the base class, so it works with both `Component`
and `CompositeComponent`:

```cpp
class CPU : public Clocked<CompositeComponent> { ... };
```

### Functional Components

The `Functional<Base>` CRTP mixin is the untimed counterpart to `Clocked`.
Subclasses override `advance(Tick now)`, which is called repeatedly with a
monotonic tick counter (synthetic timestamp, not wall time). Return true to
continue advancing, false to halt. Used for components that model behavior
without cycle accuracy.

```cpp
class Memory : public Functional<Component> {
public:
  Memory() : Functional("memory") {}
  bool advance(Tick now) override { /* process one request */ return true; }
};
```

Like `Clocked`, it is templated on the base class and self-schedules via the
engine's event queue.

### Execution Mode Selection

The `ExecMode` enum and `ExecBase<Mode, Base>` alias (in `exec_mode.h`)
allow generic components to be templated on their execution mode:

```cpp
template <ExecMode Mode>
class MyComponent : public ExecBase<Mode, Component> {
public:
  bool advance(Tick now) override { /* same logic for both modes */ }
};
```

`ExecBase<FUNCTIONAL, Component>` resolves to `Functional<Component>`;
`ExecBase<CLOCKED, Component>` resolves to `Clocked<Component>`. This
enables a single component implementation that works in either mode,
selected at compile time.

---

## Component Library

The `components/` directory provides reusable building blocks for hardware
models. These are architecture-agnostic data structures and interfaces.

**MemoryInterface** (`memory_interface.h`) is the abstract base for all
byte-addressed memory controllers. Controllers (cache controllers, memory
controllers, scratchpad memories) implement this interface. Backing stores
(`SparseMemory`, `Cache<>`) are data structures that do not implement it.

**Cache** (`cache.h`) is a set-associative cache template parameterized on
number of sets, associativity, and line size. Provides `lookup()`, `fill()`,
and `flush()`. Used as a backing store inside cache controllers.

**SparseMemory** (`sparse_memory.h`) is a page-allocated flat address space
that allocates pages on first access. Used as backing storage for memory
controllers.

**RegisterFile** (`register_file.h`) provides typed register storage with
indexed read/write access. **VectorReg** (`vector_reg.h`) models a
SIMD-width vector register with per-element access.

---

## Messages and Events

### Messages

`Message` is the base class for data sent over links. Subclass it to add
payload for your model. Every message carries a `MessageHeader` with
timestamps, port IDs, and a sequence number.

`TimestampAdvanceMessage` is used by the LBTS protocol [2][3] - it carries
no payload and declares that no real message with an earlier timestamp will
be sent on this link, allowing the receiver to safely advance its LBTS.
Known in the literature as a "null message" [Chandy-Misra 1979, Bryant
1977].

### Events

Events are the scheduling primitive. There are five types, ordered by
priority (highest priority first):

1. `BARRIER_SYNC` - synchronization barrier pseudo-event
2. `TIMESTAMP_ADVANCE` - LBTS timestamp advance (null message)
3. `TIMER_CALLBACK` - scheduled timer/tick callback (used by `Clocked`)
4. `MESSAGE_ARRIVAL` - a message arrived at a port
5. `SIM_EXIT` - simulation exit sentinel (lowest priority, ensures all real
   events at `max_ticks` fire before the exit)

Events with the same timestamp are ordered by type priority, then by
sequence number for deterministic tie-breaking.

### Zero-Allocation Event Model

`Event` is a long-lived, reusable descriptor holding a target component,
event type, and handler callback. Per-firing state (timestamp, message
payload) is stored in `EventQueueEntry`, not in the Event itself. The same
Event object can appear in the event queue multiple times at different
timestamps - no allocation is needed per scheduling.

- **Ports** own a reusable `Event` of type `MESSAGE_ARRIVAL`. Register a
  handler via `Port::set_handler()`.
- **Clocked** owns a reusable `Event` of type `TIMER_CALLBACK` that
  re-enqueues itself on each clock edge.
- **Timestamp advances** use the engine's shared `timestamp_advance_event_`
  (a handler-less Event of type `TIMESTAMP_ADVANCE`). They contribute to
  LBTS computation via their presence in the heap without executing a handler.

### Event Queues

Each partition has a thread-local `EventQueue` (a min-heap by timestamp).
Cross-partition events are staged in a `CrossPartitionQueue` - one per
(source, destination) partition pair. The queue is mutex-protected (push
from the sender partition, drain from the receiver partition). Each worker
drains its own incoming queues at the start of each epoch.

---

## Topology and Partitioning

The `Topology` owns the entire simulation graph: the root composite, all
links, clock domains, and partition assignments. It serves as the single
entry point for building and wiring a model.

### Building a Model

```cpp
SimulationEngine engine({.max_ticks = 10000, .num_threads = 4});

// Create clock domains.
auto *core_clk = engine.topology().add_clock_domain("core", 1'000'000'000);

// Build the component tree.
auto root = std::make_unique<CompositeComponent>("soc");
auto *proc = root->add_child(std::make_unique<CompositeComponent>("proc"));
auto *pipe = proc->add_child(std::make_unique<Pipeline>(*core_clk));
auto *mem = proc->add_child(std::make_unique<MemController>(*core_clk));

// Set root and wire links.
engine.topology().set_root(std::move(root));
engine.topology().add_link(pipe_req_port, mem_req_port, /*latency=*/10);
engine.topology().add_link(mem_resp_port, pipe_resp_port, /*latency=*/5);

engine.build();
auto exit = engine.run();
```

### Graph Partitioning

For parallel execution, the topology is partitioned into sub-graphs, one per
thread. The built-in partitioner uses a multilevel FM algorithm [6][7]:

1. **Coarsening** - heavy-edge matching contracts the graph level by level
2. **Initial bisection** - greedy assignment + FM refinement on the coarsest
   graph
3. **Uncoarsening** - project back, FM refine at each level
4. **k-way** - recursive bisection to reach the target partition count

The partitioner minimizes edge cut (cross-partition communication) while
balancing partition weights (configurable imbalance tolerance). No external
library dependencies.

---

## Simulation Engine

The `SimulationEngine` owns the simulation `Topology` and drives the full
lifecycle: build, run/step, and shutdown. Components participate by
scheduling events during `initialize()` or `startup()`:

- A `Clocked` component schedules clock-edge events (timing simulation).
- A `Functional` component self-schedules a timer callback and re-enqueues
  after each `advance()` (untimed simulation within the event loop).
- If no events are ever scheduled and no primaries are registered, the
  simulation terminates via quiescence.

### Single-Threaded Path

When `num_threads == 1`, the engine drains events directly from the priority
queue in timestamp order - no LBTS computation, no vector clocks, no epoch
structure. When the queue empties:

1. Update `current_time_` for external observers
2. Throttle via `WallClockSync` if sim is ahead of wall clock
3. Run service callbacks
4. Drain async event buffers
5. If new events appeared, continue processing
6. Check termination (quiescence or all primaries done)
7. If primaries are active, block on `advance_cv` until an async event or
   shutdown arrives (with a timed wait when wall-clock sync is enabled)
8. After wake, re-anchor the wall clock and drain async events

### Multi-Threaded Path

When `num_threads > 1`, N worker threads (one per partition) run a
conservative PDES loop with causal LBTS synchronization.

#### LBTS Epoch

Each worker independently follows this sequence:

1. Process all events with `timestamp <= local_lbts`
2. Send timestamp advances on outgoing cross-partition links
3. Drain incoming cross-partition queues, merging sender vector clocks
4. Drain async event buffers
5. Recompute `local_lbts` from the merged vector clock
6. If no progress possible, block until a neighbor sends an advance

Partition 0 periodically computes Global Virtual Time (GVT) as
`min(all local_lbts)`, runs service callbacks, and checks termination.

The LBTS is the local safe time - no event with a timestamp at or below
it will ever arrive from a neighbor, so all such events can be processed
without risk of causality violations.

#### Timestamp Advances

In a conservative protocol, a partition can stall if it doesn't know whether
a neighbor might send an event with a smaller timestamp. Timestamp advances
(called "null messages" in the literature [2][3]) break this deadlock: after
processing its events, each worker sends a timestamp advance on all outgoing
cross-partition links. The base timestamp is the tighter of the current
LBTS and the partition's earliest pending activity (`min(next_event_time,
local_min_outgoing)`), so the advance is `max(lbts, local_next) +
link_latency`. This enables **sync-skip**: if a partition has no events
until far in the future, its timestamp advances declare a higher lower
bound, allowing receivers to jump ahead without empty epoch cycling.
Automatically enabled when cross-partition links exist in a multi-threaded
configuration.

### Wall-Clock Synchronization

`WallClockSync` maintains a linear mapping between wall-clock (real) time
and simulation time, configured via `Config::wall_clock_ratio`:

- `0.0` (default) - disabled, simulation runs as fast as possible
- `1.0` - real-time (1 sim ns = 1 wall ns)
- `2.0` - sim runs at 2x real speed

Three capabilities:

1. **Timestamp translation** - `sim_tick_now()` converts wall time to a sim
   tick for `schedule_event_now()`, giving externally injected events
   (driver doorbells, host submissions) realistic timestamps.
2. **Throttling** - after draining events, the engine sleeps if sim time is
   ahead of the wall-clock projection.
3. **Idle advancement** - during quiescent periods, timed waits (10ms) keep
   the sim clock moving proportionally to elapsed wall time.

The mapping is anchored at simulation start and re-anchored after idle
periods to prevent time jumps. Thread-safe via an internal spinlock
(external threads call `sim_tick_now()` concurrently with the worker's
`reanchor()`).

### Service Callbacks

The engine supports registering callbacks that run periodically. In
single-threaded mode they run after each queue drain; in multi-threaded
mode they run from partition 0 during GVT computation. Use these for
periodic tasks like watchdog timers, statistics snapshots, or progress
reporting.

### Exit Events

The simulation reports its termination reason through `ExitStatus`, a
first-class value type.

- `ExitReason::COMPLETED` - normal completion (all primaries done or
  max_ticks reached)
- `ExitReason::EXIT_REQUEST` - a component called `request_exit()`
- `ExitReason::INTERRUPTED` - external interrupt (ctrl-C, watchdog)

`SimulationEngine::run()` returns an `ExitStatus`. The exit event is
also stored internally and accessible via `last_exit()` for callers that
use `step()` mode.

`request_exit()` is thread-safe: it sets the exit event, flips the `done_`
flag, and wakes all partition workers.

### Primary Component Protocol

The engine supports a consensus-based end-of-simulation protocol.
Work-producing components register as "primaries" during `startup()`
and signal when they are done:

```cpp
// In startup():
engine()->register_as_primary();

// When done producing work:
engine()->primary_ok_to_end();

// If new work arrives (re-arm):
engine()->primary_do_not_end();
```

The simulation ends gracefully when **all** registered primaries have
called `primary_ok_to_end()`. This is checked via a unified
`check_termination()` helper alongside `max_ticks` and quiescence
detection. The exit reason is `COMPLETED` with a descriptive message
distinguishing the trigger.

Primaries also suppress quiescence termination: if any primary is registered
but has not yet signaled OK, the engine will not exit when the event queue
is empty. Instead, the worker blocks on its condition variable until an
async event or shutdown request arrives. This allows models that start with
no events but expect external stimulus (e.g., a driver submitting work via
`schedule_event_now()`) to stay alive until all primaries are done.

This complements `request_exit()`: primaries are the cooperative
termination path (consensus); `request_exit()` is the unilateral force
stop (emergency brake).

### Async Event Insertion

`schedule_event_async()` allows any thread to inject events into a
partition's queue safely. Events are buffered in a per-partition
`AsyncQueue` (a mutex-protected vector) and drained into the partition's
event queue at the next safe point:

- In `worker_loop` (single-threaded): after the queue empties
- In `worker_loop` (multi-threaded): during the LBTS epoch
- In `step()`: at the start of each call, before processing

`schedule_event_now()` is the preferred API for external stimulus. It
translates the current wall time to a simulation tick (via
`WallClockSync::sim_tick_now()` when enabled, or uses `current_time_`
when disabled), then calls `schedule_event_async()`. This gives externally
injected events realistic timestamps instead of tick 0.

### Interactive Stepping (`step()`)

For GUI, debugger, or test harness use. Single-threaded only. On the first
call, initializes and starts all components. Each subsequent call drains
async events and processes all events at the next timestamp (one tick step).
Returns true if the simulation can continue. If the queue is empty but
primaries are registered, returns true so the caller can poll.

---

### Engine Lifecycle

```cpp
SimulationEngine engine({.max_ticks = 10000, .num_threads = 4});
engine.topology().set_root(std::move(my_model));
engine.topology().add_link(src_port, dst_port, latency);
engine.build();
auto exit = engine.run();
engine.shutdown();  // called automatically by destructor if needed
```

`shutdown()` is called automatically by the destructor if the engine is
still built.

## References

[1] R. M. Fujimoto, *Parallel and Distributed Simulation Systems*,
    Wiley-Interscience, 2000.

[2] K. M. Chandy and J. Misra, "Distributed Simulation: A Case Study in
    Design and Verification of Distributed Programs," *IEEE Transactions on
    Software Engineering*, vol. SE-5, no. 5, pp. 440-452, 1979.

[3] R. E. Bryant, "Simulation of Packet Communication Architecture Computer
    Systems," MIT-LCS-TR-188, Massachusetts Institute of Technology, 1977.

[4] G. Sugiyama and K. Misue, "Visualization of Structural Information:
    Automatic Drawing of Compound Digraphs," *IEEE Transactions on Systems,
    Man, and Cybernetics*, vol. 21, no. 4, pp. 876-892, 1991.

[5] F. Mattern, "Efficient Algorithms for Distributed Snapshots and Global
    Virtual Time Approximation," *Journal of Parallel and Distributed
    Computing*, vol. 18, no. 4, pp. 423-434, 1993.

[6] C. M. Fiduccia and R. M. Mattheyses, "A Linear-Time Heuristic for
    Improving Network Partitions," in *Proceedings of the 19th Design
    Automation Conference*, pp. 175-181, 1982.

[7] G. Karypis and V. Kumar, "A Fast and High Quality Multilevel Scheme for
    Partitioning Irregular Graphs," *SIAM Journal on Scientific Computing*,
    vol. 20, no. 1, pp. 359-392, 1998.
