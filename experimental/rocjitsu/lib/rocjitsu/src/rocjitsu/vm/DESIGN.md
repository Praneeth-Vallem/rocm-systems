# AMDGPU Virtual Machine Design

An AMDGPU virtual machine built on the simdojo simulation framework. Models
a complete SoC hierarchy, from command processors and shader engines down to
compute units, wavefronts, and register files. Runs within simdojo's unified
PDES epoch loop, supporting both interactive stepping and continuous
execution.

## File Overview

| File | Purpose |
|------|---------|
| `virtual_machine.h/cpp` | VirtualMachine: top-level VM, owns SoC and Driver |
| `driver.h/cpp` | Driver: KMD interface, lifecycle state machine |
| `soc.h/cpp` | SoC: XCDs, shared GPU memory, shader engine collection |
| `amdgpu/iod.h/cpp` | IOD: I/O die with memory-side cache and HBM controllers |
| `amdgpu/memory_side_cache.h/cpp` | Memory-side cache between L2 and HBM |
| `amdgpu/hbm_controller.h` | HBM memory controller wrapping GpuMemory |
| `rj_vm.cpp` | C API: create, step, run, checkpoint |
| `amdgpu/command_processor.h/cpp` | CP: dispatch packets, doorbell loop |
| `amdgpu/compute_unit.h/cpp` | CU: wavefront slots, register files, execution |
| `amdgpu/shader_engine.h/cpp` | SE: container of compute units |
| `amdgpu/xcd.h/cpp` | XCD: CP + shader engines |
| `amdgpu/wavefront.h/cpp` | Wavefront: ISA-specific thread state |
| `amdgpu/gpu_memory.h` | GpuMemory: flat address space wrapper |

---

## Component Hierarchy

```
SimulationEngine                       (simdojo - owns topology)
└── Topology
    └── VirtualMachine ("vm")          (CompositeComponent - topology root)
        ├── Driver driver_             (value member, not a child component)
        └── SoC ("gpu_soc")           (CompositeComponent)
            ├── GpuMemory ("memory")   (shared across all XCDs)
            ├── Iod[0..I] ("iod0"..)  (CompositeComponent - memory-side cache + HBM controllers)
            └── Xcd[0..N] ("xcd0"..)  (CompositeComponent)
                ├── CommandProcessor ("cp") (Component - event-driven dispatch)
                └── ShaderEngine[0..M]     (CompositeComponent)
                    └── ComputeUnit[0..K]  (CompositeComponent - register files + wavefront slots)
```

The VirtualMachine is a `simdojo::CompositeComponent` set as the topology
root. The simulation infrastructure (engine, topology, partitioning) is
managed by `SimulationEngine`; the VM represents the hardware being modeled.

---

## Driver

The Driver models the kernel-mode driver (KMD) interface presented to a
user-mode driver (e.g., rocr). It is owned as a value member of
VirtualMachine (`Driver driver_{*this}`) - the driver is part of the
hardware, not external software.

The Driver is a thin bridge between the application and the simulation
engine. It exposes two methods:

- `submit(DispatchPacket, uint32_t xcd_idx)` - enqueues a dispatch packet
  on the target XCD's command processor doorbell queue.
- `close()` - shuts down the simulation by calling
  `engine()->request_exit()`.

---

## Command Processor

The CP reads dispatch packets and distributes wavefronts across registered
compute units in round-robin order.

### Event-Driven Dispatch

The CP is event-driven. During `startup()`, it schedules a doorbell event
for any pre-loaded packets. Each doorbell event triggers `step()`, which
processes one dispatch packet: for each workgroup, it calls `dispatch_wf()`
on the next CU in round-robin order, then calls `activate()` on that CU.

- `enqueue(packet)` - pre-loads packets without triggering dispatch (used
  by config loader and tests)
- `submit(packet)` - thread-safe; appends packet and schedules an async
  doorbell event via `schedule_event_now()`
- `activate()` - registers the CP as a primary component and begins
  processing doorbell events

Each CU runs its dispatched wavefronts independently (in functional mode,
a single work event tight-loops `step()` until all wavefronts halt). When
a CU becomes idle, it fires its `on_idle` callback. When all CUs are idle
and no packets remain, the CP signals completion via
`engine()->primary_ok_to_end()`.

---

## Dispatch Packet Flow

```
Config loader / Driver::submit()
  └── cp->enqueue(packet)  or  cp->submit(packet) [+ async doorbell event]
        └── cp->step() processes one packet:
              for each workgroup:
                cu = next CU (round-robin)
                cu->dispatch_wf(wg_id, pc, sgprs, vgprs)
                  └── find idle slot, allocate SGPR/VGPR blocks
                      initialize wavefront state (pc, wg_id)
                cu->activate()
                  └── schedule work event (functional) or resume clock (clocked)
```

A `DispatchPacket` specifies:
- `kernel_entry_pc` - byte address of kernel code in GPU memory
- `workgroup_count` - number of workgroups to launch
- `wfs_per_workgroup` - wavefronts per workgroup
- `sgprs_per_wf` / `vgprs_per_wf` - register requirements (from code object)

Wavefronts are distributed round-robin across CUs within the XCD. Each CU
allocates a contiguous block in its physical SGPR and VGPR files for the
wavefront.

---

## Execution Modes

### Interactive Stepping (`rj_vm_step`)

Synchronous, single-threaded. The caller drives execution one tick at a
time:

```
rj_vm_step(vm, &active)
  └── engine.step()         process all events at next timestamp
        └── CP doorbell event fires → cp->step() drains dispatch queue
            CU tick events fire → execute one instruction per wavefront
```

Returns `active=1` while any wavefront is still executing. The engine is
built during `rj_vm_create()`, not on first step.

### Continuous Execution (`rj_vm_run`)

The simulation thread runs `engine.run()`, which drains the event queue
continuously. The main thread injects work via `schedule_event_async()`:

```
Main thread                          Simulation thread
───────────                          ─────────────────
                                     engine.run()
                                       └── epoch loop processes events
                                             │
driver().submit(packet)  ──────────►  async event → CP doorbell fires
  │                                          │
  │                                     cp->step() dispatches wavefronts
  │                                     CU tick events execute instructions
  │                                          │
driver().close()         ──────────►  done_ set, workers stop
  │                                        (close() calls engine()->request_exit())
  │
sim_thread.join()  ◄─────────────────────    engine finalizes components
  │
engine.shutdown()
```

In single-threaded mode, the engine drains events in timestamp order
without LBTS synchronization. The CP schedules doorbell events during
`startup()` for pre-loaded packets and via `schedule_event_async()` for
external submissions.

---

## Simulation Integration

The VM plugs into simdojo's `SimulationEngine` directly. The C API layer
(`rj_vm.cpp`) owns the engine and wires the VM into the topology:

1. **Construction** - Config loader creates a `VirtualMachine` and returns
   it with engine configuration. The C API sets the VM as the topology root
   via `engine.topology().set_root(std::move(vm))`.

2. **`build()`** - Partitions topology, sets up the engine.

3. **`run()`** - The engine initializes and starts all components, then
   drains events continuously. In single-threaded mode, events are processed
   in timestamp order until the queue empties and termination conditions are
   met. In multi-threaded mode, workers run LBTS-synchronized epoch loops.

4. **`step()`** - Processes all events at the next timestamp (one tick step).
   Returns whether the simulation can continue.

5. **`shutdown()`** - Calls `finalize()` on all components, tears down engine.

After `run()` returns, `last_exit()` provides a `ExitStatus` with
the termination reason (`COMPLETED`, `EXIT_REQUEST`, `INTERRUPTED`), the
simulation tick, and a human-readable message. Components can call
`engine()->request_exit(reason, code)` to stop the simulation.

---

## C API (`rj_vm.h`)

The C API wraps the VM behind an opaque `rj_vm_t` handle:

`rj_vm_t` is reference-counted (extends `RefCounted`). Use `rj_vm_retain`
and `rj_vm_release` to manage shared ownership; `rj_vm_destroy` is a
convenience wrapper that releases the last reference and tears down the VM.

| Function | Description |
|----------|-------------|
| `rj_vm_create()` | Load config from JSON file, build VM and engine |
| `rj_vm_create_from_string()` | Load config from JSON string |
| `rj_vm_retain()` | Increment reference count |
| `rj_vm_release()` | Decrement reference count; destroys when it reaches zero |
| `rj_vm_destroy()` | Tear down VM |
| `rj_vm_step()` | One interactive step |
| `rj_vm_run()` | Run to completion via driver open/close |
| `rj_vm_save_checkpoint()` | Serialize VM state to FlatBuffer |
| `rj_vm_restore_checkpoint()` | Restore VM from checkpoint file |

Internal C++ code (tests, GUI) accesses the `VirtualMachine` directly via the
config loader (`config::load_config()` / `config::load_config_from_string()`).
The `rj_vm.h` header is a pure C API with opaque handles.
