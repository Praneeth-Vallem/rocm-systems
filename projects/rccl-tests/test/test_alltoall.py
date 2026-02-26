# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import itertools
import pytest
from test_runner import run_rccl_perf, run_rccl_mpi

EXECUTABLE = "alltoall_perf"


def test_alltoall_single(gpu_counts, byte_ranges, step_factor, datatypes, memory_types, request, subtests):
    timeout = request.config.getoption("--test-timeout")
    for datatype, memory_type in itertools.product(datatypes, memory_types):
        for min_bytes, max_bytes in byte_ranges:
            # 1 thread per GPU (-t <ngpus> -g 1)
            for ngpus in gpu_counts:
                with subtests.test(dtype=datatype, mem=memory_type,
                                   bytes=f"{min_bytes}-{max_bytes}", step=step_factor, ngpus=ngpus):
                    args = ["-t", ngpus, "-g", "1", "-b", min_bytes, "-e", max_bytes,
                            "-f", str(step_factor), "-d", datatype, "-Y", memory_type]
                    env = {"HSA_FORCE_FINE_GRAIN_PCIE": "1"} if memory_type == "fine" else None
                    run_rccl_perf(EXECUTABLE, args, env_overrides=env, timeout=timeout)


@pytest.mark.mpi
def test_alltoall_mpi(gpu_counts, byte_ranges, step_factor, datatypes, request, subtests):
    timeout = request.config.getoption("--test-timeout")
    hostfile = request.config.getoption("--hostfile") or None
    for datatype in datatypes:
        for min_bytes, max_bytes in byte_ranges:
            # 1 GPU per MPI rank (-g 1); scale nprocs via gpu_counts
            for nprocs in gpu_counts:
                with subtests.test(dtype=datatype, nprocs=nprocs,
                                   bytes=f"{min_bytes}-{max_bytes}", step=step_factor):
                    args = ["-t", "1", "-g", "1", "-b", min_bytes, "-e", max_bytes,
                            "-f", str(step_factor), "-d", datatype]
                    run_rccl_mpi(EXECUTABLE, nprocs, args, hostfile=hostfile, timeout=timeout)
