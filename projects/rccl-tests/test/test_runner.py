# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import os
import subprocess
import pytest

BUILD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build")


def run_rccl_perf(executable_name, args, env_overrides=None, timeout=300):
    executable = os.path.join(BUILD_DIR, executable_name)
    cmd = [executable] + args
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=timeout, env=env)
    except subprocess.TimeoutExpired as e:
        pytest.fail(f"{executable_name} timed out after {timeout}s\n"
                    f"stdout: {e.stdout}\nstderr: {e.stderr}")

    if result.returncode != 0:
        pytest.fail(f"{executable_name} failed (rc={result.returncode})\n"
                    f"cmd: {' '.join(cmd)}\n"
                    f"stdout: {result.stdout}\nstderr: {result.stderr}")

    return result


def run_rccl_mpi(executable_name, nprocs, args, hostfile=None, timeout=300):
    executable = os.path.join(BUILD_DIR, executable_name)
    cmd = ["mpirun", "-np", str(nprocs)]
    if hostfile:
        cmd += ["-host", hostfile]
    cmd += [executable, "-p", "1"] + args

    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=timeout)
    except subprocess.TimeoutExpired as e:
        pytest.fail(f"MPI {executable_name} timed out after {timeout}s\n"
                    f"stdout: {e.stdout}\nstderr: {e.stderr}")

    if result.returncode != 0:
        pytest.fail(f"MPI {executable_name} failed (rc={result.returncode})\n"
                    f"cmd: {' '.join(cmd)}\n"
                    f"stdout: {result.stdout}\nstderr: {result.stderr}")

    return result
