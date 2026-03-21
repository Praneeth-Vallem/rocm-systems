# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import os
import re
import socket
import subprocess
import warnings
import pytest

from test_runner import BUILD_DIR


def _visible_device_count(var_name):
    """Parse a comma-separated device list, returning None if unset/empty."""
    raw = os.environ.get(var_name)
    if raw is None:
        return None
    raw = raw.strip()
    if not raw:
        return None
    return len([x for x in raw.split(",") if x.strip()])


def detect_gpu_count():
    for var in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"):
        n = _visible_device_count(var)
        if n is not None:
            return n
    try:
        out = subprocess.check_output(
            ["rocminfo"],
            text=True, timeout=30,
        )
        gpu_count = sum(
            1 for line in out.splitlines()
            if "Device Type" in line and "GPU" in line
        )
        return int(gpu_count)
    except (subprocess.SubprocessError, ValueError):
        pytest.exit("Failed to detect GPU count", returncode=1)


PROFILES = {
    "smoke": {
        "byte_ranges":  [("1K", "1G")],
        "ops":          ["sum"],
        "datatypes":    ["float", "half", "bfloat16", "fp8_e5m2"],
        "memory_types": ["coarse"],
        "step_factor":  4,
        "gpu_sweep":    "power_of_2",
    },
    "stress": {
        "byte_ranges":  [("4", "1K"), ("1K", "1M"), ("1M", "4G")],
        "ops":          ["sum", "prod", "min", "max", "avg", "mulsum"],
        "datatypes":    ["int8", "uint8", "int32", "uint32", "int64", "uint64",
                         "half", "float", "double", "bfloat16",
                         "fp8_e4m3", "fp8_e5m2"],
        "memory_types": ["coarse", "fine", "host", "managed"],
        "step_factor":  2,
        "gpu_sweep":    "all",
    },
}
DEFAULT_PROFILE = "smoke"


def pytest_addoption(parser):
    group = parser.getgroup("rccl", "RCCL-Tests Options")
    group.addoption("--hostfile", action="store", default="",
                    help="MPI hostfile for multi-node tests")
    group.addoption("--test-timeout", action="store", default=300, type=int,
                    help="Per-test subprocess timeout in seconds (default: 300)")
    group.addoption("--msg-profile", action="store", default=DEFAULT_PROFILE,
                    choices=PROFILES.keys(),
                    help="Message size profile: smoke (default) or stress")


_RCCL_VERSION_FIELDS = {
    "RCCL version": "rccl_version",
    "HIP version":  "hip_version",
    "ROCm version": "rocm_version",
    "Hostname":     "hostname",
    "Librccl path": "librccl_path",
}


def _detect_rccl_info():
    """Run a perf binary with NCCL_DEBUG=VERSION and parse metadata from stderr.

    RCCL prints key-value pairs like:
        RCCL version : 2.28.3-develop:d143eea
        HIP version  : 7.2.26015-fc0010cf6a
        ROCm version : 7.2.0.0-43-fc0010cf6a
        Hostname     : node-01.example.com
        Librccl path : /opt/rocm/lib/librccl.so.1
    """
    info = {}
    executable = os.path.join(BUILD_DIR, "all_reduce_perf")
    if not os.path.isfile(executable):
        info["_warning"] = f"binary not found: {executable}"
        return info

    env = os.environ.copy()
    env["NCCL_DEBUG"] = "VERSION"
    try:
        result = subprocess.run(
            [executable, "-b", "8", "-e", "8", "-t", "1", "-g", "1"],
            capture_output=True, text=True, timeout=30, env=env,
        )
    except subprocess.TimeoutExpired:
        info["_warning"] = f"{executable} timed out during version probe"
        return info
    except (subprocess.SubprocessError, OSError) as e:
        info["_warning"] = str(e)
        return info

    output = result.stderr + "\n" + result.stdout
    for line in output.splitlines():
        for label, key in _RCCL_VERSION_FIELDS.items():
            m = re.match(rf"{re.escape(label)}\s*:\s*(.+)", line)
            if m:
                info[key] = m.group(1).strip()
                break

    if not info and result.returncode != 0:
        hint = result.stderr.strip().splitlines()
        first_line = hint[0] if hint else f"exit code {result.returncode}"
        info["_warning"] = first_line

    return info


def pytest_report_header(config):
    profile = config.getoption("--msg-profile")
    timeout = config.getoption("--test-timeout")
    hostfile = config.getoption("--hostfile")

    rccl = _detect_rccl_info()
    hostname = rccl.get("hostname", socket.gethostname())
    hip_visible = os.environ.get("HIP_VISIBLE_DEVICES", "all")

    lines = [
        f"rccl-tests  profile: {profile}  timeout: {timeout}s"
        + (f"  hostfile: {hostfile}" if hostfile else ""),
        f"  Hostname:             {hostname}",
        f"  HIP_VISIBLE_DEVICES:  {hip_visible}",
    ]
    if rccl.get("rccl_version"):
        lines.append(f"  RCCL version:         {rccl['rccl_version']}")
    if rccl.get("hip_version"):
        lines.append(f"  HIP version:          {rccl['hip_version']}")
    if rccl.get("rocm_version"):
        lines.append(f"  ROCm version:         {rccl['rocm_version']}")
    if rccl.get("librccl_path"):
        lines.append(f"  Librccl path:         {rccl['librccl_path']}")
    if rccl.get("_warning"):
        lines.append(f"  RCCL info:            (unavailable) {rccl['_warning']}")

    return lines


@pytest.fixture(scope="session")
def gpu_info():
    ngpus = detect_gpu_count()
    if ngpus == 0:
        pytest.exit("No GPUs detected", returncode=1)
    return {"ngpus": ngpus}


@pytest.fixture(scope="session")
def profile(request):
    name = request.config.getoption("--msg-profile")
    return PROFILES[name]


@pytest.fixture(scope="session")
def byte_ranges(profile):
    return profile["byte_ranges"]


@pytest.fixture(scope="session")
def ops(profile):
    return profile["ops"]


@pytest.fixture(scope="session")
def datatypes(profile):
    return profile["datatypes"]


@pytest.fixture(scope="session")
def memory_types(profile):
    return profile["memory_types"]


@pytest.fixture(scope="session")
def step_factor(profile):
    return profile["step_factor"]


@pytest.fixture(scope="session")
def gpu_counts(gpu_info, profile):
    ngpus = gpu_info["ngpus"]
    sweep = profile["gpu_sweep"]

    if sweep == "power_of_2":
        requested = [2**x for x in range(ngpus.bit_length())]
        counts = [x for x in requested if x <= ngpus]
    elif sweep == "all":
        counts = list(range(1, ngpus + 1))
        requested = counts
    else:
        requested = list(sweep)
        counts = [x for x in requested if x <= ngpus]

    skipped = sorted(set(requested) - set(counts))
    if skipped:
        warnings.warn(
            f"gpu_sweep: requested {requested} but only {ngpus} GPUs available; "
            f"skipped {skipped}",
            stacklevel=1,
        )

    return [str(x) for x in counts]


# ---------------------------------------------------------------------------
# Failed-subtest summary: surface the exact configs that failed at the end
# ---------------------------------------------------------------------------

_FAILED_SUBTEST_REPORTS = {}
_SUBTEST_FAILED_ONCE = set()
_SUBTEST_FLAKY_PASSED = set()


def pytest_runtest_logreport(report):
    """Track failed subtests and keep only final outcomes."""
    if report.when != "call":
        return
    if not hasattr(report, "context"):
        return
    if getattr(report, "outcome", None) == "rerun":
        return
    nodeid = report.nodeid
    if report.passed:
        if nodeid in _SUBTEST_FAILED_ONCE:
            _SUBTEST_FLAKY_PASSED.add(nodeid)
        _FAILED_SUBTEST_REPORTS.pop(nodeid, None)
    elif report.failed:
        _SUBTEST_FAILED_ONCE.add(nodeid)
        _SUBTEST_FLAKY_PASSED.discard(nodeid)
        _FAILED_SUBTEST_REPORTS[nodeid] = report


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    final_flaky = sorted(
        nodeid for nodeid in _SUBTEST_FLAKY_PASSED
        if nodeid not in _FAILED_SUBTEST_REPORTS
    )

    if _FAILED_SUBTEST_REPORTS:
        terminalreporter.section("Failed subtest configurations")
        for nodeid in sorted(_FAILED_SUBTEST_REPORTS):
            report = _FAILED_SUBTEST_REPORTS[nodeid]
            terminalreporter.line(f"[FAILED] {nodeid}")
            if report.longreprtext:
                for line in report.longreprtext.strip().splitlines():
                    if line.startswith("E "):
                        terminalreporter.line(f"  {line}")
                        break

    if final_flaky:
        terminalreporter.section("Flaky subtests (failed then passed on rerun) configurations")
        for nodeid in final_flaky:
            terminalreporter.line(f"[FLAKY] {nodeid}")
