#!/usr/bin/env python3
#
# Copyright (C) Advanced Micro Devices. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

### Handle safe initialization for amdsmi

import atexit
import logging
import signal
import sys
import os

from pathlib import Path

from typing import Optional
current_path = os.path.dirname(os.path.abspath(__file__))
python_lib_path = f"{current_path}/../../share/amd_smi"
sys.path.insert(0, python_lib_path)
# Only fallback to the python library if its a compatible version
# multiple amdsmi versions installed on the system could cause issues

# Ideally we want to identify if the installed python library is incompatible and log a solution to the user
#   LD library config or reinstall, etc...
# The problem is coming from the switch over between the post install and pypi


def _log_version_and_path_diagnostics():
    """Emit best-effort diagnostics about the amdsmi Python package and the shared library location."""
    try:
        from amdsmi import _version  # type: ignore
        pkg_version = getattr(_version, "__version__", "unknown")
    except Exception as exc:  # pragma: no cover - defensive
        pkg_version = f"unavailable ({exc})"

    # Resolve paths from the *wrapper* module, not from this CLI file.
    # The wrapper lives in the amdsmi package dir; the CLI lives elsewhere.
    try:
        import amdsmi.amdsmi_wrapper as _w
        wrapper_path = Path(_w.__file__).resolve()
        wrapper_dir = wrapper_path.parent
    except Exception:
        wrapper_dir = Path("<unknown>")

    print(f"[amdsmi-cli] Python package version: {pkg_version}")
    print(f"[amdsmi-cli] CLI dir: {Path(__file__).resolve().parent}")
    print(f"[amdsmi-cli] Wrapper dir: {wrapper_dir}")
    print(f"[amdsmi-cli] sys.path[0]: {sys.path[0]}")


def _check_version_compatibility(expected_version: Optional[str] = None) -> None:
    """
    Verify that the Python package version matches the expected CLI/lib version (if provided).
    If mismatched, log guidance and abort to avoid loading an incompatible library.

    The actual library loading (pip vs system context) is handled entirely by
    amdsmi_wrapper._load_library().  This function only checks version strings
    and that the wrapper can resolve *some* loadable library candidate.
    """
    try:
        from amdsmi import _version  # type: ignore
        pkg_version = getattr(_version, "__version__", None)
    except Exception:
        _log_version_and_path_diagnostics()
        print("[amdsmi-cli] Failed to read amdsmi Python package version; aborting to avoid incompatibility.")
        sys.exit(1)

    if expected_version and pkg_version and pkg_version != expected_version:
        _log_version_and_path_diagnostics()
        print(f"[amdsmi-cli] Version mismatch: expected {expected_version}, found {pkg_version}.")
        print("[amdsmi-cli] Please install a matching amdsmi wheel from PyPI or reinstall the ROCm package,")
        print("[amdsmi-cli] and ensure LD_LIBRARY_PATH/ldconfig points to the matching shared library.")
        sys.exit(1)

    # Delegate the library-existence check to the wrapper's own detection logic.
    # _build_candidate_paths() uses the wrapper's __file__ to correctly resolve
    # pip (libamd_smi_python.so next to wrapper) vs system (/opt/rocm/lib/libamd_smi.so).
    try:
        from amdsmi.amdsmi_wrapper import _build_candidate_paths
        candidates = _build_candidate_paths()
        for candidate in candidates:
            if isinstance(candidate, str):
                # bare "libamd_smi.so" — let the dynamic linker resolve it later
                return
            if candidate.exists():
                return
    except Exception:
        # If the wrapper isn't importable at all, fall through to the
        # ImportError handler in the try/except block below.
        return

    _log_version_and_path_diagnostics()
    print("[amdsmi-cli] Unable to locate the AMD SMI shared library in expected locations:")
    for c in candidates:
        print(f"  - {c}")
    print("[amdsmi-cli] Install the amdsmi wheel that bundles the Python shared library,")
    print("[amdsmi-cli] or adjust LD_LIBRARY_PATH/ldconfig to point to a compatible lib.")
    sys.exit(1)


try:
    # TODO Add version checking & debug to check pathing
    # The expected version string can be wired in from packaging if desired.
    _check_version_compatibility(expected_version=None)
    from amdsmi import amdsmi_interface, amdsmi_exception
except ImportError as e:
    print(f"Unhandled import error: {e}")
    print("Failed to import the amdsmi Python library. Ensure it is installed in Python.")
    print(f"Alternatively, verify that the library is in the path:\n{python_lib_path}")
    sys.exit(1)

# Using basic python logging for user errors and development
logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.ERROR)  # User level logging
# This traceback limit only affects this file, once the code hit's the cli portion it get's reset to the user's preference
sys.tracebacklimit = -1  # Disable traceback when raising errors

# On initial import set initialized variable
AMDSMI_INITIALIZED = False
AMDSMI_INIT_FLAG = amdsmi_interface.AmdSmiInitFlags.INIT_ALL_PROCESSORS
AMD_VENDOR_ID = 4098


def check_amdgpu_driver():
    """Returns true if amdgpu is found in the list of initialized modules"""
    amd_gpu_status_file = Path("/sys/module/amdgpu/initstate")
    if amd_gpu_status_file.exists():
        try:
            return amd_gpu_status_file.read_text(encoding="ascii").strip() == "live"
        except OSError:
            pass

    # If the driver is loaded either as a module OR built in, this dir will be populated
    drv = Path("/sys/bus/pci/drivers/amdgpu")
    if not drv.exists():
        return False

    # Check if a symlink exists that loosely matches PCI BDF format
    # ex: 0000:03:00.0
    for p in drv.iterdir():
        if p.is_symlink() and ":" in p.name and "." in p.name:
            return True
    return False


def check_amd_hsmp_driver():
    """Returns true if amd_hsmp or hsmp_acpi is found in the list of initialized modules"""
    amd_cpu_status_file = Path("/dev/hsmp")
    if amd_cpu_status_file.exists():
        return True
    return False


def check_amd_ionic_driver():
    """Returns true if ionic is found in the list of initialized modules"""
    status_file = Path("/sys/module/ionic/initstate")
    if status_file.exists():
        if status_file.read_text(encoding="ascii").strip() == "live":
            return True
    return False


def amdsmi_cli_init():
    """Initializes AMDSMI Library for the CLI

    Checks for the presence of the amdgpu, amd_hsmp or hsmp_acpi drivers and initializes the
    AMD SMI library based on the live drivers found.

    Return:
        init_flag: the flag used to initialize the AMD SMI library without error

    Raises:
        err: AmdSmiLibraryException if not successful in initializing any drivers
    """
    init_flag = 0
    if check_amdgpu_driver():
        init_flag |= amdsmi_interface.AmdSmiInitFlags.INIT_AMD_GPUS
        logging.debug("amdgpu driver's initstate is live")
    if check_amd_hsmp_driver():
        init_flag |= amdsmi_interface.AmdSmiInitFlags.INIT_AMD_CPUS
        logging.debug("hsmp driver's initstate is live")
    if check_amd_ionic_driver():
        logging.debug("ionic driver's initstate is live")
        init_flag |= amdsmi_interface.AmdSmiInitFlags.INIT_AMD_NICS

    try:
        amdsmi_interface.amdsmi_init(init_flag)
    except (
        amdsmi_interface.AmdSmiLibraryException,
        amdsmi_interface.AmdSmiParameterException,
    ) as e:
        # parameter exception thrown if init_flag is 0, but err_code will be set to 0 in that case, so must check if init_flag is 0 too
        if (
            e.err_code
            in (
                amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_NOT_INIT,
                amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_DRIVER_NOT_LOADED,
            )
            or init_flag == 0
        ):
            logging.error(
                "Drivers not loaded (amdgpu, amd_hsmp, ionic, rdma drivers not found in modules)"
            )
            sys.exit(-1)
        else:
            raise e

    logging.debug(
        f"AMDSMI initialized with atleast one driver successfully | init flag: {init_flag}"
    )

    return init_flag


def amdsmi_cli_shutdown():
    """Shutdown AMDSMI instance

    Raises:
        err: AmdSmiLibraryException if not successful
    """
    try:
        amdsmi_interface.amdsmi_shut_down()
    except amdsmi_exception.AmdSmiLibraryException as e:
        logging.error("Unable to cleanly shut down amd-smi-lib")
        raise e


def signal_handler(sig, frame):
    logging.debug(f"Handling signal: {sig}")
    try:
        sys.exit(0)
    except Exception as e:
        logging.error(
            "Unable to cleanly shut down amd-smi-lib, exception: %s", str(type(e).__name__)
        )
        os._exit(0)


if not AMDSMI_INITIALIZED:
    AMDSMI_INIT_FLAG = amdsmi_cli_init()
    AMDSMI_INITIALIZED = True
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    atexit.register(amdsmi_cli_shutdown)
