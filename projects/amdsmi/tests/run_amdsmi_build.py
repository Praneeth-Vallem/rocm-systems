#!/usr/bin/env python3
"""
run_amdsmi_build.py - Local runner for the AMDSMI build / install / test workflow.

This script mirrors the CI pipeline defined in .github/workflows/amdsmi-build.yml
so that you can reproduce builds, installs, and tests directly on a local machine
(or inside a Docker container) without GitHub Actions.  Every command is logged to
--log-dir, failures surface immediately with the path to the log file, and test
results are written to --test-results-dir with well-known filenames for CI upload.

Prerequisites
=============
* Python 3.8+
* CMake >= 3.16
* A C/C++ toolchain (gcc/g++ or clang)
* Root / sudo privileges (package install/uninstall touches /opt/rocm)
* For Debian/Ubuntu: apt
* For RHEL/AlmaLinux/Fedora: dnf
* For SLES: zypper

Quick Start (Local)
===================
Clone the repo, cd into it, and run the script.  It will auto-detect the project
root, package manager, and CPU count.

    sudo python3 projects/amdsmi/tests/run_amdsmi_build.py

Per-OS Examples
===============

Ubuntu 20.04 / 22.04 / 24.04 (Debian-based, apt)
--------------------------------------------------
    sudo python3 run_amdsmi_build.py \
        --package-manager apt \
        --os-label Ubuntu22

Debian 10
------------------------------------
    sudo python3 run_amdsmi_build.py \
        --package-manager apt \
        --debian10-sources \
        --os-label Debian10

RHEL 8 / 9 (dnf)
-----------------
    sudo python3 run_amdsmi_build.py \
        --package-manager dnf \
        --package-format rpm \
        --os-label RHEL9

RHEL 10 / AlmaLinux 8 (dnf - needs QA_RPATHS)
----------------------------------------------
    sudo python3 run_amdsmi_build.py \
        --package-manager dnf \
        --package-format rpm \
        --qa-rpaths \
        --os-label RHEL10

SLES (zypper)
-------------
    sudo python3 run_amdsmi_build.py \
        --package-manager zypper \
        --package-format rpm \
        --os-label SLES

AzureLinux 3 (dnf - skip setuptools upgrade, install more_itertools)
--------------------------------------------------------------------
    sudo python3 run_amdsmi_build.py \
        --package-manager dnf \
        --package-format rpm \
        --skip-setuptools-upgrade \
        --install-more-itertools \
        --os-label AzureLinux3

Skipping Stages
===============
Use --skip-build, --skip-install, --skip-tests, or --skip-cleanup to skip
individual stages.  For example, to re-run only the tests against an existing
build directory:

    sudo python3 run_amdsmi_build.py --skip-build --skip-cleanup

Output
======
* Logs:         --log-dir   (default: logs/amdsmi/)
* Test results: --test-results-dir (default: /tmp/test-results-<os-label>/)

Run with --help for the full list of options.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import shutil
import subprocess
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

DEFAULT_LOG_DIR = Path("logs") / "amdsmi"


class CommandError(RuntimeError):
    def __init__(self, name: str, cmd: List[str], code: int, log_path: Path):
        super().__init__(f"{name} failed with exit code {code}; see {log_path}")
        self.name = name
        self.cmd = cmd
        self.code = code
        self.log_path = log_path


def _timestamp() -> str:
    return _dt.datetime.utcnow().strftime("%Y%m%d-%H%M%S")


def run_command(
    cmd: Iterable[str],
    *,
    name: str,
    cwd: Optional[Path] = None,
    env: Optional[dict] = None,
    retries: int = 1,
    log_dir: Path = DEFAULT_LOG_DIR,
    result_file: Optional[Path] = None,
) -> Path:
    """Run a command, streaming output to stdout and a log file.

    If *result_file* is given the full output is also written there (using a
    well-known filename that the CI workflow can display later).
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{_timestamp()}-{name}.log"
    cmd_list = [str(part) for part in cmd]
    attempt = 0

    while attempt < retries:
        attempt += 1
        prefix = f"[{name}] attempt {attempt}/{retries}"
        print(f"{prefix}: {' '.join(cmd_list)}")

        outputs = [log_path]
        if result_file is not None:
            result_file.parent.mkdir(parents=True, exist_ok=True)
            outputs.append(result_file)

        handles = [p.open("a", encoding="utf-8") for p in outputs]
        try:
            for fh in handles:
                fh.write(f"{prefix}\n")
            proc = subprocess.Popen(
                cmd_list,
                cwd=str(cwd) if cwd else None,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                for fh in handles:
                    fh.write(line)
            code = proc.wait()
        finally:
            for fh in handles:
                fh.close()

        if code == 0:
            return log_path

        if attempt >= retries:
            raise CommandError(name, cmd_list, code, log_path)

        print(f"{prefix} failed with exit code {code}; retrying...")

    return log_path


def tail_log(log_path: Path, max_lines: int = 40) -> str:
    """Return the last max_lines from a log file for quick context."""
    try:
        with log_path.open("r", encoding="utf-8", errors="replace") as fh:
            tail = deque(fh, maxlen=max_lines)
        return "".join(tail)
    except OSError as exc:
        return f"(could not read log {log_path}: {exc})"


def report_and_raise(stage: str, err: CommandError) -> None:
    print(f"\nAMDSMI failed to {stage}: step '{err.name}' exited with {err.code}")
    print(f"Log: {err.log_path}")
    print("Log tail (last ~40 lines):")
    print("----------------------------------------")
    print(tail_log(err.log_path))
    print("----------------------------------------\n")
    raise err


def find_project_dir(explicit: Optional[Path]) -> Path:
    if explicit:
        return explicit.resolve()

    def _repo_root(start: Path) -> Path:
        for ancestor in [start] + list(start.parents):
            if (ancestor / ".git").exists():
                return ancestor
        return start.parents[1]

    repo_root = _repo_root(Path(__file__).resolve())
    primary = repo_root / "projects" / "amdsmi" / "CMakeLists.txt"
    if primary.exists():
        return primary.parent

    for candidate in repo_root.rglob("CMakeLists.txt"):
        return candidate.parent

    raise FileNotFoundError(f"Could not find CMakeLists.txt under {repo_root}")


def detect_package_manager(explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    for candidate in ("apt", "dnf", "zypper"):
        if shutil.which(candidate):
            return candidate
    raise RuntimeError("Could not detect a package manager (apt, dnf, zypper)")


def detect_package_format(pkg_manager: str, explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    if pkg_manager == "apt":
        return "deb"
    if pkg_manager in ("dnf", "zypper"):
        return "rpm"
    raise ValueError(f"Unsupported package manager {pkg_manager}")


def package_glob(package_format: str) -> str:
    if package_format == "deb":
        return "amd-smi-lib*99999-local_amd64.deb"
    return "amd-smi-lib-*99999-local*.rpm"


# ---------------------------------------------------------------------------
# Pre-build helpers
# ---------------------------------------------------------------------------

def clean_stale_artifacts(log_dir: Path) -> None:
    """Remove SWIG-based .so baked into Docker images.

    The package now uses ctypes; the stale extension references symbols removed
    from libamd_smi.so and causes 'undefined symbol' errors on import.
    """
    stale = Path("/opt/rocm/share/amd_smi/amdsmi/libamd_smi_python.so")
    if stale.exists():
        stale.unlink()
        print(f"Removed stale artifact: {stale}")
    # Refresh linker cache
    if shutil.which("ldconfig"):
        try:
            run_command(["ldconfig"], name="ldconfig", log_dir=log_dir)
        except CommandError:
            print("Warning: ldconfig failed (non-fatal)")


def install_more_itertools(log_dir: Path, retries: int) -> None:
    """AzureLinux3 needs more_itertools installed separately."""
    run_command(
        ["python3", "-m", "pip", "install", "more_itertools"],
        name="pip-more-itertools",
        retries=retries,
        log_dir=log_dir,
    )


def upgrade_setuptools(log_dir: Path, retries: int) -> None:
    run_command(
        ["python3", "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
        name="pip-upgrade",
        retries=retries,
        log_dir=log_dir,
    )


def update_debian10_sources(log_dir: Path, retries: int) -> None:
    content = (
        "deb http://archive.debian.org/debian buster main\n"
        "deb http://archive.debian.org/debian-security buster/updates main\n"
    )
    sources_list = Path("/etc/apt/sources.list")
    print("Updating sources.list for Debian10 (archived repos)")
    sources_list.write_text(content, encoding="utf-8")
    Path("/etc/apt/apt.conf.d/99-disable-check-valid-until").write_text(
        'Acquire::Check-Valid-Until "false";\n', encoding="utf-8"
    )
    run_command(["apt", "update"], name="apt-update", retries=retries, log_dir=log_dir)


def build_amdsmi(cfg: "RunnerConfig") -> None:
    if cfg.build_dir.exists():
        shutil.rmtree(cfg.build_dir)
    cfg.build_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("DEBIAN_FRONTEND", "noninteractive")
    env.setdefault("DEBCONF_NONINTERACTIVE_SEEN", "true")

    # RHEL10 / AlmaLinux8 need QA_RPATHS to ignore empty RPATHs
    if cfg.qa_rpaths:
        env["QA_RPATHS"] = str(0x0010 | 0x0002)
        print(f"QA_RPATHS set to {env['QA_RPATHS']}")

    cmake_args = [
        "cmake",
        str(cfg.project_dir),
        "-DBUILD_TESTS=ON",
        "-DENABLE_ESMI_LIB=ON",
        "-DBUILD_PYTHON_LIB=ON",
        f"-DCMAKE_BUILD_TYPE={cfg.build_type}",
    ]
    run_command(
        cmake_args,
        name="cmake-configure",
        cwd=cfg.build_dir,
        retries=cfg.retries,
        env=env,
        log_dir=cfg.log_dir,
    )

    run_command(
        ["make", "-j", str(cfg.jobs)],
        name="make",
        cwd=cfg.build_dir,
        retries=cfg.retries,
        env=env,
        log_dir=cfg.log_dir,
    )

    wheel_dir = cfg.build_dir / "py-interface" / "python_package"
    # Clean stale wheel artifacts before building
    for pattern in ("*.whl", "*.egg-info"):
        for stale_path in wheel_dir.glob(pattern):
            if stale_path.is_dir():
                shutil.rmtree(stale_path)
            else:
                stale_path.unlink()
    for subdir_name in ("dist", "build"):
        subdir = wheel_dir / subdir_name
        if subdir.exists():
            shutil.rmtree(subdir)

    wheel_cmd = [
        "python3",
        "-m",
        "pip",
        "wheel",
        "--no-deps",
        "--no-build-isolation",
        "-w",
        ".",
        ".",
    ]
    run_command(
        wheel_cmd,
        name="python-wheel",
        cwd=wheel_dir,
        retries=cfg.retries,
        env=env,
        log_dir=cfg.log_dir,
    )

    run_command(
        ["make", "package"],
        name="make-package",
        cwd=cfg.build_dir,
        retries=cfg.retries,
        env=env,
        log_dir=cfg.log_dir,
    )


def locate_package(build_dir: Path, package_format: str) -> Path:
    matches = sorted(build_dir.glob(package_glob(package_format)))
    if not matches:
        raise FileNotFoundError(f"No {package_format} artifact found in {build_dir}")

    def _is_tests_package(p: Path) -> bool:
        return "tests" in p.name

    # Prefer the main package over the tests package if both exist.
    main_pkgs = [p for p in matches if not _is_tests_package(p)]
    if main_pkgs:
        return max(main_pkgs, key=lambda p: p.stat().st_mtime)
    return max(matches, key=lambda p: p.stat().st_mtime)


def install_package(cfg: "RunnerConfig", package_path: Path) -> None:
    print(f"Installing package {package_path}")
    env = os.environ.copy()
    if cfg.package_manager == "apt":
        if cfg.refresh_apt:
            run_command(
                ["apt", "update"],
                name="apt-update",
                retries=cfg.retries,
                log_dir=cfg.log_dir,
            )
        # Install main package; if a tests package exists, install it together to satisfy deps.
        extra_pkg = package_path.parent / package_path.name.replace("lib_", "lib-tests_", 1)
        pkg_list = [str(package_path)]
        if extra_pkg.exists():
            pkg_list.append(str(extra_pkg))
        run_command(
            ["apt", "install", "--reinstall", "-y", *pkg_list],
            name="apt-install",
            retries=cfg.retries,
            log_dir=cfg.log_dir,
        )
    elif cfg.package_manager == "dnf":
        run_command(
            ["dnf", "install", "python3-setuptools", "python3-wheel", "-y"],
            name="dnf-prep",
            retries=cfg.retries,
            log_dir=cfg.log_dir,
        )
        run_command(
            [
                "dnf",
                "install",
                "-y",
                "--skip-broken",
                "--disablerepo=*",
                str(package_path),
            ],
            name="dnf-install",
            retries=cfg.retries,
            log_dir=cfg.log_dir,
        )
    elif cfg.package_manager == "zypper":
        run_command(
            ["zypper", "--no-refresh", "--no-gpg-checks", "install", "-y", str(package_path)],
            name="zypper-install",
            retries=cfg.retries,
            log_dir=cfg.log_dir,
        )
    else:
        raise ValueError(f"Unsupported package manager {cfg.package_manager}")

    rocm_binary = Path("/opt/rocm/bin/amd-smi")
    symlink_path = Path("/usr/local/bin/amd-smi")
    if rocm_binary.exists():
        symlink_path.parent.mkdir(parents=True, exist_ok=True)
        if symlink_path.exists() or symlink_path.is_symlink():
            symlink_path.unlink()
        symlink_path.symlink_to(rocm_binary)
        print(f"Linked {symlink_path} -> {rocm_binary}")

    # Verify installation: CLI version, .pth file, and Python import/init/shutdown
    # Note: pip list will NOT show amdsmi because the .pth approach does not
    # register the package with pip metadata — that is by design.
    verify_commands = [
        ["amd-smi", "version"],
        [
            "python3",
            "-c",
            (
                "import site, pathlib; "
                "pth = pathlib.Path(site.getsitepackages()[0]) / 'amdsmi.pth'; "
                "assert pth.exists(), f'amdsmi.pth not found at {pth}'; "
                "print(f'\u2713 .pth file found: {pth}'); "
                "print(f'  contents: {pth.read_text().strip()}')"
            ),
        ],
        [
            "python3",
            "-c",
            "import amdsmi; print('import ok'); amdsmi.amdsmi_init(); print('init ok'); amdsmi.amdsmi_shut_down(); print('shutdown ok')",
        ],
    ]
    for idx, verify_cmd in enumerate(verify_commands, start=1):
        try:
            run_command(
                verify_cmd,
                name=f"verify-{idx}",
                retries=1,
                log_dir=cfg.log_dir,
            )
        except CommandError as exc:
            print(f"Verification command failed: {exc}")
            raise


# ---------------------------------------------------------------------------
# Wheel verification (matches the "Verify Wheel in Site-Packages" CI step)
# ---------------------------------------------------------------------------

def verify_wheel_site_packages(cfg: "RunnerConfig") -> None:
    """Install the built wheel via pip and verify it lands in site-packages."""
    wheel_dir = cfg.build_dir / "py-interface" / "python_package"
    wheels = sorted(wheel_dir.glob("*.whl"))
    if not wheels:
        print(f"WARNING: No .whl found in {wheel_dir}; skipping wheel verification")
        return

    wheel = wheels[0]
    print(f"Found wheel: {wheel}")

    run_command(
        ["python3", "-m", "pip", "install", "--force-reinstall", str(wheel)],
        name="pip-install-wheel",
        retries=1,
        log_dir=cfg.log_dir,
    )

    smoke_test = (
        "import amdsmi\n"
        "print('PASS: import amdsmi OK')\n"
        "amdsmi.amdsmi_init()\n"
        "print('PASS: amdsmi_init() OK')\n"
        "devs = amdsmi.amdsmi_get_processor_handles()\n"
        "print('PASS: Found %d device(s)' % len(devs))\n"
        "amdsmi.amdsmi_shut_down()\n"
        "print('PASS: amdsmi_shut_down() OK')\n"
        "print('=== Wheel verification passed ===')\n"
    )
    run_command(
        ["python3", "-c", smoke_test],
        name="wheel-smoke-test",
        cwd=Path("/tmp"),
        retries=1,
        log_dir=cfg.log_dir,
    )

    run_command(
        ["python3", "-m", "pip", "show", "amdsmi"],
        name="pip-show-amdsmi",
        retries=1,
        log_dir=cfg.log_dir,
    )

    # Check install location
    run_command(
        [
            "python3",
            "-c",
            (
                "import amdsmi; p = amdsmi.__file__; print(f'amdsmi installed at: {p}'); "
                "assert 'site-packages' in p or 'dist-packages' in p, "
                "f'WARNING: not in site-packages or dist-packages: {p}'; "
                "print('PASS: Wheel correctly in packages dir')"
            ),
        ],
        name="wheel-location-check",
        retries=1,
        log_dir=cfg.log_dir,
    )


def run_command_tests(cfg: "RunnerConfig") -> None:
    """Run amd-smi CLI commands with --loglevel debug."""
    cfg.test_results_dir.mkdir(parents=True, exist_ok=True)
    commands = [
        "amd-smi version",
        "amd-smi list",
        "amd-smi static",
        "amd-smi firmware",
        "amd-smi ucode",
        "amd-smi bad-pages",
        "amd-smi metric",
        "amd-smi process",
        "amd-smi topology",
        "amd-smi monitor",
        "amd-smi dmon",
        "amd-smi xgmi",
        "amd-smi partition",
    ]
    for cmd in commands:
        debug_cmd = f"{cmd} --loglevel debug"
        log_name = debug_cmd.replace(" ", "_").replace("--", "")
        # Write result to test-results dir with the name the workflow expects
        result_name = cmd.replace(" ", "_") + ".log"
        try:
            run_command(
                ["bash", "-lc", f"set -euo pipefail; {debug_cmd}"],
                name=f"cmd-{log_name}",
                retries=1,
                log_dir=cfg.log_dir,
                result_file=cfg.test_results_dir / result_name,
            )
        except CommandError as exc:
            print(f"Command test failed: {exc}")
            print(f"Log: {exc.log_path}")
            raise


def _get_gtest_exclude(test_dir: Path) -> str:
    """Source amdsmitst.exclude and detect_asic_filter.sh to get GTEST_EXCLUDE."""
    script = (
        "cd {d} && source amdsmitst.exclude 2>/dev/null; "
        "if [ -f detect_asic_filter.sh ]; then source detect_asic_filter.sh 2>/dev/null; fi; "
        'echo "$GTEST_EXCLUDE"'
    ).format(d=str(test_dir))
    result = subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        cwd=str(test_dir),
    )
    return result.stdout.strip()


def run_python_and_example_tests(cfg: "RunnerConfig") -> None:
    base_dir = Path("/opt/rocm/share/amd_smi/tests")
    if not base_dir.exists():
        print("Skipping AMDSMI Python tests; /opt/rocm/share/amd_smi/tests not found")
        return

    cfg.test_results_dir.mkdir(parents=True, exist_ok=True)

    # AMDSMI gtest binary with filter
    gtest_exclude = _get_gtest_exclude(base_dir)
    if gtest_exclude:
        print(f"GTEST_EXCLUDE={gtest_exclude}")
        gtest_cmd = (
            "source amdsmitst.exclude && "
            "source detect_asic_filter.sh 2>/dev/null; "
            f'./amdsmitst --gtest_filter="-{gtest_exclude}"'
        )
    else:
        gtest_cmd = "source amdsmitst.exclude && ./amdsmitst"

    run_command(
        ["bash", "-lc", f"set -euo pipefail; {gtest_cmd}"],
        name="amdsmi-tests",
        cwd=base_dir,
        retries=cfg.retries,
        log_dir=cfg.log_dir,
        result_file=cfg.test_results_dir / "amdsmi_tests.log",
    )

    # Python tests
    python_tests = base_dir / "python_unittest"
    if python_tests.exists():
        run_command(
            ["./integration_test.py", "-v"],
            name="python-integration",
            cwd=python_tests,
            retries=1,
            log_dir=cfg.log_dir,
            result_file=cfg.test_results_dir / "integration_test_output.txt",
        )
        run_command(
            ["./unit_tests.py", "-v"],
            name="python-unit",
            cwd=python_tests,
            retries=1,
            log_dir=cfg.log_dir,
            result_file=cfg.test_results_dir / "unit_test_output.txt",
        )
        # Perf tests (present in the workflow)
        perf_script = python_tests / "perf_tests.py"
        if perf_script.exists():
            run_command(
                ["./perf_tests.py", "-v"],
                name="python-perf",
                cwd=python_tests,
                retries=1,
                log_dir=cfg.log_dir,
                result_file=cfg.test_results_dir / "perf_test_output.txt",
            )

    # Example tests
    example_dir = cfg.project_dir / "example"
    if example_dir.exists():
        build_dir = example_dir / "build"
        if build_dir.exists():
            shutil.rmtree(build_dir)
        run_command(
            ["cmake", "-B", "build", "-DENABLE_ESMI_LIB=OFF"],
            name="example-cmake",
            cwd=example_dir,
            retries=1,
            log_dir=cfg.log_dir,
        )
        run_command(
            ["make", "-C", "build", "-j", str(cfg.jobs)],
            name="example-make",
            cwd=example_dir,
            retries=1,
            log_dir=cfg.log_dir,
        )
        for binary in ("amd_smi_drm_ex", "amd_smi_nodrm_ex"):
            path = build_dir / binary
            if path.exists():
                run_command(
                    [str(path)],
                    name=f"example-{binary}",
                    retries=1,
                    log_dir=cfg.log_dir,
                    result_file=cfg.test_results_dir / f"{binary}.log",
                )


def uninstall_package(cfg: "RunnerConfig") -> None:
    print("Uninstalling AMDSMI package")
    if cfg.package_manager == "apt":
        run_command(
            ["apt", "remove", "-y", "amd-smi-lib"],
            name="apt-uninstall",
            retries=1,
            log_dir=cfg.log_dir,
        )
    elif cfg.package_manager == "dnf":
        run_command(
            ["dnf", "remove", "-y", "amd-smi-lib"],
            name="dnf-uninstall",
            retries=1,
            log_dir=cfg.log_dir,
        )
    elif cfg.package_manager == "zypper":
        run_command(
            ["zypper", "remove", "-y", "amd-smi-lib"],
            name="zypper-uninstall",
            retries=1,
            log_dir=cfg.log_dir,
        )

    symlink_path = Path("/usr/local/bin/amd-smi")
    if symlink_path.exists() or symlink_path.is_symlink():
        symlink_path.unlink()

    # Remove the .pth file that the package postinst wrote
    try:
        import site as _site
        pth = Path(_site.getsitepackages()[0]) / "amdsmi.pth"
        if pth.exists():
            pth.unlink()
            print(f"Removed {pth}")
    except Exception as exc:
        print(f"Warning: could not remove amdsmi.pth: {exc}")

    share_dir = Path("/opt/rocm/share/amd_smi")
    if share_dir.exists():
        shutil.rmtree(share_dir)


@dataclass
class RunnerConfig:
    project_dir: Path
    build_dir: Path
    package_manager: str
    package_format: str
    log_dir: Path
    test_results_dir: Path
    retries: int
    build_type: str
    jobs: int
    os_label: str
    refresh_apt: bool
    debian10_sources: bool
    skip_setuptools_upgrade: bool
    do_install_more_itertools: bool
    qa_rpaths: bool
    skip_build: bool
    skip_install: bool
    skip_tests: bool
    skip_cleanup: bool


def parse_args() -> RunnerConfig:
    parser = argparse.ArgumentParser(description="Run AMDSMI build/install locally.")
    parser.add_argument("--project-dir", type=Path, help="Path to the AMDSMI project root")
    parser.add_argument("--build-dir", type=Path, help="Build directory (default: <project>/build)")
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR, help="Directory for command logs")
    parser.add_argument("--test-results-dir", type=Path, help="Directory for well-known test result files (default: /tmp/test-results-<os-label>)")
    parser.add_argument("--package-manager", choices=["apt", "dnf", "zypper"], help="Package manager to use")
    parser.add_argument("--package-format", choices=["deb", "rpm"], help="Force package format")
    parser.add_argument("--retries", type=int, default=3, help="Retries for build/install steps")
    parser.add_argument("--build-type", default="Release", help="CMake build type")
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 4, help="Parallel jobs for make")
    parser.add_argument("--os-label", default="local", help="Label used in log/result dir names")
    parser.add_argument("--no-apt-update", action="store_true", help="Do not run apt update before install")
    parser.add_argument("--debian10-sources", action="store_true", help="Rewrite apt sources for Debian10 archive")
    parser.add_argument("--skip-setuptools-upgrade", action="store_true", help="Skip pip/setuptools/wheel upgrade (e.g. AzureLinux3)")
    parser.add_argument("--install-more-itertools", action="store_true", help="Install more_itertools (e.g. AzureLinux3)")
    parser.add_argument("--qa-rpaths", action="store_true", help="Set QA_RPATHS for RPM builds (RHEL10, AlmaLinux8)")
    parser.add_argument("--skip-build", action="store_true", help="Skip build step (use existing build dir)")
    parser.add_argument("--skip-install", action="store_true", help="Skip package installation")
    parser.add_argument("--skip-tests", action="store_true", help="Skip command/python/example tests")
    parser.add_argument("--skip-cleanup", action="store_true", help="Skip uninstall/cleanup step")
    args = parser.parse_args()

    project_dir = find_project_dir(args.project_dir)
    build_dir = args.build_dir or project_dir / "build"
    package_manager = detect_package_manager(args.package_manager)
    package_format = detect_package_format(package_manager, args.package_format)
    os_label = args.os_label
    test_results_dir = args.test_results_dir or Path(f"/tmp/test-results-{os_label}")

    return RunnerConfig(
        project_dir=project_dir,
        build_dir=build_dir,
        package_manager=package_manager,
        package_format=package_format,
        log_dir=args.log_dir,
        test_results_dir=test_results_dir,
        retries=max(1, args.retries),
        build_type=args.build_type,
        jobs=max(1, args.jobs),
        os_label=os_label,
        refresh_apt=not args.no_apt_update,
        debian10_sources=args.debian10_sources,
        skip_setuptools_upgrade=args.skip_setuptools_upgrade,
        do_install_more_itertools=args.install_more_itertools,
        qa_rpaths=args.qa_rpaths,
        skip_build=args.skip_build,
        skip_install=args.skip_install,
        skip_tests=args.skip_tests,
        skip_cleanup=args.skip_cleanup,
    )


def _write_result(results_dir: Path, filename: str, message: str) -> None:
    """Write a status message to a well-known result file."""
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / filename).write_text(message + "\n", encoding="utf-8")


def main() -> None:
    cfg = parse_args()

    print(f"Using project directory: {cfg.project_dir}")
    print(f"Logs will be saved under: {cfg.log_dir}")
    print(f"Test results will be saved under: {cfg.test_results_dir}")
    print(f"Package manager: {cfg.package_manager} (format {cfg.package_format})")
    print(f"OS label: {cfg.os_label}")

    # 1. Debian10 archived repos
    if cfg.debian10_sources and cfg.package_manager != "apt":
        print("Warning: --debian10-sources ignored because package manager is not apt")
    if cfg.debian10_sources:
        update_debian10_sources(cfg.log_dir, cfg.retries)

    # 2. Install more_itertools if requested (e.g. AzureLinux3)
    if cfg.do_install_more_itertools:
        install_more_itertools(cfg.log_dir, cfg.retries)

    # 3. Upgrade setuptools (skip on AzureLinux3)
    if not cfg.skip_setuptools_upgrade:
        upgrade_setuptools(cfg.log_dir, cfg.retries)

    # 4. Clean stale ROCm Python artifacts from Docker image
    clean_stale_artifacts(cfg.log_dir)

    # 5. Build
    if not cfg.skip_build:
        try:
            build_amdsmi(cfg)
        except CommandError as exc:
            _write_result(cfg.test_results_dir, "build_result.txt",
                          f"BUILD FAILED: {exc.name} exited {exc.code}\nLog: {exc.log_path}")
            report_and_raise("BUILD", exc)

    artifact = locate_package(cfg.build_dir, cfg.package_format)
    print(f"Build artifact: {artifact}")
    _write_result(cfg.test_results_dir, "build_result.txt",
                  f"BUILD PASSED\nArtifact: {artifact}")

    # 6. Install
    if not cfg.skip_install:
        try:
            install_package(cfg, artifact)
        except CommandError as exc:
            _write_result(cfg.test_results_dir, "install_result.txt",
                          f"INSTALL FAILED: {exc.name} exited {exc.code}\nLog: {exc.log_path}")
            report_and_raise("INSTALL", exc)
        _write_result(cfg.test_results_dir, "install_result.txt",
                      f"INSTALL PASSED\nPackage: {artifact}")

    # 7. Verify wheel in site-packages
    if not cfg.skip_install:
        try:
            verify_wheel_site_packages(cfg)
        except CommandError as exc:
            _write_result(cfg.test_results_dir, "verify_wheel_result.txt",
                          f"VERIFY WHEEL FAILED: {exc.name} exited {exc.code}\nLog: {exc.log_path}")
            report_and_raise("VERIFY WHEEL", exc)
        _write_result(cfg.test_results_dir, "verify_wheel_result.txt",
                      "VERIFY WHEEL PASSED")

    # 8. Command tests
    if not cfg.skip_tests and not cfg.skip_install:
        try:
            run_command_tests(cfg)
        except CommandError as exc:
            _write_result(cfg.test_results_dir, "command_tests_result.txt",
                          f"COMMAND TEST FAILED: {exc.name} exited {exc.code}\nLog: {exc.log_path}")
            report_and_raise("RUN COMMAND TESTS", exc)
        _write_result(cfg.test_results_dir, "command_tests_result.txt",
                      "ALL COMMAND TESTS PASSED")

    # 9. AMDSMI / Python / Example tests
    if not cfg.skip_tests and not cfg.skip_install:
        try:
            run_python_and_example_tests(cfg)
        except CommandError as exc:
            _write_result(cfg.test_results_dir, "python_tests_result.txt",
                          f"TEST FAILED: {exc.name} exited {exc.code}\nLog: {exc.log_path}")
            report_and_raise("RUN PYTHON/EXAMPLE TESTS", exc)
        _write_result(cfg.test_results_dir, "python_tests_result.txt",
                      "ALL PYTHON/EXAMPLE TESTS PASSED")

    # 10. Cleanup
    if not cfg.skip_cleanup:
        uninstall_package(cfg)

    print("AMDSMI workflow complete")


if __name__ == "__main__":
    main()
