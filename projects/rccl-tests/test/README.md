# RCCL-Tests Unit Test Suite

## Contents

- [Prerequisites](#prerequisites)
- [Running Tests](#running-tests)
- [Message Size Profiles](#message-size-profiles)
- [Test Markers](#test-markers)
- [Filtering Tests](#filtering-tests)
- [Collectives Covered](#collectives-covered)
- [Additional Options](#additional-options)

## Prerequisites

1. Install Python test dependencies:
   ```shell
   pip install -r requirements.txt
   ```

2. Set `LD_LIBRARY_PATH` to include the RCCL library:
   ```shell
   export LD_LIBRARY_PATH=/path/to/rccl-install/lib:$LD_LIBRARY_PATH
   ```

3. Set `PATH` and `LD_LIBRARY_PATH` to include MPI:
   ```shell
   export PATH=/path/to/mpi-install/bin:$PATH
   export LD_LIBRARY_PATH=/path/to/mpi-install/lib:$LD_LIBRARY_PATH
   ```

4. The perf binaries must be built first (see the main [README.md](../README.md) for build instructions). The test runner expects them in `../build/` relative to the `test/` directory.

## Running Tests

Run all non-MPI tests with the default smoke profile (1K-1G):
```shell
python3 -m pytest
```

Run MPI tests only:
```shell
python3 -m pytest -m mpi
```

Exclude MPI tests:
```shell
python3 -m pytest -m "not mpi"
```

## Message Size Profiles

The `--msg-profile` flag controls which byte ranges, operations, datatypes, and memory types are tested. This allows the same test files to serve both quick CI validation and thorough nightly sweeps.

### `smoke` (default)

- **Byte ranges**: 1K - 1G
- **Operations**: sum (where applicable)
- **Datatypes**: float, half, bfloat16, fp8_e5m2
- **Memory types**: coarse

Intended for CI precheckin -- quick validation across all collectives with limited parameter combinations.

```shell
python3 -m pytest --msg-profile=smoke
```

### `stress`

- **Byte ranges**: 4 - 1K, 1K - 1M, 1M - 4G (three separate sweeps)
- **Operations**: sum, prod, min, max, avg, mulsum
- **Datatypes**: int8, uint8, int32, uint32, int64, uint64, half, float, double, bfloat16, fp8_e4m3, fp8_e5m2
- **Memory types**: coarse, fine, host, managed

Intended for nightly CI -- full coverage of all parameter combinations and large message sizes.

```shell
python3 -m pytest --msg-profile=stress
```

## Test Markers

| Marker | Description |
|--------|-------------|
| `mpi`  | Tests requiring MPI runtime |

Examples:
```shell
python3 -m pytest -m mpi              # MPI tests only
python3 -m pytest -m "not mpi"        # non-MPI tests only
```

## Filtering Tests

Use `-k` for keyword-based filtering against test names:

```shell
# Run smoke tests but exclude allreduce_bias
python3 -m pytest -k "not allreduce_bias"

# Exclude allreduce_bias and MPI tests
python3 -m pytest -k "not allreduce_bias and not mpi"

# Only run allreduce tests (includes allreduce_bias)
python3 -m pytest -k "allreduce"

# Only run allreduce but not allreduce_bias
python3 -m pytest -k "allreduce and not bias"
```

Use `--ignore` to exclude by file:

```shell
python3 -m pytest --ignore=test_allreduce_bias.py
```

## Collectives Covered

| Test file | Perf binary |
|-----------|-------------|
| test_allgather.py | all_gather_perf |
| test_allreduce.py | all_reduce_perf |
| test_allreduce_bias.py | all_reduce_bias_perf |
| test_alltoall.py | alltoall_perf |
| test_alltoallv.py | alltoallv_perf |
| test_broadcast.py | broadcast_perf |
| test_gather.py | gather_perf |
| test_hypercube.py | hypercube_perf |
| test_reduce.py | reduce_perf |
| test_reducescatter.py | reduce_scatter_perf |
| test_scatter.py | scatter_perf |
| test_sendrecv.py | sendrecv_perf |

## Additional Options

| Option | Description |
|--------|-------------|
| `--test-timeout=600` | Per-test subprocess timeout in seconds (default: 300) |
| `--hostfile=hosts.txt` | MPI hostfile |
| `--junitxml=testreport.xml` | JUnit XML report for CI |
| `--html=report.html --self-contained-html` | HTML report with embedded stdout/stderr |
| `--json-report --json-report-file=report.json` | Machine-readable JSON report |
| `-v --tb=short` | Verbose output with short tracebacks (enabled by default in pytest.ini) |
