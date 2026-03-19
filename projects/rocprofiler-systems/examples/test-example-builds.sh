#!/bin/bash
# Test standalone builds for each rocprofiler-systems example.
# Run from the examples/ directory.
#
# Usage:
#   cd projects/rocprofiler-systems/examples
#   ./test-example-builds.sh
#
# Optional: set ROCM_PATH for GPU examples (default: /opt/rocm)
#   ROCM_PATH=/opt/rocm-7.2.0 ./test-example-builds.sh

set -e

ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

FAILED=()
PASSED=()

# CPU-only examples (no CMAKE_PREFIX_PATH needed)
CPU_EXAMPLES=(code-coverage rewrite-caller thread-limit parallel-overhead trace-time-window mpi openmp shmem)

# GPU/ROCm examples (need CMAKE_PREFIX_PATH)
GPU_EXAMPLES=(transpose scratch-memory sdma_test transferBench user-api roctx causal fork jpegdecode videodecode rccl lulesh)

# Meta directory (hpc contains multiple sub-examples)
META_EXAMPLES=(hpc)

build_example() {
    local dir="$1"
    local extra_args="${2:-}"
    local build_dir="$SCRIPT_DIR/../build/examples/$dir"
    echo "=== Building $dir ==="
    if (cd "$dir" && cmake -B "$build_dir" $extra_args && cmake --build "$build_dir" --parallel 4); then
        PASSED+=("$dir")
        return 0
    else
        FAILED+=("$dir")
        return 1
    fi
}

echo "Building CPU-only examples..."
for dir in "${CPU_EXAMPLES[@]}"; do
    if [[ -d "$dir" ]]; then
        build_example "$dir" || true
    else
        echo "Skipping $dir (not found)"
    fi
done

echo ""
echo "Building GPU/ROCm examples..."
for dir in "${GPU_EXAMPLES[@]}"; do
    if [[ -d "$dir" ]]; then
        build_example "$dir" "-DCMAKE_PREFIX_PATH=$ROCM_PATH" || true
    else
        echo "Skipping $dir (not found)"
    fi
done

echo ""
echo "Building meta examples (hpc)..."
for dir in "${META_EXAMPLES[@]}"; do
    if [[ -d "$dir" ]]; then
        build_example "$dir" "-DCMAKE_PREFIX_PATH=$ROCM_PATH" || true
    else
        echo "Skipping $dir (not found)"
    fi
done

echo ""
echo "=========================================="
echo "Summary"
echo "=========================================="
echo "Passed: ${#PASSED[@]}"
printf '  %s\n' "${PASSED[@]}"
echo ""
echo "Failed: ${#FAILED[@]}"
printf '  %s\n' "${FAILED[@]}"

if [[ ${#FAILED[@]} -gt 0 ]]; then
    exit 1
fi
