#!/bin/bash

# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Format all C++ source files using clang-format.
# Run this to ensure all source files follow the project's .clang-format rules.
#
# Usage:
#   ./clang_format.sh [directory]
#
# If no directory is given, defaults to the project's source tree.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SRC_DIR="${1:-$PROJECT_ROOT/lib}"

if ! command -v clang-format &>/dev/null; then
  echo "error: clang-format not found in PATH" >&2
  exit 1
fi

# Find all .cpp and .h files under the source tree.
files=$(find "$SRC_DIR" -type f \( -name '*.cpp' -o -name '*.h' \))

if [ -z "$files" ]; then
  echo "No C++ files found in $SRC_DIR"
  exit 0
fi

count=0
for f in $files; do
  clang-format -i -style=file:"$PROJECT_ROOT/.clang-format" "$f"
  count=$((count + 1))
done

echo "Formatted $count files in $SRC_DIR"
