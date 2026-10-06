#!/usr/bin/env bash
set -e
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Single debug build for stepping in VSCode
cmake -S "${ROOT_DIR}/cpp" -B "${ROOT_DIR}/build-debug" -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS="-g -O0 -fno-omit-frame-pointer"
cmake --build "${ROOT_DIR}/build-debug" --target test_hnsw_accuracy -j
echo "Built debug test binary: ${ROOT_DIR}/build-debug/tests/test_hnsw_accuracy"
