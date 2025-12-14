#!/usr/bin/env bash
set -e
# Single debug build for stepping in VSCode
rm -rf build-debug
cmake -S cpp -B build-debug -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS="-g -O0 -fno-omit-frame-pointer"
cmake --build build-debug --target test_hnsw_accuracy -j
echo "Built debug test binary: build-debug/test_hnsw_accuracy"