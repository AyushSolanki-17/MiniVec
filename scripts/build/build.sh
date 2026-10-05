#!/usr/bin/env bash
set -e
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cmake -S "${ROOT_DIR}/cpp" -B "${ROOT_DIR}/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "${ROOT_DIR}/build" --config Release
echo "Build complete: ${ROOT_DIR}/build"
