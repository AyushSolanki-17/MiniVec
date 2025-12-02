#!/usr/bin/env bash
set -e
mkdir -p build
cd build
cmake ../cpp
cmake --build . --config Release
echo "Built. To test, set PYTHONPATH to $(pwd) and run python tests"
