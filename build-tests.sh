#!/usr/bin/env bash
set -e

# Create and enter build folder
mkdir -p build
cd build

# Configure the project
cmake ../cpp -DCMAKE_BUILD_TYPE=Release

# Build the project, including tests
cmake --build . --config Release --target all

echo "Build complete."

# Run the tests using Google Test
if command -v ctest &> /dev/null; then
    echo "Running tests..."
    ctest --output-on-failure
else
    echo "No CTest found. If using gtest, you can run the test binaries manually."
fi
