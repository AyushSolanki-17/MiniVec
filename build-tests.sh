#!/usr/bin/env bash
set -e

# ------------------ Release Build ------------------
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

cd ..

# ------------------ Debug + Sanitizer Build ------------------
echo "Creating debug + sanitizer build..."

rm -rf build-sanitize
cmake -S . -B build-sanitize -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -g" \
  -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address,undefined" \
  -DCMAKE_C_FLAGS="-fsanitize=address,undefined -g"

cmake --build build-sanitize -j

# Run tests (verbose) - this runs with sanitizer
echo "Running sanitizer build tests..."
ctest --test-dir build-sanitize -j 1 --output-on-failure -V