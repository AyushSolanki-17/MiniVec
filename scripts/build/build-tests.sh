#!/usr/bin/env bash
set -e
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# ------------------ Release Build ------------------
# Create and enter build folder
mkdir -p "${ROOT_DIR}/build"

# Configure the project
cmake -S "${ROOT_DIR}/cpp" -B "${ROOT_DIR}/build" -DCMAKE_BUILD_TYPE=Release

# Build the project, including tests
cmake --build "${ROOT_DIR}/build" --config Release --target all

echo "Build complete."

# Run the tests using Google Test
if command -v ctest &> /dev/null; then
    echo "Running tests..."
    ctest --test-dir "${ROOT_DIR}/build" --output-on-failure
else
    echo "No CTest found. If using gtest, you can run the test binaries manually."
fi


# ------------------ Debug + Sanitizer Build ------------------
echo "Creating debug + sanitizer build..."

rm -rf "${ROOT_DIR}/build-sanitize"
cmake -S "${ROOT_DIR}/cpp" -B "${ROOT_DIR}/build-sanitize" -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -g" \
  -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address,undefined" \
  -DCMAKE_C_FLAGS="-fsanitize=address,undefined -g"

cmake --build "${ROOT_DIR}/build-sanitize" -j

# Run tests (verbose) - this runs with sanitizer
echo "Running sanitizer build tests..."
ctest --test-dir "${ROOT_DIR}/build-sanitize" -j 1 --output-on-failure -V
