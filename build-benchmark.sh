#!/usr/bin/env bash
set -e

ROOT_DIR="$(pwd)"

# ============================================================
# 1. Release build (benchmarks + tests)
# ============================================================
echo "==================== RELEASE BUILD ===================="

mkdir -p build
cd build

cmake ../cpp \
  -DCMAKE_BUILD_TYPE=Release

cmake --build . --config Release --target all -j

echo "✔ Release build complete."

# ------------------------------------------------------------
# Run unit tests (GoogleTest)
# ------------------------------------------------------------
# if command -v ctest &> /dev/null; then
#     echo "Running unit tests..."
#     ctest --output-on-failure
# else
#     echo "⚠️  CTest not found, skipping tests"
# fi

# ------------------------------------------------------------
# Run benchmarks (Google Benchmark)
# ------------------------------------------------------------
BENCH_BIN="./benchmarks/bench_hnsw"
BENCH_OUT_DIR="${ROOT_DIR}/bench_results"
mkdir -p "${BENCH_OUT_DIR}"

if [[ -f "${BENCH_BIN}" ]]; then
    echo "Running HNSW benchmarks..."

    "${BENCH_BIN}" \
        --benchmark_min_time=0.2 \
        --benchmark_repetitions=5 \
        --benchmark_report_aggregates_only=true \
        --benchmark_out="${BENCH_OUT_DIR}/bench_hnsw.json" \
        --benchmark_out_format=json

    echo "✔ Benchmark results written to:"
    echo "  ${BENCH_OUT_DIR}/bench_hnsw.json"
else
    echo "❌ Benchmark binary not found: ${BENCH_BIN}"
fi

cd "${ROOT_DIR}"

# ============================================================
# 2. Debug + Sanitizer build (optional but recommended)
# ============================================================
echo ""
echo "================ DEBUG + SANITIZER BUILD ================"

rm -rf build-sanitize

cmake -S cpp -B build-sanitize \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -g" \
  -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address,undefined" \
  -DCMAKE_C_FLAGS="-fsanitize=address,undefined -g"

cmake --build build-sanitize -j

echo "✔ Sanitizer build complete."

# ------------------------------------------------------------
# Run sanitizer tests
# ------------------------------------------------------------
echo "Running sanitizer tests..."
ctest --test-dir build-sanitize -j 1 --output-on-failure -V

echo ""
echo "==================== DONE ===================="
