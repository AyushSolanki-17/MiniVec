cmake_minimum_required(VERSION 3.18)
project(minivec_cpp LANGUAGES CXX)

# =========================================================
# Global configuration
# =========================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# =========================================================
# Core library
# =========================================================
add_library(minivec_core
    src/utils.cpp
    src/dist.cpp
    src/hnsw_node.cpp
    src/hnsw.cpp
)

target_include_directories(minivec_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Optimizations for core library only
if (MSVC)
    target_compile_options(minivec_core PRIVATE /O2 /EHsc)
else()
    target_compile_options(minivec_core PRIVATE -O3 -march=native -funroll-loops)
endif()

# =========================================================
# Dependencies (FetchContent)
# =========================================================
include(FetchContent)

# ---- pybind11 (for Python bindings) ----
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v3.0.1
)
FetchContent_MakeAvailable(pybind11)

# ---- Google Benchmark (performance benchmarks) ----
FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.8.3
)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(benchmark)

# ---- GoogleTest (unit & accuracy tests) ----
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.zip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(googletest)

# =========================================================
# Python bindings (pybind11 module)
# =========================================================
find_package(Python COMPONENTS Interpreter Development REQUIRED)

pybind11_add_module(minivec_cpp
    src/bindings.cpp
    src/pywrappers.cpp
)

target_link_libraries(minivec_cpp
    PRIVATE
        minivec_core
)

target_include_directories(minivec_cpp
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

message(STATUS "Using Python: ${Python_EXECUTABLE}")

# =========================================================
# Tests (GoogleTest)
# =========================================================
enable_testing()
include(GoogleTest)

file(GLOB TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp")

foreach(test_src ${TEST_SOURCES})
    get_filename_component(test_name ${test_src} NAME_WE)

    add_executable(${test_name} ${test_src})

    target_link_libraries(${test_name}
        PRIVATE
            minivec_core
            gtest
            gtest_main
            pthread
    )

    target_include_directories(${test_name}
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    set_target_properties(${test_name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
    )

    gtest_discover_tests(${test_name}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        PROPERTIES LABELS "unit"
    )
endforeach()

# =========================================================
# Benchmarks (Google Benchmark)
# =========================================================
add_executable(bench_hnsw
    benchmarks/bench_hnsw_search.cpp
)

target_link_libraries(bench_hnsw
    PRIVATE
        minivec_core
        benchmark::benchmark
        pthread
)

set_target_properties(bench_hnsw PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/benchmarks"
)
