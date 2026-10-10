cmake_minimum_required(VERSION 3.18)
project(minivec_cpp LANGUAGES CXX)

# =========================================================
# Global configuration
# =========================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(MINIVEC_BUILD_PYTHON "Build the pybind11 Python extension" ON)
option(MINIVEC_BUILD_TESTS "Build C++ unit tests" ON)
option(MINIVEC_BUILD_BENCHMARKS "Build Google Benchmark targets" ON)

# Sanitizers: "address" (ASAN + UBSAN), "thread" (TSAN), or empty for none.
# Applied to every target defined below, including fetched test dependencies.
set(MINIVEC_SANITIZE "" CACHE STRING "Sanitizer to build with: address, thread, or empty")
set_property(CACHE MINIVEC_SANITIZE PROPERTY STRINGS "" address thread)
if (MINIVEC_SANITIZE STREQUAL "address")
    set(MINIVEC_SANITIZER_FLAGS -fsanitize=address,undefined -fno-sanitize-recover=undefined)
elseif (MINIVEC_SANITIZE STREQUAL "thread")
    set(MINIVEC_SANITIZER_FLAGS -fsanitize=thread)
elseif (NOT MINIVEC_SANITIZE STREQUAL "")
    message(FATAL_ERROR "MINIVEC_SANITIZE must be address, thread, or empty (got '${MINIVEC_SANITIZE}')")
endif()
if (MINIVEC_SANITIZER_FLAGS)
    if (MSVC)
        message(FATAL_ERROR "MINIVEC_SANITIZE is supported with GCC and Clang only")
    endif()
    add_compile_options(${MINIVEC_SANITIZER_FLAGS} -fno-omit-frame-pointer -g)
    add_link_options(${MINIVEC_SANITIZER_FLAGS})
endif()

# =========================================================
# Core library
# =========================================================
add_library(minivec_core
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
    target_compile_options(minivec_core PRIVATE -O3 -funroll-loops)
endif()

# =========================================================
# Dependencies (FetchContent)
# =========================================================
include(FetchContent)

if (MINIVEC_BUILD_PYTHON)
    # Prefer the build environment's pybind11 (used by Python package builds).
    # Keep FetchContent as a convenient fallback for direct CMake builds.
    find_package(pybind11 CONFIG QUIET)
    if (NOT pybind11_FOUND)
        FetchContent_Declare(
            pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
            GIT_TAG v3.0.1
        )
        FetchContent_MakeAvailable(pybind11)
    endif()
endif()

if (MINIVEC_BUILD_BENCHMARKS)
    # ---- Google Benchmark (performance benchmarks) ----
    FetchContent_Declare(
        benchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.8.3
    )
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(benchmark)
endif()

if (MINIVEC_BUILD_TESTS)
    # ---- GoogleTest (unit & accuracy tests) ----
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.zip
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(googletest)
endif()

# =========================================================
# Python bindings (pybind11 module)
# =========================================================
if (MINIVEC_BUILD_PYTHON)
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
    install(TARGETS minivec_cpp
        LIBRARY DESTINATION .
        RUNTIME DESTINATION .
    )
endif()

# =========================================================
# Tests (GoogleTest)
# =========================================================
if (MINIVEC_BUILD_TESTS)
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
endif()

# =========================================================
# Benchmarks (Google Benchmark)
# =========================================================
if (MINIVEC_BUILD_BENCHMARKS)
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
endif()
