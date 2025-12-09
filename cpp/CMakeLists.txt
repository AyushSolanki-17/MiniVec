cmake_minimum_required(VERSION 3.18)
project(minivec_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(minivec_core
    src/utils.cpp
    src/dist.cpp
    src/hnsw_node.cpp
    src/hnsw.cpp
)
target_include_directories(minivec_core PUBLIC include)

# Find Python and pybind11 in the active environment
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

# Source files (edit names if yours differ)
set(minivec_cpp
    src/bindings.cpp
    src/pywrappers.cpp
    src/utils.cpp
    src/dist.cpp
    src/hnsw_node.cpp
    src/hnsw.cpp
    # src/hnsw_ef.cpp
)

# Create the pybind11 module first
pybind11_add_module(minivec_cpp ${minivec_cpp})

# Now add include directories (after the target exists)
target_link_libraries(minivec_cpp PRIVATE minivec_core)
target_include_directories(minivec_cpp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Optimization flags
if (MSVC)
  target_compile_options(minivec_core PRIVATE /O2 /EHsc)
else()
  target_compile_options(minivec_core PRIVATE -O3 -march=native -funroll-loops)
endif()

# Print helpful info
message(STATUS "CMAKE_CURRENT_SOURCE_DIR = ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "Using Python: ${Python_EXECUTABLE}")

# Test executable
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.zip
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(googletest)

# add_executable(test_hnsw tests/test_hnsw.cpp)
# target_link_libraries(test_hnsw PRIVATE minivec_core gtest gtest_main pthread)
# target_include_directories(test_hnsw PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
# add_test(NAME HNSWTest COMMAND )
# add_executable(test_hnsw tests/test_hnsw.cpp)
# target_link_libraries(test_hnsw PRIVATE minivec_core gtest gtest_main pthread)
# target_include_directories(test_hnsw PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
enable_testing()  # ensure testing is enabled

add_executable(test_hnsw tests/test_hnsw.cpp)
target_link_libraries(test_hnsw PRIVATE minivec_core gtest gtest_main pthread)
target_include_directories(test_hnsw PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Let CTest discover the gtest tests
include(GoogleTest)
gtest_discover_tests(test_hnsw)


