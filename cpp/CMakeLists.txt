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

# Ensure FetchContent is available
include(FetchContent)

# Fetch pybind11 so we don't depend on system install
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v3.0.1   # choose a stable tag you want; update as needed
)
# set to true to use add_subdirectory-like behavior
FetchContent_MakeAvailable(pybind11)

# After FetchContent_MakeAvailable you can use pybind11 targets
# (this avoids needing find_package(pybind11 REQUIRED))

# Find Python and pybind11 in the active environment
find_package(Python COMPONENTS Interpreter Development REQUIRED)
# find_package(pybind11 REQUIRED)

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
enable_testing()

file(GLOB TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp")

foreach(test_src ${TEST_SOURCES})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name} PRIVATE minivec_core gtest gtest_main pthread)
    target_include_directories(${test_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    set_target_properties(${test_name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
    )

    # Discover gtest tests inside the binary and register with CTest
    include(GoogleTest)
    gtest_discover_tests(${test_name}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        PROPERTIES LABELS "unit"
    )
endforeach()



