cmake_minimum_required(VERSION 3.18)
project(minivec_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find Python and pybind11 in the active environment
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

# Source files (edit names if yours differ)
set(SRC
    src/bindings.cpp
    src/utils.cpp
    # src/hnsw.cpp
    # src/hnsw_ef.cpp
)

# Create the pybind11 module first
pybind11_add_module(minivec_cpp ${SRC})

# Now add include directories (after the target exists)
target_include_directories(minivec_cpp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Optimization flags
if (MSVC)
  target_compile_options(minivec_cpp PRIVATE /O2 /EHsc)
else()
  target_compile_options(minivec_cpp PRIVATE -O3 -march=native -funroll-loops)
endif()

# Print helpful info
message(STATUS "CMAKE_CURRENT_SOURCE_DIR = ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "Using Python: ${Python_EXECUTABLE}")