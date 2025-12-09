/**
 * @file pywrappers.hpp
 * @brief Python wrappers for C++ distance functions
 * 
 * @details 
 * This file defines Python wrappers for C++ distance functions.
 * 
 */

#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace minivec {

namespace py = pybind11;

// Alias for a NumPy float32 1-D C-contiguous array used as vector input
// from Python.
using np_float1d =
    py::array_t<float, py::array::c_style | py::array::forcecast>;

// Computes the squared L2 (Euclidean) distance between two 1-D float arrays.
//
// Both inputs must be 1-D, C-contiguous float32 NumPy arrays of equal length.
// Implementations are expected to validate shapes and raise on mismatch.
//
// Args:
//   a: First input vector.
//   b: Second input vector.
//
// Returns:
//   Squared L2 distance between a and b as a Python float.
py::float_ py_l2_squared_distance(const np_float1d& a, const np_float1d& b);

// Computes the L2 (Euclidean) distance between two 1-D float arrays.
//
// Both inputs must be 1-D, C-contiguous float32 NumPy arrays of equal length.
// Implementations are expected to validate shapes and raise on mismatch.
//
// Args:
//   a: First input vector.
//   b: Second input vector.
//
// Returns:
//   L2 distance between a and b as a Python float.
py::float_ py_l2_distance(const np_float1d& a, const np_float1d& b);

}  // namespace minivec