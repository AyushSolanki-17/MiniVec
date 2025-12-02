#pragma once

#include <pybind11/numpy.h>

namespace py = pybind11;

// Compute L2 distance between two 1-D float32 numpy arrays.
// Throws runtime_error on shape mismatch.
float l2_distance(py::array_t<float> a, py::array_t<float> b);

// Normalize a float32 1-D numpy array in-place (L2 norm).
void normalize_inplace(py::array_t<float> a);
