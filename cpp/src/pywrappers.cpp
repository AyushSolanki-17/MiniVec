#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "minivec/dist.hpp"
#include "minivec/pywrappers.hpp"
#include <stdexcept>

namespace py = pybind11;

using np_float1d = py::array_t<float, py::array::c_style | py::array::forcecast>;
py::float_ minivec::py_l2_squared_distance(const np_float1d &a,
                                           const np_float1d &b)
{
    auto A = a.request();
    auto B = b.request();
    if (A.ndim != 1 || B.ndim != 1)
        throw std::runtime_error("Expected 1-D arrays");
    if (A.size != B.size)
        throw std::runtime_error("Size mismatch");
    const float *aptr = static_cast<const float *>(A.ptr);
    const float *bptr = static_cast<const float *>(B.ptr);
    int dim = static_cast<int>(A.size);
    float result = 0.0f;
    // release GIL while computing heavy C++ code
    {
        py::gil_scoped_release release;
        result = minivec::l2_squared_distance(aptr, bptr, dim);
    }

    return py::float_(result);
}

py::float_ minivec::py_l2_distance(const np_float1d &a,
                                   const np_float1d &b)
{
    auto A = a.request();
    auto B = b.request();
    if (A.ndim != 1 || B.ndim != 1)
        throw std::runtime_error("Expected 1-D arrays");
    if (A.size != B.size)
        throw std::runtime_error("Size mismatch");
    const float *aptr = static_cast<const float *>(A.ptr);
    const float *bptr = static_cast<const float *>(B.ptr);
    int dim = static_cast<int>(A.size);
    float result = 0.0f;
    {
        py::gil_scoped_release release;
        result = minivec::l2_distance(aptr, bptr, dim);
    }
    return py::float_(result);
}
