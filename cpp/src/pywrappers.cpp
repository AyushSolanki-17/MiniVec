#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "minivec/dist.hpp"
#include "minivec/pywrappers.hpp"
#include <stdexcept>
#include <limits>
#include <vector>

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
    if (A.size > std::numeric_limits<int>::max())
        throw std::overflow_error("Vector dimension exceeds the supported C++ integer range");
    const float *aptr = static_cast<const float *>(A.ptr);
    const float *bptr = static_cast<const float *>(B.ptr);
    int dim = static_cast<int>(A.size);
    std::vector<float> a_copy, b_copy;
    if (dim > 0) {
        a_copy.assign(aptr, aptr + dim);
        b_copy.assign(bptr, bptr + dim);
    }
    float result = 0.0f;
    // release GIL while computing heavy C++ code
    {
        py::gil_scoped_release release;
        result = minivec::l2_squared_distance(a_copy.data(), b_copy.data(), dim);
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
    if (A.size > std::numeric_limits<int>::max())
        throw std::overflow_error("Vector dimension exceeds the supported C++ integer range");
    const float *aptr = static_cast<const float *>(A.ptr);
    const float *bptr = static_cast<const float *>(B.ptr);
    int dim = static_cast<int>(A.size);
    std::vector<float> a_copy, b_copy;
    if (dim > 0) {
        a_copy.assign(aptr, aptr + dim);
        b_copy.assign(bptr, bptr + dim);
    }
    float result = 0.0f;
    {
        py::gil_scoped_release release;
        result = minivec::l2_distance(a_copy.data(), b_copy.data(), dim);
    }
    return py::float_(result);
}
