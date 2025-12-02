#include "utils.hpp"
#include <cmath>
#include <stdexcept>

float l2_distance(py::array_t<float> a, py::array_t<float> b) {
    // Request unchecked access (fast) and check dims
    auto ra = a.unchecked<1>();
    auto rb = b.unchecked<1>();

    if (ra.size() != rb.size()) {
        throw std::runtime_error("l2_distance: dimension mismatch");
    }

    double sum = 0.0;
    const ssize_t N = ra.size();
    for (ssize_t i = 0; i < N; ++i) {
        double d = static_cast<double>(ra(i)) - static_cast<double>(rb(i));
        sum += d * d;
    }
    return static_cast<float>(std::sqrt(sum));
}

void normalize_inplace(py::array_t<float> a) {
    auto ra = a.mutable_unchecked<1>();
    const ssize_t N = ra.size();
    double sumsq = 0.0;
    for (ssize_t i = 0; i < N; ++i) {
        double v = ra(i);
        sumsq += v * v;
    }
    double norm = std::sqrt(sumsq);
    if (norm <= 1e-12) return; // avoid divide by zero
    for (ssize_t i = 0; i < N; ++i) {
        ra(i) = static_cast<float>(ra(i) / norm);
    }
}
