// cpp/src/bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "utils.hpp"
#include "hnsw.hpp"   // header exists as skeleton (can be empty for now)

namespace py = pybind11;

// if you implemented a free l2_distance in utils.cpp, bind it here
PYBIND11_MODULE(minivec_cpp, m) {
    m.doc() = "minivec C++ core module (l2_distance, HNSW skeleton)";

    m.def("l2_distance",
          &l2_distance,
          "Compute L2 distance between two 1D float32 numpy arrays");

    // Expose HNSWIndex skeleton so Python can construct it (methods added later)
    // py::class_<HNSWIndex>(m, "HNSWIndex")
    //     .def(py::init<int>(), py::arg("M") = 16)
    //     .def("random_level", &HNSWIndex::random_level)
    //     .def_readonly("entry_point", &HNSWIndex::entry_point)
    //     .def_readonly("max_level", &HNSWIndex::max_level);
}
