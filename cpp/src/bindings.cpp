// cpp/src/bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "minivec/utils.hpp"
#include "minivec/hnsw.hpp"
#include "minivec/pywrappers.hpp"

namespace py = pybind11;

static const float *numpy_to_ptr(py::array_t<float> &arr, int expected_dim)
{
    if (arr.ndim() != 1 || arr.shape(0) != expected_dim)
        throw std::runtime_error("Vector must be 1D float32 with correct dimension.");
    return arr.data();
}
PYBIND11_MODULE(minivec_cpp, m)
{
    m.doc() = "minivec C++ core module";

    m.def("l2_squared", &minivec::py_l2_squared_distance, "Squared L2 distance between two float32 1-D numpy arrays");
    m.def("l2", &minivec::py_l2_distance, "L2 distance (sqrt) between two float32 1-D numpy arrays");
    py::class_<HNSWIndexSimple>(m, "HNSWIndexSimple")
        .def(py::init<int, int, int, int>(),
             py::arg("dim"),
             py::arg("M") = 16,
             py::arg("efConstruction") = 200,
             py::arg("efSearch") = 200)

        .def("insert_vector", [](HNSWIndexSimple &index, py::array_t<float> vec)
             {
                 auto ptr = numpy_to_ptr(vec, index.get_vector_dim());
                 return index.insert_vector(ptr); }, py::arg("vector"))

        .def("search", [](HNSWIndexSimple &index, py::array_t<float> vec, int k)
             {
                 auto ptr = numpy_to_ptr(vec, index.get_vector_dim());
                 auto result = index.search_top_k(ptr, index.get_efSearch(), k);
                 // return Python list of tuples (id, distance)
                 return result; }, py::arg("query"), py::arg("k"))

        .def("get_vector", [](HNSWIndexSimple &index, int id)
             {
                const float* ptr = index.get_vector_ptr(id);
                int dim = index.get_vector_dim();
                py::array_t<float> arr(dim);
                std::memcpy(arr.mutable_data(), ptr, dim * sizeof(float));
                return arr; })
        .def("node_count", &HNSWIndexSimple::get_node_count)
        .def("max_level", &HNSWIndexSimple::get_max_layer)
        .def("entry_point", &HNSWIndexSimple::get_entry_point);
}
