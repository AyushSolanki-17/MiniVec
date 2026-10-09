/**
 * @file bindings.cpp
 * @brief Python bindings for the HNSW index
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <cstring>
#include <vector>

#include "minivec/utils.hpp"
#include "minivec/hnsw.hpp"
#include "minivec/pywrappers.hpp"

namespace py = pybind11;

template <typename Array>
static const float *numpy_to_ptr(Array &arr, int expected_dim)
{
    if (arr.ndim() != 1 || arr.shape(0) != expected_dim)
        throw std::runtime_error("Vector must be 1D float32 with correct dimension.");
    return arr.data();
}

PYBIND11_MODULE(minivec_cpp, m)
{
    m.doc() = "minivec C++ core module (HNSW)";

    // ---------------- Distance helpers ----------------
    m.def("l2_squared", &minivec::py_l2_squared_distance,
          "Squared L2 distance between two float32 numpy vectors");

    m.def("l2", &minivec::py_l2_distance,
          "L2 distance between two float32 numpy vectors");

    // ---------------- SearchStats binding ----------------
    py::class_<minivec::SearchStats>(m, "SearchStats")
        .def(py::init<>())
        .def_readwrite("visited_nodes", &minivec::SearchStats::visited_nodes)
        .def_readwrite("distance_calls", &minivec::SearchStats::distance_calls)
        .def_readwrite("layer_visits", &minivec::SearchStats::layer_visits);

    // ---------------- HNSWIndexSimple binding ----------------
    py::class_<minivec::HNSWIndexSimple>(m, "HNSWIndexSimple")
        .def(py::init<
                 int, int, int, int,
                 bool, uint32_t,
                 const std::string &,
                 const std::string &>(),
             py::arg("dim"),
             py::arg("M") = 16,
             py::arg("efConstruction") = 200,
             py::arg("efSearch") = 200,
             py::arg("deterministic_levelgen") = false,
             py::arg("levelgen_seed") = 42,
             py::arg("distance") = "l2_squared",
             py::arg("final_distance") = "l2")

        // -------- insertion --------
        .def("insert_vector", [](minivec::HNSWIndexSimple &index, py::array_t<float, py::array::c_style | py::array::forcecast> vec)
             {
                 auto ptr = numpy_to_ptr(vec, index.get_vector_dim());
                 std::vector<float> owned(ptr, ptr + index.get_vector_dim());
                 minivec::NodeId id;
                 { py::gil_scoped_release release; id = index.insert_vector(owned.data()); }
                 return id; }, py::arg("vector"))

        .def("insert_vectors", [](minivec::HNSWIndexSimple &index, py::array_t<float, py::array::c_style | py::array::forcecast> vectors)
             {
                 if (vectors.ndim() != 2 || vectors.shape(1) != index.get_vector_dim())
                     throw py::value_error("Vectors must be a 2D array with the index dimension.");
                 const auto rows = vectors.shape(0);
                 const auto dim = vectors.shape(1);
                 const float *data = vectors.data();
                 std::vector<float> owned;
                 if (rows > 0)
                     owned.assign(data, data + rows * dim);
                 std::vector<minivec::NodeId> ids;
                 ids.reserve(static_cast<size_t>(rows));
                 {
                     py::gil_scoped_release release;
                     for (py::ssize_t row = 0; row < rows; ++row)
                         ids.push_back(index.insert_vector(owned.data() + row * dim));
                 }
                 return ids; }, py::arg("vectors"))

        // -------- search (no stats) --------
        .def("search", [](minivec::HNSWIndexSimple &index, py::array_t<float, py::array::c_style | py::array::forcecast> vec, int k)
             {
                auto ptr = numpy_to_ptr(vec, index.get_vector_dim());
                std::vector<float> owned(ptr, ptr + index.get_vector_dim());
                std::vector<minivec::Candidate> results;
                { py::gil_scoped_release release; results = index.search_top_k(owned.data(), index.get_efSearch(), k); }

                py::list out;
                for (const auto &c : results)
                {
                    out.append(py::make_tuple(c.id, c.distance));
                }
                return out; }, py::arg("query"), py::arg("k"))

        // -------- search (with stats) --------
        .def("search_with_stats", [](minivec::HNSWIndexSimple &index, py::array_t<float, py::array::c_style | py::array::forcecast> vec, int k)
             {
         auto ptr = numpy_to_ptr(vec, index.get_vector_dim());
         std::vector<float> owned(ptr, ptr + index.get_vector_dim());

         minivec::SearchStats stats;
         std::vector<minivec::Candidate> results;
         { py::gil_scoped_release release; results = index.search_top_k(
             owned.data(), index.get_efSearch(), k, &stats); }

         py::list out;
         for (const auto &c : results)
         {
             out.append(py::make_tuple(c.id, c.distance));
         }

         py::dict stats_dict;
         stats_dict["visited_nodes"] = stats.visited_nodes;
         stats_dict["distance_calls"] = stats.distance_calls;
         stats_dict["layer_visits"] = stats.layer_visits;

         return py::make_tuple(out, stats_dict); }, py::arg("query"), py::arg("k"))

        .def("search_batch", [](minivec::HNSWIndexSimple &index, py::array_t<float, py::array::c_style | py::array::forcecast> queries, int k)
             {
                 if (queries.ndim() != 2 || queries.shape(1) != index.get_vector_dim())
                     throw py::value_error("Queries must be a 2D array with the index dimension.");
                 const auto rows = queries.shape(0);
                 const auto dim = queries.shape(1);
                 const float *data = queries.data();
                 std::vector<float> owned;
                 if (rows > 0)
                     owned.assign(data, data + rows * dim);
                 std::vector<std::vector<std::pair<minivec::NodeId, float>>> results;
                 results.reserve(static_cast<size_t>(rows));
                 {
                     py::gil_scoped_release release;
                     for (py::ssize_t row = 0; row < rows; ++row) {
                         auto row_results = index.search_top_k(owned.data() + row * dim, index.get_efSearch(), k);
                         std::vector<std::pair<minivec::NodeId, float>> converted;
                         converted.reserve(row_results.size());
                         for (const auto &candidate : row_results)
                             converted.emplace_back(candidate.id, candidate.distance);
                         results.push_back(std::move(converted));
                     }
                 }
                 return results; }, py::arg("queries"), py::arg("k"))

        // -------- getters --------
        .def("node_count", &minivec::HNSWIndexSimple::get_node_count)
        .def("max_level", &minivec::HNSWIndexSimple::get_max_layer)
        .def("entry_point", &minivec::HNSWIndexSimple::get_entry_point)
        .def("vector_dim", &minivec::HNSWIndexSimple::get_vector_dim)
        .def("M", &minivec::HNSWIndexSimple::get_M)
        .def("efConstruction", &minivec::HNSWIndexSimple::get_efConstruction)
        .def("efSearch", &minivec::HNSWIndexSimple::get_efSearch)
        .def("set_ef_search", &minivec::HNSWIndexSimple::set_efSearch,
             py::arg("ef_search"))

        // -------- vector access --------
        .def("get_vector", [](minivec::HNSWIndexSimple &index, minivec::NodeId id)
             {
                 const float *ptr = index.get_vector_ptr(id);
                 int dim = index.get_vector_dim();
                 py::array_t<float> arr(dim);
                 std::memcpy(arr.mutable_data(), ptr, dim * sizeof(float));
                 return arr; });
}
