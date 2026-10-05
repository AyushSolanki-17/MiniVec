/**
 * @file vecstore.hpp
 * @brief Declaration of vector storage for HNSW index
 * @details
 * This file defines the VecStore class, which is used to store vectors in a
 * contiguous array in memory.
 */
#pragma once

#include <cassert>
#include <deque>
#include <vector>

namespace minivec {
// Stores fixed-dimensional vectors in stable, independently contiguous blocks.
//
// Vectors are appended sequentially and addressed by an integer id in
// [0, size()). Each vector has the same dimensionality `dim`; appending does
// not invalidate pointers to vectors already stored.
struct VecStore {
  std::deque<std::vector<float>> data;
  int dim = 0;

  // Creates an empty store with vectors of dimension dim_.
  explicit VecStore(int dim_ = 0) : dim(dim_) {}

  // Returns the number of stored vectors.
  inline int size() const { return static_cast<int>(data.size()); }

  // Adds a vector to the store.
  //
  // Args:
  //   vals: Pointer to an array of `dim` floats.
  //
  // Returns:
  //   The id of the newly added vector.
  inline int add(const float* vals) {
    assert(dim > 0);
    int id = size();
    data.emplace_back(vals, vals + dim);
    return id;
  }

  // Returns a const pointer to the vector with the given id.
  inline const float* ptr(int id) const {
    return data[static_cast<size_t>(id)].data();
  }

  // Returns a mutable pointer to the vector with the given id.
  inline float* ptr_mut(int id) {
    return data[static_cast<size_t>(id)].data();
  }

  // Removes all stored vectors.
  inline void clear() { data.clear(); }
};
}  // namespace minivec
