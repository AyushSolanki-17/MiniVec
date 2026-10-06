/**
 * @file vecstore.hpp
 * @brief Declaration of vector storage for HNSW index
 * @details
 * This file defines the VecStore class, which keeps each fixed-dimensional
 * vector in a stable contiguous block and indexes blocks with 64-bit IDs.
 */
#pragma once

#include <deque>
#include <vector>
#include <cstdint>
#include <stdexcept>

namespace minivec {
// Stores fixed-dimensional vectors in stable, independently contiguous blocks.
//
// Vectors are appended sequentially and addressed by a signed 64-bit id in
// [0, size()). Each vector has the same dimensionality `dim`; appending does
// not invalidate pointers to vectors already stored.
struct VecStore {
  std::deque<std::vector<float>> data;
  int dim = 0;

  // Creates an empty store with vectors of dimension dim_.
  explicit VecStore(int dim_ = 0) : dim(dim_) {}

  // Returns the number of stored vectors.
  inline std::int64_t size() const { return static_cast<std::int64_t>(data.size()); }

  // Adds a vector to the store.
  //
  // Args:
  //   vals: Pointer to an array of `dim` floats.
  //
  // Returns:
  //   The id of the newly added vector.
  inline std::int64_t add(const float* vals) {
    if (dim <= 0)
      throw std::logic_error("VecStore: dimension must be positive before adding vectors");
    if (!vals)
      throw std::invalid_argument("VecStore: vector pointer must not be null");
    std::int64_t id = size();
    data.emplace_back(vals, vals + dim);
    return id;
  }

  // Returns a const pointer to the vector with the given id.
  inline const float* ptr(std::int64_t id) const {
    return data.at(static_cast<std::size_t>(id)).data();
  }

  // Returns a mutable pointer to the vector with the given id.
  inline float* ptr_mut(std::int64_t id) {
    return data.at(static_cast<std::size_t>(id)).data();
  }

  // Removes all stored vectors.
  inline void clear() { data.clear(); }

  inline void pop_back() { data.pop_back(); }
};
}  // namespace minivec
