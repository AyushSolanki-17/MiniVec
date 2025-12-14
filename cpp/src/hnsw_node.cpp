/**
 * @file hnsw_node.cpp
 * @brief Implementation of HNSW node functionality
 * @details
 * This file implements the HNSWNodeSimple class, which represents a node in the HNSW graph.
 */
#include "minivec/hnsw_node.hpp"
#include <iostream>
#include <thread>

// Initializes a node with the given id, number of layers, and neighbor capacity.
//
// Args:
//   _id: Node identifier.
//   layers: Total number of layers for this node (must be >= 1).
//   M: Expected maximum number of neighbors per layer.
HNSWNodeSimple::HNSWNodeSimple(int _id, int layers, int M)
    : id(_id),  layer(layers > 0 ? layers - 1 : 0), neighbors()
{
    if (layers < 1)
    {
        throw std::invalid_argument("HNSWNodeSimple: layers must be >= 1");
    }
    // Allocate exactly `layers` adjacency lists: neighbors[0] .. neighbors[layers-1]
    neighbors.resize(layers);
    for (std::vector<int> &v : neighbors)
        v.reserve(std::max(0, M + 2));
}

// Returns the node id.
const int HNSWNodeSimple::get_id() const
{
    return id;
}


// Returns the highest layer index for this node.
int HNSWNodeSimple::get_layer() const
{
    return layer;
}

// Throws an exception if the given layer is out of bounds.
void HNSWNodeSimple::check_layer_bounds_or_throw(int t_layer) const
{
    if (layer < 0 || t_layer > layer)
    {
        std::ostringstream oss;
        oss << "HNSWNodeSimple: layer out of bounds: " << t_layer << " (layer=" << layer << ")";
        throw std::out_of_range(oss.str());
    }
}

// Returns the neighbor list for a given layer.
//
// Args:
//   layer: Layer index in [0, layer].
//
// Returns:
//   Vector of neighbor ids.
const std::vector<int> HNSWNodeSimple::get_neighbors(int t_layer) const
{
    // check_layer_bounds_or_throw(layer);
    std::shared_lock lock(mtx);
    if (t_layer < 0 || t_layer > layer) {
        static const std::vector<int> EMPTY_VEC;
        return {};
  }
  if (neighbors.size()==0)
  {
      static const std::vector<int> EMPTY_VEC;
      return {};
  }
  
  return neighbors[t_layer];
}

bool HNSWNodeSimple::add_neighbor_nolock(int id, int layer, int *out_index)
{
    // caller must have locked mtx(exclusive)
    check_layer_bounds_or_throw(layer);
    std::vector<int> &vec = neighbors[layer];
    auto it = std::find(vec.begin(), vec.end(), id);
    if (it != vec.end()) {
        if (out_index) *out_index = static_cast<int>(it - vec.begin());
        return false;
    }
    vec.push_back(id);
    if (out_index) *out_index = static_cast<int>(vec.size()) - 1;
    return true;
}

bool HNSWNodeSimple::remove_neighbor_nolock(int id, int layer)
{
    check_layer_bounds_or_throw(layer);
    std::vector<int> &vec = neighbors[layer];
    auto it = std::find(vec.begin(), vec.end(), id);
    if (it == vec.end()) return false;
    vec.erase(it);
    return true;
}


// Adds a neighbor at the given layer.
//
// Args:
//   id: Neighbor node id to add.
//   layer: Layer index where the neighbor is added.
//
// Returns:
//   Index at which the neighbor was inserted.
bool HNSWNodeSimple::add_neighbor(int id, int layer, int *out_index)
{
    check_layer_bounds_or_throw(layer);
    std::unique_lock lock(mtx);
    return add_neighbor_nolock(id, layer, out_index);
}

// Removes a neighbor from the given layer, if present.
//
// Args:
//   id: Neighbor node id to remove.
//   layer: Layer index to remove from.
//
// Returns:
//   1 if the neighbor was removed, 0 if it was not found.
bool HNSWNodeSimple::remove_neighbor(int id, int layer, bool preserve_order)
{
    check_layer_bounds_or_throw(layer);
    std::unique_lock lock(mtx);
    return remove_neighbor_nolock(id, layer);
    // std::vector<int> &vec = neighbors[layer];
    // std::vector<int>::iterator it = std::find(vec.begin(), vec.end(), id);
    // if (it == vec.end())
    //     return false;

    // size_t idx = it - vec.begin();

    // if (!preserve_order)
    // {
    //     // swap-with-last and pop (O(1))
    //     size_t last = vec.size() - 1;
    //     if (idx != last)
    //     {
    //         vec[idx] = vec[last];
    //     }
    //     vec.pop_back();
    // }
    // else
    // {
    //     // preserve order (O(n))
    //     vec.erase(it);
    // }
    // return true;
}


// Returns true if the given neighbor id is present in the given layer.
//
// Args:
//   id: Neighbor node id to check.
//   layer: Layer index to check.
//
// Returns:
//   True if the neighbor is present, false otherwise.
bool HNSWNodeSimple::has_neighbor(int id, int layer) const {
  check_layer_bounds_or_throw(layer);
  std::shared_lock lock(mtx);
  const auto &vec = neighbors[layer];
  return std::find(vec.begin(), vec.end(), id) != vec.end();
}

// Reserves capacity for the given layer.
//
// Args:
//   layer: Layer index to reserve.
//   capacity: Minimum capacity to reserve.
void HNSWNodeSimple::reserve_layer(int layer, size_t capacity) {
  check_layer_bounds_or_throw(layer);
  std::unique_lock lock(mtx);
  neighbors[layer].reserve(capacity);
}

// Clears the given layer.
//
// Args:
//   layer: Layer index to clear.
void HNSWNodeSimple::clear_layer(int layer) {
  check_layer_bounds_or_throw(layer);
  std::unique_lock lock(mtx);
  neighbors[layer].clear();
}