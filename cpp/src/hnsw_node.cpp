/**
 * @file hnsw_node.cpp
 * @brief Implementation of HNSW node functionality
 * @details
 * This file implements the HNSWNodeSimple class, which represents a node in the HNSW graph.
 */
#include "minivec/hnsw_node.hpp"
#include <iostream>
#include <thread>
namespace minivec
{
// Initializes a node with the given id, number of layers, and neighbor capacity.
//
// Args:
//   _id: Node identifier.
//   layers: Total number of layers for this node (must be >= 1).
//   M: Expected maximum number of neighbors per layer.
HNSWNodeSimple::HNSWNodeSimple(NodeId _id, int layers, int M)
    : id(_id), layer(layers > 0 ? layers - 1 : 0),
      layer_offsets(), layer_sizes()
{
    if (layers < 1)
    {
        throw std::invalid_argument("HNSWNodeSimple: layers must be >= 1");
    }
    layer_offsets.resize(static_cast<size_t>(layers) + 1, 0);
    layer_sizes.resize(static_cast<size_t>(layers), 0);
    // Reserve the common base-layer degree; all layers share one packed buffer.
    neighbor_ids.reserve(static_cast<size_t>(std::max(M, 0)) * 2 + 2);
}

// Returns the node id.
NodeId HNSWNodeSimple::get_id() const
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
    if (t_layer < 0 || layer < 0 || t_layer > layer)
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
const std::vector<NodeId> HNSWNodeSimple::get_neighbors(int t_layer) const
{
    std::shared_lock lock(mtx);
    check_layer_bounds_or_throw(t_layer);
    const size_t begin = layer_offsets[t_layer];
    const size_t end = begin + layer_sizes[t_layer];
    return {neighbor_ids.begin() + begin, neighbor_ids.begin() + end};
}

bool HNSWNodeSimple::add_neighbor_nolock(NodeId id, int layer, int *out_index)
{
    // caller must have locked mtx(exclusive)
    check_layer_bounds_or_throw(layer);
    const size_t begin = layer_offsets[layer];
    const size_t end = begin + layer_sizes[layer];
    auto it = std::find(neighbor_ids.begin() + begin, neighbor_ids.begin() + end, id);
    if (it != neighbor_ids.begin() + end) {
        if (out_index) *out_index = static_cast<int>(it - (neighbor_ids.begin() + begin));
        return false;
    }
    neighbor_ids.insert(neighbor_ids.begin() + end, id);
    ++layer_sizes[layer];
    for (size_t i = static_cast<size_t>(layer) + 1; i < layer_offsets.size(); ++i)
        ++layer_offsets[i];
    if (out_index) *out_index = static_cast<int>(layer_sizes[layer]) - 1;
    return true;
}

bool HNSWNodeSimple::remove_neighbor_nolock(NodeId id, int layer)
{
    check_layer_bounds_or_throw(layer);
    const size_t begin = layer_offsets[layer];
    const size_t end = begin + layer_sizes[layer];
    auto it = std::find(neighbor_ids.begin() + begin, neighbor_ids.begin() + end, id);
    if (it == neighbor_ids.begin() + end) return false;
    neighbor_ids.erase(it);
    --layer_sizes[layer];
    for (size_t i = static_cast<size_t>(layer) + 1; i < layer_offsets.size(); ++i)
        --layer_offsets[i];
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
bool HNSWNodeSimple::add_neighbor(NodeId id, int layer, int *out_index)
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
bool HNSWNodeSimple::remove_neighbor(NodeId id, int layer, bool preserve_order)
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
bool HNSWNodeSimple::has_neighbor(NodeId id, int layer) const {
  check_layer_bounds_or_throw(layer);
  std::shared_lock lock(mtx);
  const size_t begin = layer_offsets[layer];
  const size_t end = begin + layer_sizes[layer];
  return std::find(neighbor_ids.begin() + begin, neighbor_ids.begin() + end, id) != neighbor_ids.begin() + end;
}

// Reserves capacity for the given layer.
//
// Args:
//   layer: Layer index to reserve.
//   capacity: Minimum capacity to reserve.
void HNSWNodeSimple::reserve_layer(int layer, size_t capacity) {
  check_layer_bounds_or_throw(layer);
  std::unique_lock lock(mtx);
  const size_t additional = capacity > layer_sizes[layer] ? capacity - layer_sizes[layer] : 0;
  neighbor_ids.reserve(neighbor_ids.size() + additional);
}

// Clears the given layer.
//
// Args:
//   layer: Layer index to clear.
void HNSWNodeSimple::clear_layer(int layer) {
  check_layer_bounds_or_throw(layer);
  std::unique_lock lock(mtx);
  const size_t begin = layer_offsets[layer];
  const size_t end = begin + layer_sizes[layer];
  neighbor_ids.erase(neighbor_ids.begin() + begin, neighbor_ids.begin() + end);
  const size_t removed = layer_sizes[layer];
  layer_sizes[layer] = 0;
  for (size_t i = static_cast<size_t>(layer) + 1; i < layer_offsets.size(); ++i)
      layer_offsets[i] -= removed;
}
}
