// Implements the HNSWNodeSimple methods declared in hnsw_node.hpp.

#include "minivec/hnsw_node.hpp"

// Initializes a node with the given id, number of levels, and neighbor capacity.
//
// Args:
//   _id: Node identifier.
//   levels: Total number of levels for this node (must be >= 1).
//   M: Expected maximum number of neighbors per level.
HNSWNodeSimple::HNSWNodeSimple(int _id, int levels, int M)
    : id(_id), level(levels - 1), neighbors(levels) {
  for (auto& v : neighbors) v.reserve(M + 2);
}

// Returns the node id.
const int HNSWNodeSimple::getId() const {
  return id;
}

// Returns the neighbor list for a given layer.
//
// Args:
//   layer: Layer index in [0, level].
const std::vector<int>& HNSWNodeSimple::getNeighbors(int layer) const {
  return neighbors[layer];
}

// Returns the highest layer index for this node.
const int HNSWNodeSimple::getLevel() const {
  return level;
}

// Adds a neighbor at the given layer.
//
// Args:
//   id: Neighbor node id to add.
//   layer: Layer index where the neighbor is added.
//
// Returns:
//   Index at which the neighbor was inserted.
int HNSWNodeSimple::addNeighbor(int id, int layer) {
  neighbors[layer].push_back(id);
  return static_cast<int>(neighbors[layer].size()) - 1;
}

// Removes a neighbor from the given layer, if present.
//
// Args:
//   id: Neighbor node id to remove.
//   layer: Layer index to remove from.
//
// Returns:
//   1 if the neighbor was removed, 0 if it was not found.
int HNSWNodeSimple::removeNeighbor(int id, int layer) {
  auto it = std::find(neighbors[layer].begin(), neighbors[layer].end(), id);
  if (it != neighbors[layer].end()) {
    neighbors[layer].erase(it);
    return 1;
  }
  return 0;
}