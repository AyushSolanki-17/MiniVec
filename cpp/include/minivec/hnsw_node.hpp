/**
 * @file hnsw_node.hpp
 * @brief Declaration of HNSW node representation
 *
 * @details
 * This file defines the HNSWNodeSimple class, which represents a node in a
 * HNSW (Hierarchical Navigable Small World) graph.
 * Part of the MiniVec project.
 */
#pragma once

#include <vector>

// Simple HNSW node representation used by the MiniVec HNSW engine.
//
// Each node has an integer id, a highest level index, and per-level neighbor
// lists. This type is intentionally minimal and relies only on standard
// containers.
struct HNSWNodeSimple {
   private:
    // Unique identifier of the node within the index.
    int id;

    // Highest level (0-based) on which this node exists.
    // A value of 0 means the node exists only on the base layer.
    int level;

    // Adjacency lists for each level; neighbors[l] holds neighbor IDs at level l.
    std::vector<std::vector<int>> neighbors;

   public:
    // Constructs an HNSW node with a given id, number of levels, and capacity.
    //
    // Levels are in the range [0, levels - 1]. For each level, an adjacency
    // list is created and may reserve up to M neighbors.
    //
    // Args:
    //   _id: Integer identifier of this node (default -1 for placeholder).
    //   levels: Number of levels for this node (must be >= 1).
    //   M: Expected maximum number of neighbors per level (for reserve).
    HNSWNodeSimple(int _id = -1, int levels = 1, int M = 16);

    // Returns the unique identifier of this node.
    //
    // Returns:
    //   Integer ID associated with this node.
    const int getId() const;

    // Returns the neighbor list for a specific layer.
    //
    // Caller must ensure that layer is in range [0, getLevel()].
    //
    // Args:
    //   layer: Zero-based index of the level.
    //
    // Returns:
    //   Const reference to the vector of neighbor node IDs at the given layer.
    const std::vector<int>& getNeighbors(int layer) const;

    // Returns the highest level index of this node.
    //
    // Returns:
    //   Highest (0-based) level on which this node exists.
    const int getLevel() const;

    // Adds a neighbor to the specified layer.
    //
    // This does not enforce uniqueness or degree constraints.
    //
    // Args:
    //   id: Identifier of the neighbor to add.
    //   layer: Zero-based level index where the neighbor is added.
    //
    // Returns:
    //   Implementation-defined status code.
    int addNeighbor(int id, int layer);

    // Removes a neighbor from the specified layer, if present.
    //
    // Args:
    //   id: Identifier of the neighbor to remove.
    //   layer: Zero-based level index from which the neighbor is removed.
    //
    // Returns:
    //   Implementation-defined status code.
    int removeNeighbor(int id, int layer);
};