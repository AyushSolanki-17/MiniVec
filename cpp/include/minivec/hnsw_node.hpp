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
#include <shared_mutex>
#include <optional>
#include <limits>

namespace minivec
{
    // Simple HNSW node representation used by the MiniVec HNSW engine.
    //
    // Each node has an integer id, a highest layer index, and per-layer neighbor
    // lists. This type is intentionally minimal and relies only on standard
    // containers.
    struct HNSWNodeSimple
    {
    private:
        // Unique identifier of the node within the index.
        int id;

        // Highest layer (0-based) on which this node exists.
        // A value of 0 means the node exists only on the base layer.
        int layer;

        // Adjacency lists for each layer; neighbors[l] holds neighbor IDs at layer l.
        std::vector<std::vector<int>> neighbors;

        // Mutex for thread-safe access to the node.
        mutable std::shared_mutex mtx;

        void check_layer_bounds_or_throw(int layer) const;

    public:
        // Constructs an HNSW node with a given id, number of layers, and capacity.
        //
        // layers are in the range [0, layers - 1]. For each layer, an adjacency
        // list is created and may reserve up to M neighbors.
        //
        // Args:
        //   _id: Integer identifier of this node (default -1 for placeholder).
        //   layers: Number of layers for this node (must be >= 1).
        //   M: Expected maximum number of neighbors per layer (for reserve).
        HNSWNodeSimple(int _id = -1, int layers = 1, int M = 16);

        // Disable copy to avoid accidental expensive shallow copies.
        HNSWNodeSimple(const HNSWNodeSimple &) = delete;
        HNSWNodeSimple &operator=(const HNSWNodeSimple &) = delete;
        HNSWNodeSimple(HNSWNodeSimple &&) noexcept = delete;
        HNSWNodeSimple &operator=(HNSWNodeSimple &&) noexcept = delete;

        // Returns the unique identifier of this node.
        //
        // Returns:
        //   Integer ID associated with this node.
        const int get_id() const;

        // Returns the neighbor list for a specific layer.
        //
        // Caller must ensure that layer is in range [0, getlayer()].
        //
        // Args:
        //   layer: Zero-based index of the layer.
        //
        // Returns:
        //   Const reference to the vector of neighbor node IDs at the given layer.
        const std::vector<int> get_neighbors(int layer) const;

        // Returns the highest layer index of this node.
        //
        // Returns:
        //   Highest (0-based) layer on which this node exists.
        int get_layer() const;

        // Adds a neighbor to the specified layer.
        //
        // This does not enforce uniqueness or degree constraints.
        //
        // Args:
        //   id: Identifier of the neighbor to add.
        //   layer: Zero-based layer index where the neighbor is added.
        //   out_index: If not null, will be set to the index at which the neighbor
        //     was added.
        //
        // Returns:
        //   Implementation-defined status code.
        bool add_neighbor(int id, int layer, int *out_index = nullptr);

        // Removes a neighbor from the specified layer, if present.
        //
        // Args:
        //   id: Identifier of the neighbor to remove.
        //   layer: Zero-based layer index from which the neighbor is removed.
        //   preserve_order: If true, the neighbor will be swapped with the last
        //     neighbor and then removed.
        //
        // Returns:
        //   Implementation-defined status code.
        bool remove_neighbor(int id, int layer, bool preserve_order = false);

        // Returns whether given neighbor exists in the layer.
        //
        // Args:
        //   id: Identifier of the neighbor.
        //   layer: Zero-based layer index of the neighbor.
        //
        // Returns:
        //   True if the neighbor exists, false otherwise.
        bool has_neighbor(int id, int layer) const;

        // Reserves capacity for the given layer.
        //
        // Args:
        //   layer: Zero-based layer index.
        //   capacity: Desired capacity of the layer.
        //
        void reserve_layer(int layer, size_t capacity);

        // Clears the neighbor list for the given layer.
        //
        // Args:
        //   layer: Zero-based layer index.
        void clear_layer(int layer);

        // Expose node mutex so callers can lock two nodes without race / deadlock.
        // Return by reference to avoid copying mutex object.
        std::shared_mutex &getMutex() const { return mtx; }

        // internal: caller must hold exclusive lock on mtx
        bool add_neighbor_nolock(int id, int layer, int *out_index = nullptr);
        bool remove_neighbor_nolock(int id, int layer);
    };
} // namespace minivec