/**
 * @file hnsw.hpp
 * @brief Declaration of HNSW index representation
 * 
 * @details
 * This file defines the HNSWIndexSimple class, which represents a
 * HNSW (Hierarchical Navigable Small World) index for float vectors.
 * This header defines a minimal, in-memory HNSW (Hierarchical Navigable Small
 * World) index implementation used by the minivec project. The index stores
 * vectors in a VecStore and maintains a graph of HNSWNodeSimple nodes across
 * multiple levels.
 * The primary responsibilities of HNSWIndexSimple are:
 *    - Managing vector storage and node metadata (levels, neighbors).
 *    - Building the HNSW graph incrementally as new vectors are inserted.
 *    - Providing approximate nearest neighbor search via ef-search and 
 *      greedy search on different layers.
 *    - Part of the MiniVec project.
 */

#pragma once

#include "vecstore.hpp"
#include "hnsw_node.hpp"
#include "utils.hpp"

#include <vector>

// HNSWIndexSimple implements a basic HNSW index for float vectors.
//
// The index stores vectors in a VecStore and links them via an HNSW graph
// (HNSWNodeSimple). Each node can exist at multiple layers, forming a
// hierarchical small-world graph.
class HNSWIndexSimple {
private:
    // Maximum number of bi-directional connections (neighbors) per node
    // on each layer.
    int M;

    // Current maximum layer index in the graph (0-based).
    int max_level;

    // ID of the current entry point node in the top layer of the graph.
    int entry_point;

    // Dimensionality of all stored vectors.
    int dim;

    // Construction-time search breadth (efConstruction).
    int efConstruction;

    // Default search-time breadth (efSearch).
    int efSearch;

    // Underlying storage for all vectors, indexed by node ID.
    VecStore store;

    // Level generator used to assign a random level to each new node.
    HNSWLevelGenerator level_gen;

    // All HNSW nodes (graph vertices) stored in a contiguous vector.
    // The index into this vector is the node ID and corresponds to the
    // same index in the VecStore.
    std::vector<HNSWNodeSimple> nodes;

public:
    // Constructs an HNSWIndexSimple with the given dimensionality and parameters.
    //
    // Args:
    //   dim_: Dimensionality of all vectors to be stored in the index.
    //   M_: Maximum number of neighbors per node per layer (default 16).
    //   efConstruction_: Search breadth used during graph construction
    //       (default 200).
    //   efSearch_: Default search breadth used during queries (default 200).
    HNSWIndexSimple(int dim_, int M_ = 16, int efConstruction_ = 200, int efSearch_ = 200);

    // Adds a node with an externally provided level.
    //
    // Args:
    //   vec_vals: Pointer to a float array of length `dim` representing
    //       the vector to insert. The memory is copied into the VecStore.
    //   level: Level at which this node is created (0-based).
    //
    // Returns:
    //   Integer ID of the newly added node (index into `nodes` and `store`).
    int add_node(const float* vec_vals, int level);

    // Returns a const pointer to the stored vector for the given node ID.
    //
    // Args:
    //   id: Node ID whose vector is requested.
    //
    // Returns:
    //   Pointer to a float array of length `dim` representing the stored
    //   vector, or nullptr if `id` is invalid.
    const float* get_vector_ptr(int id) const;

    // Returns the level (maximum layer index) for a given node ID.
    //
    // Args:
    //   id: Node ID whose level is requested.
    //
    // Returns:
    //   The maximum layer index (0-based) on which this node exists.
    const int get_level(int id) const;

    // Returns the current entry point node ID used for top-level search.
    //
    // Returns:
    //   Node ID of the entry point, or -1 if the index is empty.
    const int get_entry_point() const;

    // Returns the current maximum layer index in the HNSW graph.
    //
    // Returns:
    //   Integer representing the highest layer index (0-based) that has
    //   at least one node.
    const int get_max_level() const;

    // Returns the total number of nodes currently stored in the index.
    //
    // Returns:
    //   Integer count of nodes.
    const int get_node_count() const;

    // Returns the dimensionality of all vectors in the index.
    //
    // Returns:
    //   Integer dimension of stored vectors.
    const int get_vector_dim() const;

    // Returns the maximum number of neighbors (M) per node per layer.
    //
    // Returns:
    //   Integer M value used by this index.
    const int get_M() const;

    // Returns the construction-time search breadth (efConstruction).
    //
    // Returns:
    //   Integer efConstruction value used for insertions.
    const int get_efConstruction() const;

    // Returns the default search-time breadth (efSearch).
    //
    // Returns:
    //   Integer efSearch value used for queries.
    const int get_efSearch() const;

    // Inserts a new vector into the index using an automatically generated level.
    //
    // Args:
    //   vec_vals: Pointer to a float array of length `dim` representing
    //       the vector to insert.
    //
    // Returns:
    //   Integer ID of the inserted node.
    int insert_vector(const float* vec_vals);

    // Prunes neighbors of a node at a given level to enforce the HNSW degree constraint.
    //
    // Args:
    //   id: Node ID whose neighbors are to be pruned.
    //   level: Layer index where pruning should be applied.
    void prune_neighbours(int id, int level);

    // Performs ef-search on a specific layer starting from an entry node.
    //
    // Args:
    //   query: Pointer to a float array of length `dim` representing the
    //       query vector.
    //   entry_id: Node ID from which to start the search on this layer.
    //   layer: Layer index (0-based) on which to perform the search.
    //   ef: Search breadth (maximum size of the candidate set).
    //
    // Returns:
    //   A priority queue (max-heap) of Candidate objects ordered by
    //   CandidateCompareInverse.
    std::priority_queue<Candidate, std::vector<Candidate>, CandidateCompareInverse>
    ef_search_layer(const float* query, int entry_id, int layer, int ef);

    // Performs greedy search on a specific layer starting from an entry node.
    //
    // Args:
    //   query: Pointer to a float array of length `dim` representing the
    //       query vector.
    //   entry_id: Node ID from which to start the search on this layer.
    //   layer: Layer index (0-based) on which to perform the search.
    //
    // Returns:
    //   Node ID of the closest node found on this layer using greedy descent.
    int greedy_search_layer(const float* query, int entry_id, int layer);

    // Performs a top-k approximate nearest neighbor search.
    //
    // Args:
    //   query: Pointer to a float array of length `dim` representing the
    //       query vector.
    //   ef: Search breadth used on the bottom layer.
    //   k: Number of nearest neighbors to return.
    //
    // Returns:
    //   A vector of Candidate objects representing the top-k approximate
    //   nearest neighbors, typically sorted by increasing distance.
    std::vector<Candidate> search_top_k(const float* query, int ef, int k);

    // Filters the candidate set to produce the final top-k results.
    //
    // Args:
    //   query: Pointer to a float array of length `dim` representing the
    //       query vector.
    //   candidates: Priority queue of Candidate objects, typically the
    //       result of ef_search_layer().
    //   k: Number of nearest neighbors to keep.
    //
    // Returns:
    //   A vector of Candidate objects representing the top-k results.
    std::vector<Candidate> filter_top_k(
        const float* query,
        std::priority_queue<Candidate, std::vector<Candidate>, CandidateCompareInverse>& candidates,
        int k);

    // Clears all data from the index while keeping configuration parameters.
    void clear();
};