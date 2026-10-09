/**
 * @file hnsw.cpp
 * @brief Implementation of HNSW index
 * @details
 * This file implements the HNSWIndexSimple class, which represents a
 * HNSW (Hierarchical Navigable Small World) index for float vectors.
 * This implementation provides methods for inserting vectors, searching for
 * nearest neighbors, and managing the HNSW graph structure.
 */
#include "minivec/hnsw.hpp"
#include "minivec/dist.hpp"

#include <algorithm>
#include <cmath>
#include <mutex>
#include <sstream>
#include <shared_mutex>
#include <iostream>
#include <string>
#include <limits>
#include <stdexcept>
#include <utility>

namespace minivec
{
    namespace
    {
        void validate_vector(const float *values, int dimension, const char *context)
        {
            if (!values)
                throw std::invalid_argument(std::string(context) + ": vector pointer must not be null");
            for (int i = 0; i < dimension; ++i)
            {
                if (!std::isfinite(values[i]))
                    throw std::invalid_argument(std::string(context) + ": vectors must contain only finite values");
            }
        }
    }

    // Constructs a simple HNSW index.
    //
    // Args:
    //   dim_: Dimensionality of stored vectors.
    //   M_: Upper-layer neighbor limit (layer 0 permits up to 2M).
    //   efConstruction_: Size of dynamic candidate list during construction.
    //   efSearch_: Default ef parameter for search.
    HNSWIndexSimple::HNSWIndexSimple(int dim_, int M_, int efConstruction_, int efSearch_,
                                     bool deterministic_levelgen, uint32_t levelgen_seed,
                                     const std::string &distance_func_name_,
                                     const std::string &final_distance_func_name_)
        : M(M_),
          max_layer(-1),
          entry_point(-1),
          dim(dim_),
          efConstruction(efConstruction_),
          efSearch(efSearch_),
          store(dim_),
          layer_gen(deterministic_levelgen ? minivec::HNSWLevelGenerator::from_M(M_, deterministic_levelgen, levelgen_seed) : minivec::HNSWLevelGenerator::from_M(M_)),
          distance_func(get_distance_metric(distance_func_name_)),
          final_distance_func(get_distance_metric(final_distance_func_name_))
    {
        if (dim <= 0)
            throw std::invalid_argument("HNSWIndexSimple: dimension must be positive");
        if (M > std::numeric_limits<int>::max() / 2)
            throw std::invalid_argument("HNSWIndexSimple: M is too large");
        if (efConstruction <= 0)
            throw std::invalid_argument("HNSWIndexSimple: efConstruction must be positive");
        if (efSearch <= 0)
            throw std::invalid_argument("HNSWIndexSimple: efSearch must be positive");
    }

    // Adds a new node with the given vector and layer.
    //
    // The node is appended to the internal store and node list. The entry point
    // and max_layer are updated if this node becomes the new top-layer entry.
    //
    // Args:
    //   vec_vals: Pointer to the vector data of size dim.
    //   layer: layer assigned to this node (0-based external layer).
    //
    // Returns:
    //   The id of the newly added node.
    NodeId HNSWIndexSimple::add_node(const float *vec_vals, int layer)
    {
        validate_vector(vec_vals, dim, "add_node");
        std::lock_guard<std::mutex> mutation_lock(mutation_mtx);
        std::unique_lock<std::shared_mutex> idx_lock(index_mtx);
        const NodeId id = add_node_unlocked(vec_vals, layer);
        if (layer > max_layer)
        {
            max_layer = layer;
            entry_point = id;
        }
        return id;
    }

    NodeId HNSWIndexSimple::add_node_unlocked(const float *vec_vals, int layer)
    {
        if (layer < 0 || layer > layer_gen.max_level())
            throw std::invalid_argument("add_node: layer is outside the supported range");
        if (nodes.size() >= static_cast<size_t>(std::numeric_limits<NodeId>::max()))
            throw std::length_error("HNSWIndexSimple: node ID capacity exhausted");
        auto node = std::make_unique<HNSWNodeSimple>(static_cast<NodeId>(nodes.size()), layer + 1, M);
        NodeId id = store.add(vec_vals);

        // Check for valid id.
        if (id != static_cast<NodeId>(nodes.size()))
        {
            std::ostringstream oss;
            oss << "add_node: store.add() returned id " << id
                << " but expected " << nodes.size() << ". "
                << "Please ensure store.add() uses the same id space as nodes.";
            store.pop_back();
            throw std::runtime_error(oss.str());
        }
        try
        {
            nodes.emplace_back(std::move(node));
        }
        catch (...)
        {
            store.pop_back();
            throw;
        }
        if (entry_point == -1)
        {
            entry_point = id;
            max_layer = layer;
        }
        return id;
    }

    // Returns a pointer to the stored vector for the given node id.
    const float *HNSWIndexSimple::get_vector_ptr(NodeId id) const
    {
        std::shared_lock<std::shared_mutex> idx_lock(index_mtx);
        return get_vector_ptr_unlocked(id);
    }

    const float *HNSWIndexSimple::get_vector_ptr_unlocked(NodeId id) const
    {
        throw_if_invalid_node_id(nodes, id, "get_vector_ptr");
        const float *p = store.ptr(id);
        if (!p)
        {
            std::ostringstream oss;
            oss << "get_vector_ptr: store.ptr(" << id << ") returned nullptr";
            throw std::runtime_error(oss.str());
        }
        return p;
    }

    // Returns the top layer of the node with the given id.
    int HNSWIndexSimple::get_layer(NodeId id) const
    {
        std::shared_lock<std::shared_mutex> idx_lock(index_mtx);
        throw_if_invalid_node_id(nodes, id, "get_layer");
        return nodes[id]->get_layer();
    }

    // Returns the current entry point id for the index.
    NodeId HNSWIndexSimple::get_entry_point() const
    {
        std::shared_lock<std::shared_mutex> idx_lock(index_mtx);
        return entry_point;
    }

    // Returns the current maximum layer in the index.
    int HNSWIndexSimple::get_max_layer() const
    {
        std::shared_lock<std::shared_mutex> idx_lock(index_mtx);
        return max_layer;
    }

    // Returns the number of nodes stored in the index.
    NodeId HNSWIndexSimple::get_node_count() const
    {
        std::shared_lock<std::shared_mutex> idx_lock(index_mtx);
        return nodes.size();
    }

    // Returns the dimensionality of vectors in this index.
    int HNSWIndexSimple::get_vector_dim() const
    {
        return dim;
    }

    // Returns the maximum number of neighbors per layer.
    int HNSWIndexSimple::get_M() const
    {
        return M;
    }

    // Returns the ef parameter used during index construction.
    int HNSWIndexSimple::get_efConstruction() const
    {
        return efConstruction;
    }

    // Returns the default ef parameter used during search.
    int HNSWIndexSimple::get_efSearch() const
    {
        return efSearch.load(std::memory_order_relaxed);
    }

    void HNSWIndexSimple::set_efSearch(int efSearch_)
    {
        if (efSearch_ <= 0)
            throw std::invalid_argument("HNSWIndexSimple: efSearch must be positive");
        efSearch.store(efSearch_, std::memory_order_relaxed);
    }

    // Prunes neighbors of a node at a given layer using the HNSW diversity rule.
    //
    // Keeps at most M (or 2M on layer 0) neighbors that are close and diverse.
    //
    // Args:
    //   id: Node id whose neighbors are pruned.
    //   layer: Layer index where pruning is applied.
    void HNSWIndexSimple::prune_neighbours(NodeId id, int layer)
    {
        std::lock_guard<std::mutex> mutation_lock(mutation_mtx);
        std::shared_lock<std::shared_mutex> idx_lock(index_mtx);
        prune_neighbours_unlocked(id, layer);
    }

    void HNSWIndexSimple::prune_neighbours_unlocked(NodeId id, int layer)
    {
        // Validate id
        throw_if_invalid_node_id(nodes, id, "prune_neighbours");
        std::vector<NodeId> nbrs = nodes[static_cast<size_t>(id)]->get_neighbors(layer);
        const int max_neighbors = layer == 0 ? 2 * M : M;
        if (nbrs.size() <= static_cast<size_t>(max_neighbors))
            return; // Nothing to prune.

        // Build candidate list with distances to the node.
        std::vector<Candidate> candidates;
        candidates.reserve(nbrs.size());
        const float *id_ptr = get_vector_ptr_unlocked(id);
        for (NodeId n : nbrs)
        {
            float d = compute_distance(distance_func, id_ptr, get_vector_ptr_unlocked(n), dim);
            candidates.push_back({n, d});
        }

        // Sort by increasing distance to the node.
        std::sort(candidates.begin(), candidates.end(), candidate_distance_less);

        int pool = std::min<int>((int)candidates.size(), std::max<int>(M * 2, (int)M + 8));

        // Diversity-based selection.
        std::vector<NodeId> selected;
        selected.reserve(max_neighbors);

        // Cache pointers for selected neighbors
        std::vector<const float *> selected_ptrs;
        selected_ptrs.reserve(max_neighbors);

        for (int i = 0; i < pool && (int)selected.size() < max_neighbors; ++i)
        {
            const Candidate &c = candidates[i];
            const float *c_ptr = get_vector_ptr_unlocked(c.id);

            bool good = true;

            for (size_t j = 0; j < selected_ptrs.size(); ++j)
            {
                float ds = compute_distance(distance_func, c_ptr, selected_ptrs[j], dim);

                // HNSW diversity rule
                if (ds < c.distance)
                {
                    good = false;
                    break;
                }
            }

            if (good)
            {
                selected.push_back(c.id);
                selected_ptrs.push_back(c_ptr);
            }
        }

        // Fallback: if diversity too strict, fill remaining slots with nearest unused candidates.
        if ((int)selected.size() < max_neighbors)
        {
            for (int i = 0; i < pool && (int)selected.size() < max_neighbors; ++i)
            {
                NodeId cand_id = candidates[i].id;
                // add if not already selected
                if (std::find(selected.begin(), selected.end(), cand_id) == selected.end())
                    selected.push_back(cand_id);
            }
        }

        NodeId n_nodes = static_cast<NodeId>(nodes.size());

        // Sort selected for efficient lookup.
        std::sort(selected.begin(), selected.end());
        for (NodeId old : nbrs)
        {
            // Skip if already selected.
            if (std::binary_search(selected.begin(), selected.end(), old))
                continue;

            if (old < 0 || old >= n_nodes)
                continue;
            if (old == id)
                continue;

            // Pruning only changes this node's adjacency list. Removing the
            // reverse edge can isolate the other node and disconnect the graph.
            std::unique_lock lock(nodes[id]->getMutex());
            nodes[id]->remove_neighbor_nolock(old, layer);
        }
    }

    // Inserts a new vector into the index.
    //
    // Generates a random layer for the new node, navigates from the entry point
    // down to that layer using greedy search, then connects the node to neighbors
    // on each relevant layer using efConstruction.
    //
    // Args:
    //   vec_vals: Pointer to the vector data of size dim.
    //
    // Returns:
    //   The id of the inserted node.
    NodeId HNSWIndexSimple::insert_vector(const float *vec_vals)
    {
        validate_vector(vec_vals, dim, "insert_vector");
        std::lock_guard<std::mutex> mutation_lock(mutation_mtx);
        int new_layer = layer_gen.getRandomLayer();

        NodeId id;
        // Now size >= 2; start search from old_entry (not self).
        NodeId current;
        int local_max_layer;
        {
            std::unique_lock<std::shared_mutex> idx_lock(index_mtx);
            id = add_node_unlocked(vec_vals, new_layer);
            if (nodes.size() == 1)
            {
                entry_point = id;
                max_layer = new_layer;
                return id;
            }
            current = entry_point;
            local_max_layer = max_layer;
        }

        // Validate old entry.
        if (current < 0 || static_cast<std::uint64_t>(current) >= nodes.size())
        {
            std::ostringstream oss;
            oss << "insert_vector: invalid old_entry " << current;
            throw std::runtime_error(oss.str());
        }

        // Greedy descent on upper layers.
        for (int layer = local_max_layer; layer >= new_layer + 1; layer--)
        {
            current = greedy_search_layer_unlocked(vec_vals, current, layer, nullptr);
        }

        // Connect on layers from new_layer down to 0.
        int connect_start_layer = std::min(new_layer, local_max_layer);
        for (int layer = connect_start_layer; layer >= 0; layer--)
        {
            std::priority_queue<Candidate, std::vector<Candidate>, MaxHeapCompare> neighbors_pq =
                ef_search_layer_unlocked(vec_vals, current, layer, efConstruction, nullptr);
            const int max_neighbors = layer == 0 ? 2 * M : M;
            std::vector<Candidate> neighbors = filter_top_k_unlocked(vec_vals, neighbors_pq, max_neighbors, true);

            for (const Candidate &neighbor : neighbors)
            {
                // Skip self
                if (neighbor.id == id)
                    continue;
                // Validate neighbor id
                throw_if_invalid_node_id(nodes, neighbor.id, "insert_vector: neighbor id");
                NodeId a = id;
                NodeId b = neighbor.id;
                if (a == b)
                    continue;

                // Lock node mutexes in consistent order (lower id first) to avoid deadlocks.
                if (a < b)
                {
                    std::scoped_lock lock(nodes[a]->getMutex(), nodes[b]->getMutex());
                    nodes[a]->add_neighbor_nolock(b, layer);
                    nodes[b]->add_neighbor_nolock(a, layer);
                }
                else
                {
                    std::scoped_lock lock(nodes[b]->getMutex(), nodes[a]->getMutex());
                    nodes[a]->add_neighbor_nolock(b, layer);
                    nodes[b]->add_neighbor_nolock(a, layer);
                }

                // keep at most M neighbors for the neighbor node.
                // Local capacity control: prune under lock to keep it consistent.
                // Acquire single-node locks for pruning (prune_neighbours will call node-level methods that acquire their own locks).
                std::vector<NodeId> nbrs = nodes[static_cast<size_t>(neighbor.id)]->get_neighbors(layer);
                const int neighbor_limit = layer == 0 ? 2 * M : M;
                if ((int)nbrs.size() > neighbor_limit)
                {
                    prune_neighbours_unlocked(neighbor.id, layer);
                }
            }
            // Update starting point for next (lower) layer to closest found here.
            if (layer > 0 && !neighbors.empty())
            {
                current = neighbors[0].id; // filter_top_k sorts by increasing dist, so [0] is closest.
            }
        }
        // Update entry point and max_layer if needed.
        {
            std::unique_lock<std::shared_mutex> idx_lock(index_mtx);
            if (new_layer > max_layer)
            {
                max_layer = new_layer;
                entry_point = id;
            }
        }
        return id;
    }

    // Performs greedy search on a single layer starting from entry_id.
    //
    // Iteratively moves to any neighbor that is closer to the query until no
    // improvement is possible.
    //
    // Args:
    //   query: Pointer to query vector of size dim.
    //   entry_id: Starting node id for this layer.
    //   layer: Layer index where the search is performed.
    //
    // Returns:
    //   Id of the closest node found on this layer.
    NodeId HNSWIndexSimple::greedy_search_layer(const float *query, NodeId entry_id, int layer, SearchStats *stats)
    {
        validate_vector(query, dim, "greedy_search_layer");
        std::shared_lock<std::shared_mutex> idx_lock(index_mtx);
        return greedy_search_layer_unlocked(query, entry_id, layer, stats);
    }

    NodeId HNSWIndexSimple::greedy_search_layer_unlocked(const float *query, NodeId entry_id, int layer, SearchStats *stats)
    {
        throw_if_invalid_node_id(nodes, entry_id, "greedy_search_layer: entry_id");
        NodeId n_nodes = static_cast<NodeId>(nodes.size());
        NodeId current = entry_id;
        bool improved = true;
        while (improved)
        {
            improved = false;
            // Defensive validation: ensure current refers to a valid node and pointer is not null
            if (current < 0 || current >= n_nodes)
            {
                std::ostringstream oss;
                oss << "greedy_search_layer: invalid current id " << current;
                throw std::out_of_range(oss.str());
            }
            if (!nodes[static_cast<size_t>(current)])
            {
                std::ostringstream oss;
                oss << "greedy_search_layer: nodes[" << current << "] is nullptr";
                throw std::runtime_error(oss.str());
            }
            // cache current pointer once
            const float *pv_curr = store.ptr(current);
            float best_dist = compute_distance(distance_func, query, pv_curr, dim);
            // Explore neighbors
            nodes[current]->for_each_neighbor(layer, [&](NodeId neighbor) {
                if (neighbor < 0 || neighbor >= n_nodes)
                    return;
                const float *pv_nei = store.ptr(neighbor);
                if (!pv_curr || !pv_nei)
                    return;

                float c_dist = compute_distance(distance_func, query, pv_nei, dim);
                // Check for improvement
                if (c_dist < best_dist)
                {
                    best_dist = c_dist;
                    current = neighbor;
                    pv_curr = pv_nei;
                    improved = true;
                    if (stats)
                        ++stats->greedy_hops;
                }
            });
        }
        return current;
    }

    // Performs ef-search on a single layer.
    //
    // Maintains a candidate list and a bounded set of best nodes, exploring
    // neighbors until no better candidates remain.
    //
    // Args:
    //   query: Pointer to query vector of size dim.
    //   entry_id: Starting node id for this layer.
    //   layer: Layer index where the search is performed.
    //   ef: Maximum size of the best_nodes set.
    //
    // Returns:
    //   A max-heap (by distance) of Candidate objects representing the best nodes.
    std::priority_queue<Candidate, std::vector<Candidate>, MaxHeapCompare>
    HNSWIndexSimple::ef_search_layer(const float *query, NodeId entry_id, int layer, int ef, SearchStats *stats)
    {
        validate_vector(query, dim, "ef_search_layer");
        if (ef <= 0)
            throw std::invalid_argument("ef_search_layer: ef must be positive");
        std::shared_lock<std::shared_mutex> idx_lock(index_mtx);
        return ef_search_layer_unlocked(query, entry_id, layer, ef, stats);
    }

    std::priority_queue<Candidate, std::vector<Candidate>, MaxHeapCompare>
    HNSWIndexSimple::ef_search_layer_unlocked(const float *query, NodeId entry_id, int layer, int ef, SearchStats *stats)
    {
        std::priority_queue<Candidate, std::vector<Candidate>, MaxHeapCompare> best_nodes;
        std::priority_queue<Candidate, std::vector<Candidate>, MinHeapCompare> candidates;
        NodeId n_nodes = static_cast<NodeId>(nodes.size());
        if (ef <= 0)
            throw std::invalid_argument("ef_search_layer: ef must be positive");
        if (n_nodes == 0)
            return best_nodes;
        if (entry_id < 0 || entry_id >= n_nodes)
        {
            std::ostringstream oss;
            oss << "ef_search_layer: invalid entry_id " << entry_id;
            throw std::out_of_range(oss.str());
        }
        // Track visited nodes to avoid re-processing.
        // Reuse per-thread visitation storage. Incrementing the epoch avoids
        // clearing an O(N) bitmap on every layer search while keeping concurrent
        // queries independent.
        struct VisitWorkspace
        {
            std::vector<uint32_t> epochs;
            uint32_t current_epoch = 0;
        };
        // One scratch workspace per thread is enough: each layer traversal
        // advances the epoch before reading marks. Keying these by index would
        // retain one O(N) bitmap for every index ever searched on this thread.
        static thread_local VisitWorkspace workspace;
        if (workspace.epochs.size() < static_cast<size_t>(n_nodes))
            workspace.epochs.resize(n_nodes, 0);
        if (++workspace.current_epoch == 0)
        {
            std::fill(workspace.epochs.begin(), workspace.epochs.end(), 0);
            workspace.current_epoch = 1;
        }
        const uint32_t visit_epoch = workspace.current_epoch;
        // Initialize with entry point.
        NodeId current = entry_id;
        float curr_dist = compute_distance(distance_func, query, store.ptr(current), dim);
        candidates.emplace(current, curr_dist);
        best_nodes.emplace(current, curr_dist);
        workspace.epochs[current] = visit_epoch;
        if (stats)
        {
            stats->visited_nodes++;
            stats->layer_visits[layer]++;
            stats->distance_calls++;
        }

        while (!candidates.empty())
        {

            curr_dist = candidates.top().distance;
            current = candidates.top().id;
            candidates.pop();
            // Termination condition: current distance worse than worst in best_nodes.
            float worst_best_distance = best_nodes.top().distance;

            if (curr_dist > worst_best_distance)
            {
                break;
            }
            // Defensive validation: ensure current refers to a valid node and pointer
            if (current < 0 || current >= n_nodes)
            {
                std::cerr << "[ERR] ef_search_layer: skipping invalid current id=" << current << "\n";
                continue;
            }
            if (!nodes[static_cast<size_t>(current)])
            {
                std::cerr << "[ERR] ef_search_layer: nodes[" << current << "] is nullptr; skipping\n";
                continue;
            }
            nodes[current]->for_each_neighbor(layer, [&](NodeId neighbor) {
                if (neighbor < 0 || neighbor >= n_nodes)
                    return;
                if (workspace.epochs[neighbor] != visit_epoch)
                {
                    workspace.epochs[neighbor] = visit_epoch;
                    if (stats)
                    {
                        stats->visited_nodes++;
                        stats->layer_visits[layer]++;
                    }

                    // Compute distance to neighbor
                    const float *pv_nei = store.ptr(neighbor);
                    if (!pv_nei)
                        return;
                    float dist = compute_distance(distance_func, query, pv_nei, dim);
                    if (stats)
                    {
                        stats->distance_calls++;
                    }

                    const int search_width = std::max(1, ef);
                    if (static_cast<int>(best_nodes.size()) < search_width)
                    {
                        candidates.emplace(neighbor, dist);
                        best_nodes.emplace(neighbor, dist);
                        worst_best_distance = best_nodes.top().distance;
                    }
                    else if (dist < worst_best_distance)
                    {
                        candidates.emplace(neighbor, dist);
                        best_nodes.pop();
                        best_nodes.emplace(neighbor, dist);
                        // Keep the bound current for the next neighbor in this
                        // adjacency list.
                        worst_best_distance = best_nodes.top().distance;
                    }
                }
            });
        }
        return best_nodes;
    }

    // Searches for the top-k nearest neighbors of a query vector.
    //
    // First performs greedy search on upper layers, then ef-search on layer 0,
    // and finally applies diversity-based filtering.
    //
    // Args:
    //   query: Pointer to query vector of size dim.
    //   ef: ef parameter for this search (size of candidate set).
    //   k: Number of neighbors to return.
    //
    // Returns:
    //   A vector of the top-k Candidate results sorted by distance.
    std::vector<Candidate> HNSWIndexSimple::search_top_k(
        const float *query, int ef, int k, SearchStats *stats)
    {
        if (k < 0)
            throw std::invalid_argument("search_top_k: k must not be negative");
        validate_vector(query, dim, "search_top_k");
        if (k == 0)
            return {};
        std::shared_lock<std::shared_mutex> idx_shared_lock(index_mtx);
        if (nodes.empty() || entry_point < 0)
            return {};
        NodeId current = entry_point;
        int lower_max_layer = max_layer;

        // Greedy search on upper layers.
        for (int layer = lower_max_layer; layer > 0; layer--)
        {
            current = greedy_search_layer_unlocked(query, current, layer, stats);
        }

        // EF search on layer 0.
        const int configured_ef = ef > 0 ? ef : efSearch.load(std::memory_order_relaxed);
        int effective_ef = std::max(configured_ef, k);
        std::priority_queue<Candidate, std::vector<Candidate>, MaxHeapCompare> candidates = ef_search_layer_unlocked(query, current, 0, effective_ef, stats);

        return filter_top_k_unlocked(query, candidates, k, false);
    }

    // Filters candidates to produce a top-k result set, optionally applying
    // the HNSW diversity heuristic before filling any remaining slots.
    //
    // Recomputes distances to the query before final sorting.
    //
    // Args:
    //   query: Pointer to query vector of size dim.
    //   candidates_pq: Max-heap of candidates from ef_search_layer.
    //   k: Number of neighbors to keep.
    //
    // Returns:
    //   A vector of up to k candidates sorted by distance to the query.
    std::vector<Candidate> HNSWIndexSimple::filter_top_k(
        const float *query,
        std::priority_queue<Candidate, std::vector<Candidate>, MaxHeapCompare> &candidates_pq,
        int k,
        bool diversity)
    {
        if (k < 0)
            throw std::invalid_argument("filter_top_k: k must not be negative");
        validate_vector(query, dim, "filter_top_k");
        if (k == 0)
            return {};
        std::shared_lock<std::shared_mutex> idx_lock(index_mtx);
        return filter_top_k_unlocked(query, candidates_pq, k, diversity);
    }

    std::vector<Candidate> HNSWIndexSimple::filter_top_k_unlocked(
        const float *query,
        std::priority_queue<Candidate, std::vector<Candidate>, MaxHeapCompare> &candidates_pq,
        int k,
        bool diversity)
    {
        if (k < 0)
            throw std::invalid_argument("filter_top_k: k must not be negative");
        if (k == 0)
            return {};
        std::vector<Candidate> top_k;
        const size_t result_capacity = std::min(static_cast<size_t>(k), candidates_pq.size());
        top_k.reserve(result_capacity);

        std::vector<Candidate> candidates;
        candidates.reserve(candidates_pq.size());

        while (!candidates_pq.empty())
        {
            candidates.push_back(candidates_pq.top());
            candidates_pq.pop();
        }

        std::sort(candidates.begin(), candidates.end(), candidate_distance_less);

        if (diversity)
        {
            std::vector<const float *> selected_ptrs;
            selected_ptrs.reserve(result_capacity);
            for (const Candidate &candidate : candidates)
            {
                const float *candidate_ptr = get_vector_ptr_unlocked(candidate.id);
                bool diverse_enough = true;
                for (const float *selected_ptr : selected_ptrs)
                {
                    if (compute_distance(distance_func, candidate_ptr, selected_ptr, dim) < candidate.distance)
                    {
                        diverse_enough = false;
                        break;
                    }
                }
                if (diverse_enough)
                {
                    top_k.push_back(candidate);
                    selected_ptrs.push_back(candidate_ptr);
                    if (static_cast<int>(top_k.size()) == k)
                        break;
                }
            }
            // A strict diversity test can reject every remaining candidate;
            // fill any open slots with the nearest unselected candidates.
            for (const Candidate &candidate : candidates)
            {
                if (static_cast<int>(top_k.size()) == k)
                    break;
                if (std::none_of(top_k.begin(), top_k.end(), [&](const Candidate &chosen) { return chosen.id == candidate.id; }))
                    top_k.push_back(candidate);
            }
        }
        else
        {
            for (int i = 0; i < k && i < static_cast<int>(candidates.size()); ++i)
                top_k.push_back(candidates[i]);
        }

        for (Candidate &c : top_k)
        {
            c.distance = compute_distance(final_distance_func, query, get_vector_ptr_unlocked(c.id), dim);
        }

        std::sort(top_k.begin(), top_k.end(), candidate_distance_less);

        return top_k;
    }

    // Clears all data from the index and resets state.
    void HNSWIndexSimple::clear()
    {
        std::lock_guard<std::mutex> mutation_lock(mutation_mtx);
        std::unique_lock<std::shared_mutex> idx_lock(index_mtx);
        store.clear();
        nodes.clear();
        entry_point = -1;
        max_layer = -1;
        layer_gen.reset();
    }

    // Thread-safe helper: copy neighbors for node node_id at given layer.
    std::vector<NodeId> HNSWIndexSimple::get_neighbors_copy(NodeId node_id, int layer) const
    {
        std::shared_lock<std::shared_mutex> idx_lock(index_mtx);
        // Validate node id first
        throw_if_invalid_node_id(nodes, node_id, "get_neighbors_copy");
        return nodes[node_id]->get_neighbors(layer);
    }

    // Symmetric link: add edge a->b and b->a safely without deadlock.
    void HNSWIndexSimple::link_nodes_symmetrically(NodeId a, NodeId b, int layer)
    {
        std::lock_guard<std::mutex> mutation_lock(mutation_mtx);
        std::shared_lock<std::shared_mutex> idx_lock(index_mtx);
        // Validate ids
        throw_if_invalid_node_id(nodes, a, "link_nodes_symmetrically: a");
        throw_if_invalid_node_id(nodes, b, "link_nodes_symmetrically: b");
        // ignore self-links
        if (a == b)
            return;
        if (layer < 0 || layer > nodes[a]->get_layer() || layer > nodes[b]->get_layer())
            throw std::out_of_range("link_nodes_symmetrically: layer is unavailable on both nodes");

        // Lock the two node mutexes in id order to avoid deadlocks.
        if (a < b)
        {
            std::scoped_lock lock(nodes[a]->getMutex(), nodes[b]->getMutex());
            nodes[a]->add_neighbor_nolock(b, layer);
            nodes[b]->add_neighbor_nolock(a, layer);
        }
        else
        {
            std::scoped_lock lock(nodes[b]->getMutex(), nodes[a]->getMutex());
            nodes[a]->add_neighbor_nolock(b, layer);
            nodes[b]->add_neighbor_nolock(a, layer);
        }
    }

    // Symmetric unlink: remove edge a->b and b->a safely without deadlock.
    void HNSWIndexSimple::remove_link_symmetrically(NodeId a, NodeId b, int layer)
    {
        std::lock_guard<std::mutex> mutation_lock(mutation_mtx);
        std::shared_lock<std::shared_mutex> idx_lock(index_mtx);
        // Validate ids
        throw_if_invalid_node_id(nodes, a, "remove_link_symmetrically: a");
        throw_if_invalid_node_id(nodes, b, "remove_link_symmetrically: b");
        if (a == b)
            return;
        if (layer < 0 || layer > nodes[a]->get_layer() || layer > nodes[b]->get_layer())
            throw std::out_of_range("remove_link_symmetrically: layer is unavailable on both nodes");

        // Lock both nodes in id order to avoid deadlocks.
        if (a < b)
        {
            std::scoped_lock lock(nodes[a]->getMutex(), nodes[b]->getMutex());
            nodes[a]->remove_neighbor_nolock(b, layer);
            nodes[b]->remove_neighbor_nolock(a, layer);
        }
        else
        {
            std::scoped_lock lock(nodes[b]->getMutex(), nodes[a]->getMutex());
            nodes[a]->remove_neighbor_nolock(b, layer);
            nodes[b]->remove_neighbor_nolock(a, layer);
        }
    }
} // namespace minivec
