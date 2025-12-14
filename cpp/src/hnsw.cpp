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

#include <unordered_set>
#include <iostream>
#include <thread>
#include <mutex>
#include <shared_mutex>

#include <iostream>
#include <string>

// Constructs a simple HNSW index.
//
// Args:
//   dim_: Dimensionality of stored vectors.
//   M_: Maximum number of neighbors per layer.
//   efConstruction_: Size of dynamic candidate list during construction.
//   efSearch_: Default ef parameter for search.
HNSWIndexSimple::HNSWIndexSimple(int dim_, int M_, int efConstruction_, int efSearch_)
    : M(M_),
      max_layer(-1),
      entry_point(-1),
      dim(dim_),
      store(dim_),
      layer_gen(minivec::HNSWLevelGenerator::from_M(M_)),
      efConstruction(efConstruction_),
      efSearch(efSearch_) {}

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
int HNSWIndexSimple::add_node(const float *vec_vals, int layer)
{
    int id = store.add(vec_vals);

    // lock index structure while validating and mutating nodes vector and EP/max_layer
    std::lock_guard<std::shared_mutex> idx_lock(index_mtx);
    // Check for valid id.
    if (id != static_cast<int>(nodes.size()))
    {
        std::ostringstream oss;
        oss << "add_node: store.add() returned id " << id
            << " but expected " << nodes.size() << ". "
            << "Please ensure store.add() uses the same id space as nodes.";
        throw std::runtime_error(oss.str());
    }
    // layer + 1 for 0-based indexing of layers in the node.
    // create node on heap and push unique_ptr into vector
    nodes.emplace_back(std::make_unique<HNSWNodeSimple>(id, layer + 1, M));
    if (entry_point == -1)
    {
        entry_point = id;
        max_layer = layer;
    }
    return id;
}

// Returns a pointer to the stored vector for the given node id.
const float *HNSWIndexSimple::get_vector_ptr(int id) const
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
int HNSWIndexSimple::get_layer(int id) const
{
    throw_if_invalid_node_id(nodes, id, "get_layer");
    return nodes[id]->get_layer();
}

// Returns the current entry point id for the index.
const int HNSWIndexSimple::get_entry_point() const
{
    return entry_point;
}

// Returns the current maximum layer in the index.
int HNSWIndexSimple::get_max_layer() const
{
    return max_layer;
}

// Returns the number of nodes stored in the index.
int HNSWIndexSimple::get_node_count() const
{
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
    return efSearch;
}

// Prunes neighbors of a node at a given layer using the HNSW diversity rule.
//
// Keeps at most M neighbors that are both close to the node and diverse
// among themselves. Removes pruned neighbors symmetrically.
//
// Args:
//   id: Node id whose neighbors are pruned.
//   layer: Layer index where pruning is applied.
void HNSWIndexSimple::prune_neighbours(int id, int layer)
{
    // Validate id
    throw_if_invalid_node_id(nodes, id, "prune_neighbours");
    std::vector<int> nbrs = nodes[id]->get_neighbors(layer);
    if (nbrs.size() <= M)
        return; // Nothing to prune.

    // Build candidate list with distances to the node.
    std::vector<Candidate> candidates;
    candidates.reserve(nbrs.size());
    const float *id_ptr = get_vector_ptr(id);
    for (int n : nbrs)
    {
        float d = minivec::l2_squared_distance(id_ptr, get_vector_ptr(n), dim);
        candidates.push_back({n, d});
    }

    // Sort by increasing distance to the node.
    std::sort(candidates.begin(), candidates.end(),
              [](Candidate &a, Candidate &b)
              { return a.distance < b.distance; });

    int pool = std::min<int>((int)candidates.size(), std::max<int>(M * 2, (int)M + 8));

    // Diversity-based selection.
    std::vector<int> selected;
    selected.reserve(M);

    // cache center pointer once
    const float *center_ptr = get_vector_ptr(id);

    // Cache pointers for selected neighbors
    std::vector<const float *> selected_ptrs;
    selected_ptrs.reserve(M);

    for (int i = 0; i < pool && (int)selected.size() < M; ++i)
    {
        const Candidate &c = candidates[i];
        const float *c_ptr = get_vector_ptr(c.id);

        bool good = true;

        for (size_t j = 0; j < selected_ptrs.size(); ++j)
        {
            float ds = minivec::l2_squared_distance(c_ptr, selected_ptrs[j], dim);

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
    if ((int)selected.size() < M)
    {
        for (int i = 0; i < pool && (int)selected.size() < M; ++i)
        {
            int cand_id = candidates[i].id;
            // add if not already selected
            if (std::find(selected.begin(), selected.end(), cand_id) == selected.end())
                selected.push_back(cand_id);
        }
    }

    int n_nodes = static_cast<int>(nodes.size());

    // Sort selected for efficient lookup.
    std::sort(selected.begin(), selected.end());
    for (int old : nbrs)
    {
        // Skip if already selected.
        if (std::binary_search(selected.begin(), selected.end(), old))
            continue;

        if (old < 0 || old >= n_nodes)
            continue;
        if (old == id)
            continue;

        // Lock both nodes in id order to avoid deadlocks and remove neighbor links.
        if (id < old)
        {
            std::scoped_lock lock(nodes[id]->getMutex(), nodes[old]->getMutex());
            nodes[id]->remove_neighbor_nolock(old, layer);
            nodes[old]->remove_neighbor_nolock(id, layer);
        }
        else
        {
            std::scoped_lock lock(nodes[old]->getMutex(), nodes[id]->getMutex());
            nodes[id]->remove_neighbor_nolock(old, layer);
            nodes[old]->remove_neighbor_nolock(id, layer);
        }
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
int HNSWIndexSimple::insert_vector(const float *vec_vals)
{
    int new_layer = layer_gen.getRandomLayer();

    int id = add_node(vec_vals, new_layer);
    // Now size >= 2; start search from old_entry (not self).
    int current, local_max_layer;
    {
        std::lock_guard<std::shared_mutex> idx_lock(index_mtx);
        if (static_cast<int>(nodes.size()) == 1)
        {
            entry_point = id;
            max_layer = new_layer;
            return id;
        }
        current = entry_point;
        local_max_layer = max_layer;
    }

    // Validate old entry.
    if (current < 0 || current >= static_cast<int>(nodes.size()))
    {
        std::ostringstream oss;
        oss << "insert_vector: invalid old_entry " << current;
        throw std::runtime_error(oss.str());
    }

    // Greedy descent on upper layers.
    for (int layer = local_max_layer; layer >= new_layer + 1; layer--)
    {
        current = greedy_search_layer(vec_vals, current, layer);
    }

    // Connect on layers from new_layer down to 0.
    int connect_start_layer = std::min(new_layer, local_max_layer);
    for (int layer = connect_start_layer; layer >= 0; layer--)
    {
        std::priority_queue<Candidate, std::vector<Candidate>, MaxHeapCompare> neighbors_pq =
            ef_search_layer(vec_vals, current, layer, efConstruction);
        std::vector<Candidate> neighbors = filter_top_k(vec_vals, neighbors_pq, M);

        for (const Candidate &neighbor : neighbors)
        {
            // Skip self
            if (neighbor.id == id)
                continue;
            // Validate neighbor id
            throw_if_invalid_node_id(nodes, neighbor.id, "insert_vector: neighbor id");
            int a = id;
            int b = neighbor.id;
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
            std::vector<int> nbrs = nodes[neighbor.id]->get_neighbors(layer);
            if ((int)nbrs.size() > M)
            {
                prune_neighbours(neighbor.id, layer);
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
        std::lock_guard<std::shared_mutex> idx_lock(index_mtx);
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
int HNSWIndexSimple::greedy_search_layer(const float *query, int entry_id, int layer)
{
    throw_if_invalid_node_id(nodes, entry_id, "greedy_search_layer: entry_id");
    int n_nodes = static_cast<int>(nodes.size());
    int current = entry_id;
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
        if (!nodes[current])
        {
            std::ostringstream oss;
            oss << "greedy_search_layer: nodes[" << current << "] is nullptr";
            throw std::runtime_error(oss.str());
        }
        // cache current pointer once
        const float *pv_curr = store.ptr(current);
        float best_dist = minivec::l2_squared_distance(query, pv_curr, dim);
        // Explore neighbors
        std::vector<int> nbrs = nodes[current]->get_neighbors(layer);
        for (int neighbor : nbrs)
        {
            if (neighbor < 0 || neighbor >= n_nodes)
                continue;
            const float *pv_nei = store.ptr(neighbor);
            if (!pv_curr || !pv_nei)
                continue;
            
            float c_dist = minivec::l2_squared_distance(query, pv_nei, dim);
            // Check for improvement
            if (c_dist < best_dist)
            {
                best_dist = c_dist;
                current = neighbor;
                pv_curr = pv_nei;
                improved = true;
            }
        }
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
HNSWIndexSimple::ef_search_layer(const float *query, int entry_id, int layer, int ef)
{
    std::priority_queue<Candidate, std::vector<Candidate>, MaxHeapCompare> best_nodes;
    std::priority_queue<Candidate, std::vector<Candidate>, MinHeapCompare> candidates;
    int n_nodes = static_cast<int>(nodes.size());
    if (n_nodes == 0)
        return best_nodes;
    if (entry_id < 0 || entry_id >= n_nodes)
    {
        std::ostringstream oss;
        oss << "ef_search_layer: invalid entry_id " << entry_id;
        throw std::out_of_range(oss.str());
    }
    // Track visited nodes to avoid re-processing.
    std::vector<bool> visited(n_nodes, false);
    // Initialize with entry point.
    int current = entry_id;
    float curr_dist = minivec::l2_squared_distance(query, store.ptr(current), dim);
    candidates.emplace(current, curr_dist);
    best_nodes.emplace(current, curr_dist);
    visited[current] = true;
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
        if (!nodes[current])
        {
            std::cerr << "[ERR] ef_search_layer: nodes[" << current << "] is nullptr; skipping\n";
            continue;
        }
        std::vector<int> nbrs = nodes[current]->get_neighbors(layer);

        for (int neighbor : nbrs)
        {
            if (neighbor < 0 || neighbor >= n_nodes)
                continue;
            if (!visited[neighbor])
            {
                visited[neighbor] = true;
                // Compute distance to neighbor
                const float *pv_nei = store.ptr(neighbor);
                if (!pv_nei)
                    continue;
                float dist = minivec::l2_squared_distance(query, pv_nei, dim);
                candidates.emplace(neighbor, dist);

                if (static_cast<int>(best_nodes.size()) < ef)
                {
                    // Add neighbor to best_nodes
                    best_nodes.emplace(neighbor, dist);
                }
                else if (dist < worst_best_distance)
                {
                    // Replace worst best node
                    best_nodes.pop();
                    best_nodes.emplace(neighbor, dist);
                }
            }
        }
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
    const float *query, int ef, int k)
{
    std::shared_lock<std::shared_mutex> idx_shared_lock(index_mtx);
    if (nodes.empty() || entry_point < 0)
        return {};
    int current = entry_point;
    int lower_max_layer = max_layer;

    // Greedy search on upper layers.
    for (int layer = lower_max_layer; layer > 0; layer--)
    {
        current = greedy_search_layer(query, current, layer);
    }

    // EF search on layer 0.
    int effective_ef = (ef > 0) ? ef : efSearch;
    std::priority_queue<Candidate, std::vector<Candidate>, MaxHeapCompare> candidates = ef_search_layer(query, current, 0, effective_ef);

    return filter_top_k(query, candidates, k);
}

// Filters candidates to produce a diverse top-k result set.
//
// Applies the HNSW diversity criterion among candidates and recomputes
// distances to the query before final sorting.
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
    int k)
{
    std::vector<Candidate> top_k;
    top_k.reserve(k);

    std::vector<Candidate> candidates;
    candidates.reserve(candidates_pq.size());

    while (!candidates_pq.empty())
    {
        candidates.push_back(candidates_pq.top());
        candidates_pq.pop();
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate &a, const Candidate &b)
              { return a.distance < b.distance; });

    for (int i = 0; i < k && i < static_cast<int>(candidates.size()); ++i)
    {
        top_k.push_back(candidates[i]);
    }

    for (Candidate &c : top_k)
    {
        c.distance = minivec::l2_distance(query, get_vector_ptr(c.id), dim);
    }

    std::sort(top_k.begin(), top_k.end(),
              [](const Candidate &a, const Candidate &b)
              {
                  return a.distance < b.distance;
              });

    return top_k;
}

// Clears all data from the index and resets state.
void HNSWIndexSimple::clear()
{
    store.clear();
    nodes.clear();
    entry_point = -1;
    max_layer = -1;
}

// Thread-safe helper: copy neighbors for node node_id at given layer.
std::vector<int> HNSWIndexSimple::get_neighbors_copy(int node_id, int layer) const
{
    // Validate node id first
    throw_if_invalid_node_id(nodes, node_id, "get_neighbors_copy");

    // Acquire shared lock on the node to allow concurrent readers.
    std::shared_lock<std::shared_mutex> lock(nodes[node_id]->getMutex());

    // Call node's getter while holding shared lock. We expect get_neighbors returns
    // a copy (or a reference that we immediately copy); either way this is safe.

    auto aip = nodes[node_id]->get_neighbors(layer);
    return aip;
}

// Symmetric link: add edge a->b and b->a safely without deadlock.
void HNSWIndexSimple::link_nodes_symmetrically(int a, int b, int layer)
{
    // Validate ids
    throw_if_invalid_node_id(nodes, a, "link_nodes_symmetrically: a");
    throw_if_invalid_node_id(nodes, b, "link_nodes_symmetrically: b");
    // ignore self-links
    if (a == b)
        return;

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
void HNSWIndexSimple::remove_link_symmetrically(int a, int b, int layer)
{
    // Validate ids
    throw_if_invalid_node_id(nodes, a, "remove_link_symmetrically: a");
    throw_if_invalid_node_id(nodes, b, "remove_link_symmetrically: b");
    if (a == b)
        return;

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