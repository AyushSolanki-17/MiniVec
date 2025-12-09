// Implements the HNSWIndexSimple class declared in hnsw.hpp.
#include "minivec/hnsw.hpp"
#include "minivec/dist.hpp"

#include <unordered_set>
#include <iostream>

// Constructs a simple HNSW index.
//
// Args:
//   dim_: Dimensionality of stored vectors.
//   M_: Maximum number of neighbors per layer.
//   efConstruction_: Size of dynamic candidate list during construction.
//   efSearch_: Default ef parameter for search.
HNSWIndexSimple::HNSWIndexSimple(int dim_, int M_, int efConstruction_, int efSearch_)
    : M(M_),
      max_level(-1),
      entry_point(-1),
      dim(dim_),
      store(dim_),
      level_gen(0.5f, M_),
      efConstruction(efConstruction_),
      efSearch(efSearch_) {}

// Adds a new node with the given vector and level.
//
// The node is appended to the internal store and node list. The entry point
// and max_level are updated if this node becomes the new top-level entry.
//
// Args:
//   vec_vals: Pointer to the vector data of size dim.
//   level: Level assigned to this node (0-based external level).
//
// Returns:
//   The id of the newly added node.
int HNSWIndexSimple::add_node(const float* vec_vals, int level) {
    int id = store.add(vec_vals);
    // level + 1 for 0-based indexing of layers in the node.
    nodes.emplace_back(id, std::max(level + 1, max_level + 1), M);
    if (entry_point == -1 || level > max_level) {
        entry_point = id;
        max_level = level;
    }
    return id;
}

// Returns a pointer to the stored vector for the given node id.
const float* HNSWIndexSimple::get_vector_ptr(int id) const {
    return store.ptr(id);
}

// Returns the top level of the node with the given id.
const int HNSWIndexSimple::get_level(int id) const {
    return nodes[id].getLevel();
}

// Returns the current entry point id for the index.
const int HNSWIndexSimple::get_entry_point() const {
    return entry_point;
}

// Returns the current maximum level in the index.
const int HNSWIndexSimple::get_max_level() const {
    return max_level;
}

// Returns the number of nodes stored in the index.
const int HNSWIndexSimple::get_node_count() const {
    return nodes.size();
}

// Returns the dimensionality of vectors in this index.
const int HNSWIndexSimple::get_vector_dim() const {
    return dim;
}

// Returns the maximum number of neighbors per layer.
const int HNSWIndexSimple::get_M() const {
    return M;
}

// Returns the ef parameter used during index construction.
const int HNSWIndexSimple::get_efConstruction() const {
    return efConstruction;
}

// Returns the default ef parameter used during search.
const int HNSWIndexSimple::get_efSearch() const {
    return efSearch;
}

// Prunes neighbors of a node at a given level using the HNSW diversity rule.
//
// Keeps at most M neighbors that are both close to the node and diverse
// among themselves. Removes pruned neighbors symmetrically.
//
// Args:
//   id: Node id whose neighbors are pruned.
//   level: Layer index where pruning is applied.
void HNSWIndexSimple::prune_neighbours(int id, int level) {
    std::vector<int> nbrs = nodes[id].getNeighbors(level);
    if (nbrs.size() <= M) return;  // Nothing to prune.

    // Build candidate list with distances to the node.
    std::vector<Candidate> candidates;
    candidates.reserve(nbrs.size());
    for (int n : nbrs) {
        float d = minivec::l2_distance(get_vector_ptr(id), get_vector_ptr(n), dim);
        candidates.push_back({n, d});
    }

    // Sort by increasing distance to the node.
    std::sort(candidates.begin(), candidates.end(),
              [](Candidate& a, Candidate& b) { return a.distance < b.distance; });

    // Diversity-based selection.
    std::vector<int> selected;
    selected.reserve(M);

    for (Candidate c : candidates) {
        bool good = true;
        for (int s : selected) {
            float ds = minivec::l2_distance(get_vector_ptr(c.id), get_vector_ptr(s), dim);
            // HNSW diversity criterion: skip if too close to an already selected neighbor.
            if (ds < c.distance) {
                good = false;
                break;
            }
        }
        if (good) {
            selected.push_back(c.id);
            if (selected.size() == M) break;
        }
    }

    // Remove neighbors that are not in the selected set (symmetrically).
    std::unordered_set<int> keep(selected.begin(), selected.end());
    for (int old : nbrs) {
        if (!keep.count(old)) {
            nodes[id].removeNeighbor(old, level);
            nodes[old].removeNeighbor(id, level);
        }
    }
}

// Inserts a new vector into the index.
//
// Generates a random level for the new node, navigates from the entry point
// down to that level using greedy search, then connects the node to neighbors
// on each relevant layer using efConstruction.
//
// Args:
//   vec_vals: Pointer to the vector data of size dim.
//
// Returns:
//   The id of the inserted node.
int HNSWIndexSimple::insert_vector(const float* vec_vals) {
    int new_level = level_gen.getRandomLayer();
    int id = add_node(vec_vals, new_level);
    if (entry_point == id) {
        return id;
    }

    int current = entry_point;

    // Greedy descent on upper layers.
    for (int level = max_level; level >= new_level + 1; level--) {
        current = greedy_search_layer(vec_vals, current, level);
    }

    // Connect on layers from new_level down to 0.
    for (int level = std::min(new_level, max_level); level >= 0; level--) {
        std::priority_queue<Candidate, std::vector<Candidate>, CandidateCompareInverse> neighbors_pq =
            ef_search_layer(vec_vals, current, level, efConstruction);
        std::vector<Candidate> neighbors = filter_top_k(vec_vals, neighbors_pq, M);

        for (Candidate neighbor : neighbors) {
            nodes[id].addNeighbor(neighbor.id, level);
            nodes[neighbor.id].addNeighbor(id, level);

            // Local capacity control: keep at most M neighbors for the neighbor node.
            if (nodes[neighbor.id].getNeighbors(level).size() > M) {
                int removed_id = nodes[neighbor.id].getNeighbors(level).back();
                nodes[neighbor.id].removeNeighbor(removed_id, level);
                nodes[removed_id].removeNeighbor(neighbor.id, level);
            }
        }
    }

    if (new_level > max_level) {
        max_level = new_level;
        entry_point = id;
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
int HNSWIndexSimple::greedy_search_layer(const float* query, int entry_id, int layer) {
    int current = entry_id;
    bool improved = true;
    while (improved) {
        improved = false;
        for (int neighbor : nodes[current].getNeighbors(layer)) {
            if (minivec::l2_squared_distance(query, store.ptr(neighbor), dim) <
                minivec::l2_squared_distance(query, store.ptr(current), dim)) {
                current = neighbor;
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
std::priority_queue<Candidate, std::vector<Candidate>, CandidateCompareInverse>
HNSWIndexSimple::ef_search_layer(const float* query, int entry_id, int layer, int ef) {
    std::priority_queue<Candidate, std::vector<Candidate>, CandidateCompareInverse> best_nodes;
    std::priority_queue<Candidate, std::vector<Candidate>, CandidateCompare> candidates;
    std::vector<bool> visited(nodes.size(), false);

    int current = entry_id;
    float curr_dist = minivec::l2_squared_distance(query, store.ptr(current), dim);
    candidates.emplace(current, curr_dist);
    best_nodes.emplace(current, curr_dist);
    visited[current] = true;

    while (!candidates.empty()) {
        curr_dist = candidates.top().distance;
        current = candidates.top().id;
        candidates.pop();

        if (curr_dist > best_nodes.top().distance) {
            break;
        }

        for (int neighbor : nodes[current].getNeighbors(layer)) {
            if (neighbor < 0 || neighbor >= static_cast<int>(nodes.size())) {
                continue;
            }
            if (!visited[neighbor]) {
                visited[neighbor] = true;
                float dist = minivec::l2_squared_distance(query, store.ptr(neighbor), dim);
                candidates.emplace(neighbor, dist);

                if (static_cast<int>(best_nodes.size()) < ef) {
                    best_nodes.emplace(neighbor, dist);
                } else if (dist < best_nodes.top().distance) {
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
    const float* query, int ef, int k) {
    int current = entry_point;

    // Greedy search on upper layers.
    for (int layer = max_level; layer > 0; layer--) {
        current = greedy_search_layer(query, current, layer);
    }

    // EF search on layer 0.
    std::priority_queue<Candidate, std::vector<Candidate>, CandidateCompareInverse> candidates =
        ef_search_layer(query, current, 0, ef);

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
    const float* query,
    std::priority_queue<Candidate, std::vector<Candidate>, CandidateCompareInverse>& candidates_pq,
    int k) {
    std::vector<Candidate> top_k;
    top_k.reserve(k);

    std::vector<Candidate> candidates;
    candidates.reserve(candidates_pq.size());

    while (!candidates_pq.empty()) {
        candidates.push_back(candidates_pq.top());
        candidates_pq.pop();
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& a, const Candidate& b) { return a.distance < b.distance; });

    for (Candidate& c : candidates) {
        bool good = true;
        for (Candidate& selected : top_k) {
            float d = minivec::l2_distance(get_vector_ptr(c.id), get_vector_ptr(selected.id), dim);
            if (d < c.distance) {
                good = false;
                break;
            }
        }
        if (good) {
            top_k.push_back(c);
            if (static_cast<int>(top_k.size()) >= k) break;
        }
    }

    // Recompute distances to the query for final ordering.
    for (Candidate& c : top_k) {
        c.distance = minivec::l2_distance(query, get_vector_ptr(c.id), dim);
    }

    std::sort(top_k.begin(), top_k.end(),
              [](const Candidate& a, const Candidate& b) {
                  return a.distance < b.distance;
              });

    return top_k;
}

// Clears all data from the index and resets state.
void HNSWIndexSimple::clear() {
    store.clear();
    nodes.clear();
    entry_point = -1;
    max_level = -1;
}