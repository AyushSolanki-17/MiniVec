
#pragma once
#include <map>
#include <cstdint>
#include <vector>
namespace minivec
{
    struct SearchStats
    {
        // Number of nodes discovered by ef-search traversals (a node can be
        // counted again in a different layer traversal).
        uint64_t visited_nodes = 0;
        // Distance computations made during greedy descent, ef-search, and final scoring.
        uint64_t distance_calls = 0;
        // Successful greedy moves during upper-layer descent.
        uint64_t greedy_hops = 0;
        // Successful greedy moves grouped by layer.
        std::map<int, uint64_t> hops_per_layer;
        // ef-search node discoveries grouped by layer.
        std::map<int, uint64_t> layer_visits;
        // Fraction of discovered nodes that did not change ef-search's retained best set.
        double candidate_queue_stability = 1.0;
        // Adjacent distance gaps in the final results, sorted by distance.
        std::vector<float> result_distance_gaps;
    };
} // namespace minivec
