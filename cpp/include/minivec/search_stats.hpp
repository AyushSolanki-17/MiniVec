
#pragma once
#include <map>
#include <cstdint>
namespace minivec
{
    struct SearchStats
    {
        // Number of nodes discovered by ef-search traversals (a node can be
        // counted again in a different layer traversal).
        uint64_t visited_nodes = 0;
        // Distance computations made by ef-search traversals.
        uint64_t distance_calls = 0;
        // Successful greedy moves during upper-layer descent.
        uint64_t greedy_hops = 0;
        // ef-search node discoveries grouped by layer.
        std::map<int, uint64_t> layer_visits;
    };
} // namespace minivec
