
#pragma once
#include <map>
namespace minivec
{
    struct SearchStats
    {
        // total unique nodes visited
        uint64_t visited_nodes = 0;
        // distance computations           
        uint64_t distance_calls = 0;   
        // per-layer visits      
        std::map<int, uint64_t> layer_visits; 
    };
} // namespace minivec
