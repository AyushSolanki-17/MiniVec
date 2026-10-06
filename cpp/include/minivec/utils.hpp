/**
 * @file utils.hpp
 * @brief Utility functions for the HNSW index
 * @details
 * This file defines utility functions for the HNSW index.
 * Part of the MiniVec project.
 */
#pragma once
#include "hnsw_node.hpp"
#include <cstdint>
#include <memory>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace minivec
{
// Represents a search or graph candidate with an identifier and distance.
struct Candidate
{
  NodeId id;
  float distance;

  Candidate(NodeId i, float d) : id(i), distance(d) {}
};

inline bool candidate_distance_less(const Candidate &a, const Candidate &b)
{
  return a.distance < b.distance;
}

// Centralized check helper — throws std::out_of_range for invalid node ids.
inline void throw_if_invalid_node_id(const std::vector<std::unique_ptr<HNSWNodeSimple>> &nodes, NodeId id, const char *context)
{
  if (id < 0 || static_cast<std::uint64_t>(id) >= nodes.size())
  {
    std::ostringstream oss;
    oss << context << ": invalid node id " << id << " (nodes.size()=" << nodes.size() << ")";
    throw std::out_of_range(oss.str());
  }
  if (!nodes[static_cast<std::size_t>(id)])
  {
    std::ostringstream oss;
    oss << context << ": nodes[" << id << "] is nullptr (possible allocation error or moved-out node)";
    throw std::runtime_error(oss.str());
  }
}
struct MaxHeapCompare { bool operator()(const Candidate &a, const Candidate &b) const { return candidate_distance_less(a, b); } };
struct MinHeapCompare { bool operator()(const Candidate &a, const Candidate &b) const { return candidate_distance_less(b, a); } };
}
// // | " << __FILE__ << ":" << __LINE__ << "
// #define LOG(message)                              \
//     std::cout << "[INFO]--[" << __func__ << " ] " \
//               << message << std::endl << std::flush;
