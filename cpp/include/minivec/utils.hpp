/**
 * @file utils.hpp
 * @brief Utility functions for the HNSW index
 * @details
 * This file defines utility functions for the HNSW index.
 * Part of the MiniVec project.
 */
#pragma once
#include "hnsw_node.hpp"
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace minivec
{
// Represents a search or graph candidate with an identifier and distance.
struct Candidate
{
  int id;
  float distance;

  Candidate(int i, float d) : id(i), distance(d) {}
};

// Centralized check helper — throws std::out_of_range for invalid node ids.
inline void throw_if_invalid_node_id(const std::vector<std::unique_ptr<HNSWNodeSimple>> &nodes, int id, const char *context)
{
  if (id < 0 || id >= static_cast<int>(nodes.size()))
  {
    std::ostringstream oss;
    oss << context << ": invalid node id " << id << " (nodes.size()=" << nodes.size() << ")";
    throw std::out_of_range(oss.str());
  }
  if (!nodes[id])
  {
    std::ostringstream oss;
    oss << context << ": nodes[" << id << "] is nullptr (possible allocation error or moved-out node)";
    throw std::runtime_error(oss.str());
  }
}
struct MaxHeapCompare { bool operator()(const Candidate&a,const Candidate&b) const { return a.distance < b.distance; } };
struct MinHeapCompare { bool operator()(const Candidate&a,const Candidate&b) const { return a.distance > b.distance; } };
}
// // | " << __FILE__ << ":" << __LINE__ << "
// #define LOG(message)                              \
//     std::cout << "[INFO]--[" << __func__ << " ] " \
//               << message << std::endl << std::flush;
