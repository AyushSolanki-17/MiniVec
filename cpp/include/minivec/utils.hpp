/**
 * @file utils.hpp
 * @brief Utility functions for the HNSW index
 * @details
 * This file defines utility functions for the HNSW index.
 * Part of the MiniVec project.
 */
#pragma once

#include <cmath>
#include <random>
#include <thread>

// Generates random levels for HNSW nodes using a geometric-like distribution.
class HNSWLevelGenerator {
 public:
  // Initializes the generator with a per-layer probability and maximum layer.
  //
  // Args:
  //   layer_prob: Probability of promoting a node to the next layer.
  //   max_layer: Maximum layer index (exclusive upper bound for generation).
  HNSWLevelGenerator(float layer_prob = 0.5f, int max_layer = 16)
      : distribution_(layer_prob), max_layer_(max_layer) {}

  // Returns a random layer index for a node.
  //
  // Uses a thread-local RNG to avoid contention and ensure thread safety.
  //
  // Returns:
  //   The generated layer index in the range [0, max_layer_].
  int getRandomLayer() {
    thread_local static std::mt19937 rng(std::random_device{}());
    int level = 0;
    while (level < max_layer_ && distribution_(rng)) {
      ++level;
    }
    return level;
  }

 private:
  std::bernoulli_distribution distribution_;
  int max_layer_;
};

// Represents a search or graph candidate with an identifier and distance.
struct Candidate {
  int id;
  float distance;

  Candidate(int i, float d) : id(i), distance(d) {}
};

// Comparator for a min-heap based on distance.
struct CandidateCompare {
  bool operator()(const Candidate& a, const Candidate& b) const {
    // Smaller distance has higher priority.
    return a.distance > b.distance;
  }
};

// Comparator for a max-heap based on distance.
struct CandidateCompareInverse {
  bool operator()(const Candidate& a, const Candidate& b) const {
    // Larger distance has higher priority.
    return a.distance < b.distance;
  }
};