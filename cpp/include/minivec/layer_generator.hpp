/**
 * @file layer_generator.hpp
 * @brief Layer generator for HNSW index
 * 
 * @details 
 * This file defines the layer generator for the HNSW index.
 */
#pragma once
#include <random>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <cstdint>
#include <limits>

namespace minivec {

// HNSWLevelGenerator
//
// Convention:
//  - p = probability to *go up one level* (small, e.g. 1/(M+1)).
//  - P(reaching level L) = p^L (i.e. probability node's level is >= L).
//  - So P(level == 0) = 1 - p (majority at level 0 when p is small).
//
// Use from_M(M) to get canonical p = 1 / (M + 1).
class HNSWLevelGenerator {
 public:
  explicit HNSWLevelGenerator(double p,
                              double eps = 1e-6,
                              std::optional<uint32_t> seed = std::nullopt)
      : p_(p), eps_(eps), seed_(seed)
  {
    if (!(p_ > 0.0 && p_ <= 1.0)) {
      throw std::invalid_argument("HNSWLevelGenerator: p must be in (0,1]");
    }
    if (!(eps_ > 0.0 && eps_ < 1.0)) {
      throw std::invalid_argument("HNSWLevelGenerator: eps must be in (0,1)");
    }
    compute_max_level();
  }

  static HNSWLevelGenerator from_M(int M,
                                   double eps = 1e-6,
                                   std::optional<uint32_t> seed = std::nullopt)
  {
    if (M <= 0)
      throw std::invalid_argument("HNSWLevelGenerator::from_M: M must be > 0");
    double p = 1.0 / (static_cast<double>(M) + 1.0);
    return HNSWLevelGenerator(p, eps, seed);
  }

  // Sample level in [0, max_level()] using the geometric model:
  // increment while uniform < p_ (p_ is probability to go up).
  int getRandomLayer() const
  {
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    auto &rng = get_thread_rng();

    int level = 0;
    // Continue to next level with probability p_ (small).
    while (level < max_level_ && uniform(rng) < p_) {
      ++level;
    }
    return level;
  }

  double p() const noexcept { return p_; }
  double eps() const noexcept { return eps_; }
  int max_level() const noexcept { return max_level_; }

 private:
  double p_;
  double eps_;
  int max_level_{0};
  std::optional<uint32_t> seed_;

  // Compute max_level such that p_^max_level <= eps => max_level >= log(eps)/log(p_)
  void compute_max_level()
  {
    if (p_ >= 1.0) {
      max_level_ = 0;
      return;
    }
    // Solve p_^L <= eps  ->  L * log(p) <= log(eps)  (both logs are negative)
    double raw = std::log(eps_) / std::log(p_);
    int cap = static_cast<int>(std::ceil(raw));
    if (cap < 1) cap = 1;
    constexpr int HARD_CAP = 64;
    if (cap > HARD_CAP) cap = HARD_CAP;
    max_level_ = cap;
  }

  // Thread-local RNG with optional deterministic seeding.
  std::mt19937& get_thread_rng() const
  {
    // Initialize RNG with non-deterministic seed by default.
    thread_local std::mt19937 rng = []() {
      std::random_device rd;
      std::seed_seq seq{rd(), rd(), rd(), rd()};
      return std::mt19937{seq};
    }();

    // If a deterministic seed was provided, mix it into the thread RNG once.
    thread_local bool seeded = false;
    if (!seeded && seed_.has_value()) {
      uintptr_t tid = reinterpret_cast<uintptr_t>(&rng);
      std::seed_seq seq{ seed_.value(),
                         static_cast<uint32_t>(tid & 0xffffffffu),
                         static_cast<uint32_t>((tid >> 32) & 0xffffffffu) };
      rng.seed(seq);
      seeded = true;
    } else if (!seeded) {
      seeded = true;
    }
    return rng;
  }
};
} // namespace minivec
