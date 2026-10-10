// tests/test_hnsw_oracle.cpp
//
// Recall oracle: builds 20K-vector indexes on uniform and clustered data,
// compares search results against exact brute-force neighbors, and checks
// that at least 99.9% of nodes are reachable from the entry point on layer 0.
#include <gtest/gtest.h>
#include "minivec/hnsw.hpp"

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <queue>
#include <random>
#include <vector>

namespace {

constexpr int kDim = 64;
constexpr int kCount = 20000;
constexpr int kQueries = 200;
constexpr int kK = 10;
constexpr int kM = 16;
constexpr int kEfConstruction = 200;
constexpr int kEfSearch = 100;

using Matrix = std::vector<std::vector<float>>;

Matrix uniform_vectors(int count, std::mt19937 &rng) {
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    Matrix out(count, std::vector<float>(kDim));
    for (auto &v : out) {
        for (auto &x : v) x = dist(rng);
    }
    return out;
}

Matrix clustered_vectors(int count, const Matrix &centers, std::mt19937 &rng) {
    std::uniform_int_distribution<std::size_t> pick(0, centers.size() - 1);
    std::normal_distribution<float> noise(0.0f, 0.05f);
    Matrix out(count, std::vector<float>(kDim));
    for (auto &v : out) {
        const auto &c = centers[pick(rng)];
        for (int d = 0; d < kDim; ++d) v[d] = c[d] + noise(rng);
    }
    return out;
}

std::vector<minivec::NodeId> exact_knn(const std::vector<float> &q, const Matrix &db, int k) {
    std::vector<std::pair<float, minivec::NodeId>> dists;
    dists.reserve(db.size());
    for (std::size_t i = 0; i < db.size(); ++i) {
        float d = 0.0f;
        for (int j = 0; j < kDim; ++j) {
            const float diff = q[j] - db[i][j];
            d += diff * diff;
        }
        dists.emplace_back(d, static_cast<minivec::NodeId>(i));
    }
    std::partial_sort(dists.begin(), dists.begin() + k, dists.end());
    std::vector<minivec::NodeId> ids;
    for (int i = 0; i < k; ++i) ids.push_back(dists[i].second);
    return ids;
}

double mean_recall(minivec::HNSWIndexSimple &index, const Matrix &db, const Matrix &queries) {
    double total = 0.0;
    for (const auto &q : queries) {
        const auto truth = exact_knn(q, db, kK);
        const auto found = index.search_top_k(q.data(), kEfSearch, kK);
        int hits = 0;
        for (const auto &c : found) {
            if (std::find(truth.begin(), truth.end(), c.id) != truth.end()) ++hits;
        }
        total += static_cast<double>(hits) / kK;
    }
    return total / queries.size();
}

std::size_t layer0_reachable(const minivec::HNSWIndexSimple &index) {
    const auto count = static_cast<std::size_t>(index.get_node_count());
    std::vector<bool> seen(count, false);
    std::queue<minivec::NodeId> pending;
    pending.push(index.get_entry_point());
    seen[pending.front()] = true;
    std::size_t reached = 1;
    while (!pending.empty()) {
        const auto node = pending.front();
        pending.pop();
        for (const auto next : index.get_neighbors_copy(node, 0)) {
            if (!seen[next]) {
                seen[next] = true;
                ++reached;
                pending.push(next);
            }
        }
    }
    return reached;
}

std::unique_ptr<minivec::HNSWIndexSimple> build(const Matrix &db) {
    auto index = std::make_unique<minivec::HNSWIndexSimple>(
        kDim, kM, kEfConstruction, kEfSearch, /*deterministic_levelgen=*/true, /*levelgen_seed=*/7);
    for (const auto &v : db) index->insert_vector(v.data());
    return index;
}

}  // namespace

TEST(HNSWOracle, UniformRecallAndReachability) {
    std::mt19937 rng(1);
    const Matrix db = uniform_vectors(kCount, rng);
    const Matrix queries = uniform_vectors(kQueries, rng);
    auto index = build(db);

    const double recall = mean_recall(*index, db, queries);
    std::cout << "uniform recall@" << kK << " ef=" << kEfSearch << ": " << recall << std::endl;
    // Measured 0.864 on 2026-10-10 (Apple silicon, Release). Uniform 64-dim data
    // is hard for any HNSW: hnswlib 0.8.0 reaches 0.881 on the same distribution
    // and parameters. Threshold = measured - 0.02, rounded down.
    EXPECT_GE(recall, 0.84);
    EXPECT_GE(layer0_reachable(*index), static_cast<std::size_t>(kCount * 999 / 1000));
}

TEST(HNSWOracle, ClusteredRecallAndReachability) {
    std::mt19937 rng(2);
    const Matrix centers = uniform_vectors(50, rng);
    const Matrix db = clustered_vectors(kCount, centers, rng);
    const Matrix queries = clustered_vectors(kQueries, centers, rng);
    auto index = build(db);

    const double recall = mean_recall(*index, db, queries);
    std::cout << "clustered recall@" << kK << " ef=" << kEfSearch << ": " << recall << std::endl;
    // Measured 0.9995 on 2026-10-10 (Apple silicon, Release); threshold = measured - 0.02.
    EXPECT_GE(recall, 0.97);
    EXPECT_GE(layer0_reachable(*index), static_cast<std::size_t>(kCount * 999 / 1000));
}
