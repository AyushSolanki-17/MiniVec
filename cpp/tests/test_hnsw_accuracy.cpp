// tests/test_hnsw_accuracy.cpp
#include <gtest/gtest.h>
#include "minivec/hnsw.hpp"
#include "minivec/dist.hpp"
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>
#include <numeric>
#include <iostream>

using namespace std::chrono;

// compute top-k brute force
std::vector<int> brute_force_knn(const std::vector<float>& query, const std::vector<std::vector<float>>& db, int k) {
    int n = db.size();
    int dim = query.size();
    std::vector<std::pair<float, int>> dists;
    dists.reserve(n);
    for (int j = 0; j < n; ++j) {
        float d = 0.0f;
        for (int d_i = 0; d_i < dim; ++d_i) {
            float diff = query[d_i] - db[j][d_i];
            d += diff * diff;
        }
        dists.emplace_back(d, j);
    }
    std::nth_element(dists.begin(), dists.begin() + k, dists.end());
    std::sort(dists.begin(), dists.begin() + k);
    std::vector<int> out;
    out.reserve(k);
    for (int t = 0; t < k; ++t) {
        out.push_back(dists[t].second);
    }
    return out;
}


static float recall_at_k(const std::vector<int>& gt, const std::vector<int>& pred) {
    int k = gt.size();
    int found = 0;
    for (int id : pred) {
        if (std::find(gt.begin(), gt.end(), id) != gt.end()) ++found;
    }
    return float(found) / float(k);
}

TEST(HNSWAccuracy, RecallSmallDataset) {
    const int dim = 128;
    const int N = 10000;    // DB size (small for brute force)
    const int Q = 100;    // number of queries
    const int K = 25;

    // make data
    std::mt19937 rng(123);
    std::normal_distribution<float> dist(0.0f,1.0f);
    std::vector<std::vector<float>> db(N, std::vector<float>(dim));
    for (int i=0;i<N;++i) for (int d=0; d<dim; ++d) db[i][d] = dist(rng);

    // Build index
    // experiment: increase connectivity and construction ef
    minivec::HNSWIndexSimple index(dim, /*M=*/32, /*efConstruction=*/200, /*efSearch=*/100);
    //HNSWIndexSimple index(dim);
    for (int i=0;i<N;++i) index.insert_vector(db[i].data());

    // Query set
    std::vector<std::vector<float>> queries(Q, std::vector<float>(dim));
    for (int q=0;q<Q;++q) for (int d=0; d<dim; ++d) queries[q][d] = dist(rng);

    // Evaluate recall@K
    float sum_recall = 0.0f;
    for (int q=0;q<Q;++q) {
        auto gt = brute_force_knn(queries[q], db, K);
        auto out = index.search_top_k(queries[q].data(), 100, K);
        std::vector<int> pred;
        for (auto &c: out) pred.push_back(c.id);
        sum_recall += recall_at_k(gt, pred);
    }
    float mean_recall = sum_recall / Q;
    std::cout << "mean recall@" << K << " = " << mean_recall << std::endl;

    // a sanity check: recall should be reasonably high for small N (tweak threshold)
    ASSERT_GE(mean_recall, 0.7f);
}



TEST(HNSWDebug, LevelGeneratorIsolated) {
    // Test the generator directly, without HNSW index
    const int M = 16;
    const int N = 10000;
    
    auto gen = minivec::HNSWLevelGenerator::from_M(M);
    
    std::cout << "Generator config:\n";
    std::cout << "  p = " << gen.p() << "\n";
    std::cout << "  max_level = " << gen.max_level() << "\n";
    std::cout << "  eps = " << gen.eps() << "\n\n";
    
    std::map<int, int> level_counts;
    for (int i = 0; i < N; ++i) {
        int level = gen.getRandomLayer();
        level_counts[level]++;
    }
    
    std::cerr << "Direct generator test (no HNSW):\n";
    for (auto [lv, cnt] : level_counts) {
        double pct = 100.0 * cnt / N;
        std::cerr << "  Level " << lv << ": " << cnt << " (" << pct << "%)\n";
    }
    
    // Should have ~94% at level 0
    EXPECT_GT(level_counts[0], 9000);
}