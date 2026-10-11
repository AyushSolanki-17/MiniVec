#include <gtest/gtest.h>
#include "binturong/hnsw.hpp"
#include "binturong/dist.hpp"

TEST(HNSWSearchStats, StatsAreNonZero) {
    const int dim = 32;
    binturong::HNSWIndexSimple index(dim, 16, 100, 50);

    std::vector<std::vector<float>> db(100, std::vector<float>(dim, 0.1f));
    for (auto& v : db)
        index.insert_vector(v.data());

    binturong::SearchStats stats;
    auto res = index.search_top_k(db[0].data(), 50, 5, &stats);

    EXPECT_GT(stats.visited_nodes, 0);
    EXPECT_GT(stats.distance_calls, 0);
    EXPECT_FALSE(stats.layer_visits.empty());
}

TEST(HNSWSearchStats, DistanceCallsLowerBound) {
    constexpr int dim = 32;
    binturong::HNSWIndexSimple index(dim, 16, 100, 50);

    for (int i = 0; i < 200; ++i) {
        std::vector<float> v(dim, float(i));
        index.insert_vector(v.data());
    }

    std::vector<float> query(dim, 42.0f);

    binturong::SearchStats stats;
    index.search_top_k(query.data(), 50, 10, &stats);

    EXPECT_GE(stats.distance_calls, stats.visited_nodes);
}

TEST(HNSWSearchStats, LayerVisitsAreBounded) {
    binturong::HNSWIndexSimple index(32, 16, 100, 50);

    for (int i = 0; i < 200; ++i) {
        std::vector<float> v(32, float(i));
        index.insert_vector(v.data());
    }

    binturong::SearchStats stats;
    index.search_top_k(index.get_vector_ptr(0), 50, 5, &stats);

    int total_layer_visits = 0;
    for (auto& [layer, count] : stats.layer_visits)
        total_layer_visits += count;

    EXPECT_GE(total_layer_visits, stats.visited_nodes);
}

TEST(HNSWSearchStats, DeterministicStatsMatch) {
    binturong::HNSWIndexSimple index(
        32, 16, 100, 50,
        /*deterministic=*/true,
        /*seed=*/42);

    for (int i = 0; i < 200; ++i) {
        std::vector<float> v(32, float(i));
        index.insert_vector(v.data());
    }

    binturong::SearchStats s1, s2;

    index.search_top_k(index.get_vector_ptr(0), 50, 5, &s1);
    index.search_top_k(index.get_vector_ptr(0), 50, 5, &s2);

    EXPECT_EQ(s1.visited_nodes, s2.visited_nodes);
    EXPECT_EQ(s1.distance_calls, s2.distance_calls);
    EXPECT_EQ(s1.layer_visits, s2.layer_visits);
}

TEST(HNSWSearchStats, ExposesEstimatorSignals) {
    binturong::HNSWIndexSimple index(8, 8, 50, 30, true, 42);
    for (int i = 0; i < 80; ++i) {
        std::vector<float> v(8, float(i));
        index.insert_vector(v.data());
    }
    std::vector<float> query(8, 12.5f);
    binturong::SearchStats stats;
    const auto results = index.search_top_k(query.data(), 30, 6, &stats);

    EXPECT_GE(stats.distance_calls, stats.visited_nodes);
    uint64_t total_hops = 0;
    for (const auto &[layer, hops] : stats.hops_per_layer) {
        EXPECT_GT(layer, 0);
        total_hops += hops;
    }
    EXPECT_EQ(stats.greedy_hops, total_hops);
    EXPECT_GE(stats.candidate_queue_stability, 0.0);
    EXPECT_LE(stats.candidate_queue_stability, 1.0);
    EXPECT_EQ(stats.result_distance_gaps.size(), results.size() > 0 ? results.size() - 1 : 0);
    for (size_t i = 1; i < results.size(); ++i)
        EXPECT_FLOAT_EQ(stats.result_distance_gaps[i - 1], results[i].distance - results[i - 1].distance);
}
