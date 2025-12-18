#include <gtest/gtest.h>
#include "minivec/hnsw.hpp"
#include "minivec/dist.hpp"

TEST(HNSWSearchStats, StatsAreNonZero) {
    const int dim = 32;
    minivec::HNSWIndexSimple index(dim, 16, 100, 50);

    std::vector<std::vector<float>> db(100, std::vector<float>(dim, 0.1f));
    for (auto& v : db)
        index.insert_vector(v.data());

    minivec::SearchStats stats;
    auto res = index.search_top_k(db[0].data(), 50, 5, &stats);

    EXPECT_GT(stats.visited_nodes, 0);
    EXPECT_GT(stats.distance_calls, 0);
    EXPECT_FALSE(stats.layer_visits.empty());
}

TEST(HNSWSearchStats, DistanceCallsLowerBound) {
    constexpr int dim = 32;
    minivec::HNSWIndexSimple index(dim, 16, 100, 50);

    for (int i = 0; i < 200; ++i) {
        std::vector<float> v(dim, float(i));
        index.insert_vector(v.data());
    }

    std::vector<float> query(dim, 42.0f);

    minivec::SearchStats stats;
    index.search_top_k(query.data(), 50, 10, &stats);

    EXPECT_GE(stats.distance_calls, stats.visited_nodes);
}

TEST(HNSWSearchStats, LayerVisitsAreBounded) {
    minivec::HNSWIndexSimple index(32, 16, 100, 50);

    for (int i = 0; i < 200; ++i) {
        std::vector<float> v(32, float(i));
        index.insert_vector(v.data());
    }

    minivec::SearchStats stats;
    index.search_top_k(index.get_vector_ptr(0), 50, 5, &stats);

    int total_layer_visits = 0;
    for (auto& [layer, count] : stats.layer_visits)
        total_layer_visits += count;

    EXPECT_GE(total_layer_visits, stats.visited_nodes);
}

TEST(HNSWSearchStats, DeterministicStatsMatch) {
    minivec::HNSWIndexSimple index(
        32, 16, 100, 50,
        /*deterministic=*/true,
        /*seed=*/42);

    for (int i = 0; i < 200; ++i) {
        std::vector<float> v(32, float(i));
        index.insert_vector(v.data());
    }

    minivec::SearchStats s1, s2;

    index.search_top_k(index.get_vector_ptr(0), 50, 5, &s1);
    index.search_top_k(index.get_vector_ptr(0), 50, 5, &s2);

    EXPECT_EQ(s1.visited_nodes, s2.visited_nodes);
    EXPECT_EQ(s1.distance_calls, s2.distance_calls);
    EXPECT_EQ(s1.layer_visits, s2.layer_visits);
}
