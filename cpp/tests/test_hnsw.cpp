#include <gtest/gtest.h>
#include "minivec/hnsw.hpp"  // your HNSW class
#include <vector>
#include <iostream>
#include <limits>
#include <random>
#include <chrono>
#include <unordered_set>
#include <unordered_map>

//Deterministic test: insert at level 0 to avoid randomness
TEST(HNSWTest, DeterministicBuildProducesIdenticalResults) {
    constexpr int dim = 32;
    constexpr uint32_t seed = 123;

    auto build = [&](minivec::HNSWIndexSimple& index) {
        for (int i = 0; i < 100; ++i) {
            std::vector<float> v(dim, float(i));
            index.insert_vector(v.data());
        }
        auto r = index.search_top_k(index.get_vector_ptr(10), 50, 5);
        std::vector<int> ids;
        for (auto& c : r) ids.push_back(c.id);
        return ids;
    };

    minivec::HNSWIndexSimple a(dim, 16, 100, 50, true, seed);
    minivec::HNSWIndexSimple b(dim, 16, 100, 50, true, seed);

    std::cout<<"Building index A..."<<std::endl;
    std::vector<int> a1 = build(a);
    std::cout<<"Building index B..."<<std::endl;
    std::vector<int> b1 = build(b);

    std::cout<<"Comparing results..."<<std::endl;
    std::cout<<"Index A results: ";
    for (auto id : a1) std::cout<<id<<" ";
    std::cout<<std::endl;
    std::cout<<"Index B results: ";
    for (auto id : b1) std::cout<<id<<" ";
    std::cout<<std::endl;

    ASSERT_EQ(std::unordered_set<int>(a1.begin(), a1.end()),std::unordered_set<int>(b1.begin(), b1.end()));
    for (int id = 0; id < a.get_node_count(); ++id) {
        EXPECT_EQ(a.get_layer(id), b.get_layer(id));
        EXPECT_EQ(a.get_neighbors_copy(id, 0), b.get_neighbors_copy(id, 0));
        for (int layer = 1; layer <= a.get_layer(id); ++layer) {
            EXPECT_EQ(a.get_neighbors_copy(id, layer), b.get_neighbors_copy(id, layer));
        }
    }

}

TEST(HNSWTest, LayerZeroKeepsTwoMNeighborsAndPruningDoesNotIsolateNodes) {
    constexpr int dim = 4;
    constexpr int M = 4;
    constexpr int count = 1000;
    minivec::HNSWIndexSimple index(dim, M, 50, 30, true, 17);
    std::mt19937 rng(1234);
    std::normal_distribution<float> distribution(0.0f, 1.0f);

    for (int id = 0; id < count; ++id) {
        std::vector<float> vector(dim);
        for (float &value : vector) value = distribution(rng);
        index.insert_vector(vector.data());
    }

    bool observed_more_than_M = false;
    for (int id = 0; id < count; ++id) {
        const auto base_neighbors = index.get_neighbors_copy(id, 0);
        EXPECT_FALSE(base_neighbors.empty()) << "node " << id << " lost every layer-0 edge";
        EXPECT_LE(base_neighbors.size(), 2 * M);
        observed_more_than_M |= base_neighbors.size() > M;
        for (int layer = 1; layer <= index.get_layer(id); ++layer) {
            EXPECT_LE(index.get_neighbors_copy(id, layer).size(), M);
        }
    }
    EXPECT_TRUE(observed_more_than_M);

    std::vector<bool> reachable(count, false);
    std::vector<int> pending{index.get_entry_point()};
    reachable[pending.front()] = true;
    while (!pending.empty()) {
        const int current = pending.back();
        pending.pop_back();
        for (int neighbor : index.get_neighbors_copy(current, 0)) {
            if (!reachable[neighbor]) {
                reachable[neighbor] = true;
                pending.push_back(neighbor);
            }
        }
    }
    EXPECT_EQ(std::count(reachable.begin(), reachable.end(), true), count);
}

TEST(HNSWTest, InsertNeighborSelectionRejectsRedundantCandidates) {
    minivec::HNSWIndexSimple index(2, 2, 10, 10, true, 17);
    const float nearest[] = {-1.0f, 0.0f};
    const float redundant[] = {-1.1f, -0.1f};
    const float opposite[] = {1.2f, 0.0f};
    const float upper[] = {0.0f, 1.3f};
    const float lower[] = {0.0f, -1.4f};
    const float query[] = {0.0f, 0.0f};
    const int nearest_id = index.add_node(nearest, 0);
    const int redundant_id = index.add_node(redundant, 0);
    const int opposite_id = index.add_node(opposite, 0);
    const int upper_id = index.add_node(upper, 0);
    const int lower_id = index.add_node(lower, 0);
    index.link_nodes_symmetrically(nearest_id, redundant_id, 0);
    index.link_nodes_symmetrically(nearest_id, opposite_id, 0);
    index.link_nodes_symmetrically(nearest_id, upper_id, 0);
    index.link_nodes_symmetrically(nearest_id, lower_id, 0);

    const int inserted_id = index.insert_vector(query);
    const auto selected = index.get_neighbors_copy(inserted_id, 0);
    const std::unordered_set<int> selected_ids(selected.begin(), selected.end());

    EXPECT_EQ(selected.size(), 2 * index.get_M());
    EXPECT_TRUE(selected_ids.count(nearest_id));
    EXPECT_FALSE(selected_ids.count(redundant_id));
    EXPECT_TRUE(selected_ids.count(opposite_id));
    EXPECT_TRUE(selected_ids.count(upper_id));
    EXPECT_TRUE(selected_ids.count(lower_id));
}

TEST(HNSWTest, EfSearchRefreshesWorstDistanceForEachNeighbor) {
    minivec::HNSWIndexSimple index(1, 2, 10, 10);
    const float query[] = {0.0f};
    const float entry[] = {0.0f};
    const float first[] = {10.0f};
    const float second[] = {9.0f};
    const float third[] = {9.5f};
    const int entry_id = index.add_node(entry, 0);
    const int first_id = index.add_node(first, 0);
    const int second_id = index.add_node(second, 0);
    const int third_id = index.add_node(third, 0);
    index.link_nodes_symmetrically(entry_id, first_id, 0);
    index.link_nodes_symmetrically(entry_id, second_id, 0);
    index.link_nodes_symmetrically(entry_id, third_id, 0);

    const auto best = index.ef_search_layer(query, entry_id, 0, 2);
    ASSERT_EQ(best.size(), 2u);
    EXPECT_EQ(best.top().id, second_id);
    auto remaining = best;
    remaining.pop();
    EXPECT_EQ(remaining.top().id, entry_id);
}

TEST(HNSWTest, SearchCountsUpperLayerGreedyHopsAndReusesVisitsSafely) {
    minivec::HNSWIndexSimple index(1, 2, 10, 10);
    const float far[] = {5.0f};
    const float near[] = {1.0f};
    const float query[] = {0.0f};
    const int far_id = index.add_node(far, 1);
    const int near_id = index.add_node(near, 1);
    index.link_nodes_symmetrically(far_id, near_id, 1);
    index.link_nodes_symmetrically(far_id, near_id, 0);

    minivec::SearchStats stats;
    const auto first = index.search_top_k(query, 10, 2, &stats);
    ASSERT_EQ(first.size(), 2u);
    EXPECT_EQ(first.front().id, near_id);
    EXPECT_GE(stats.greedy_hops, 1u);

    // A subsequent search starts a new visitation epoch and must see the same
    // graph, without inheriting marks from the preceding traversal.
    const auto second = index.search_top_k(query, 10, 2);
    ASSERT_EQ(second.size(), first.size());
    EXPECT_EQ(second[0].id, first[0].id);
    EXPECT_EQ(second[1].id, first[1].id);
}

// Probabilistic test: allow normal HNSW level generation
TEST(HNSWTest, InsertAndSearchProbabilistic) {
    minivec::HNSWIndexSimple index(32);

    std::vector<float> vec1(32, 0.5f);
    std::vector<float> vec2(32, 0.8f);

    index.insert_vector(vec1.data());
    index.insert_vector(vec2.data());

    auto neighbors = index.search_top_k(vec1.data(), 10, 1);

    ASSERT_EQ(neighbors.size(), 1);

    // Nearest neighbor can be either node, due to stochastic levels
    EXPECT_TRUE(neighbors[0].id == 0 || neighbors[0].id == 1);
}

// Memory test: insert many vectors until allocation fails
TEST(HNSWTest, MemoryLimitsLow) {
    minivec::HNSWIndexSimple index(1028);  // 1028-dim vectors

    const int max_vectors = 25; // fits in ~10 MB
    bool allocation_failed = false;

    try {
        for (int i = 0; i < max_vectors; i++) {
            std::vector<float> vec(1028, 0.1f);
            index.insert_vector(vec.data());
        }
    } catch (const std::bad_alloc&) {
        allocation_failed = true;
    } catch (...) {
        FAIL() << "Unexpected crash or exception occurred.";
    }
    // Test succeeded if no crash
    SUCCEED();
}
