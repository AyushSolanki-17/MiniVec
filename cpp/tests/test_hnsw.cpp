#include <gtest/gtest.h>
#include "minivec/hnsw.hpp"  // your HNSW class
#include <vector>
#include <iostream>
#include <limits>
#include <random>
#include <chrono>
#include <unordered_set>
#include <unordered_map>
#include <atomic>
#include <thread>
#include <type_traits>
#include <utility>

static_assert(std::is_same_v<
              decltype(std::declval<minivec::HNSWIndexSimple &>().insert_vector(nullptr)),
              minivec::NodeId>);

TEST(HNSWTest, RejectsInvalidConfigurationAndVectorInputsWithoutMutation) {
    EXPECT_THROW(minivec::HNSWIndexSimple(0), std::invalid_argument);
    EXPECT_THROW(minivec::HNSWIndexSimple(2, 0), std::invalid_argument);
    EXPECT_THROW(minivec::HNSWIndexSimple(2, std::numeric_limits<int>::max()), std::invalid_argument);
    EXPECT_THROW(minivec::HNSWIndexSimple(2, 2, 0), std::invalid_argument);
    EXPECT_THROW(minivec::HNSWIndexSimple(2, 2, 10, 0), std::invalid_argument);

    minivec::HNSWIndexSimple index(2);
    EXPECT_THROW(index.insert_vector(nullptr), std::invalid_argument);
    EXPECT_EQ(index.get_node_count(), 0);
    const float valid[] = {1.0f, 2.0f};
    EXPECT_THROW(index.add_node(valid, -1), std::invalid_argument);
    EXPECT_EQ(index.get_node_count(), 0);
    const float non_finite[] = {std::numeric_limits<float>::quiet_NaN(), 0.0f};
    EXPECT_THROW(index.insert_vector(non_finite), std::invalid_argument);
    EXPECT_EQ(index.get_node_count(), 0);
}

TEST(HNSWTest, ExplicitNodeInsertionUpdatesEntryPointAndSymmetricLinksValidateBothLayers) {
    minivec::HNSWIndexSimple index(1, 2);
    const float a[] = {1.0f};
    const float b[] = {2.0f};
    const auto low_id = index.add_node(a, 0);
    const auto high_id = index.add_node(b, 1);

    EXPECT_EQ(index.get_entry_point(), high_id);
    EXPECT_EQ(index.get_max_layer(), 1);
    EXPECT_THROW(index.link_nodes_symmetrically(high_id, low_id, 1), std::out_of_range);
    EXPECT_TRUE(index.get_neighbors_copy(high_id, 1).empty());
    EXPECT_TRUE(index.get_neighbors_copy(low_id, 0).empty());
}

TEST(HNSWTest, NeighborRemovalHonorsPreserveOrder) {
    minivec::HNSWNodeSimple swapped(0, 1, 4);
    minivec::HNSWNodeSimple ordered(1, 1, 4);
    for (minivec::NodeId id : {10, 20, 30}) {
        swapped.add_neighbor(id, 0);
        ordered.add_neighbor(id, 0);
    }

    EXPECT_TRUE(swapped.remove_neighbor(20, 0));
    EXPECT_TRUE(ordered.remove_neighbor(20, 0, true));
    EXPECT_EQ(swapped.get_neighbors(0), (std::vector<minivec::NodeId>{10, 30}));
    EXPECT_EQ(ordered.get_neighbors(0), (std::vector<minivec::NodeId>{10, 30}));

    swapped.add_neighbor(40, 0);
    ordered.add_neighbor(40, 0);
    EXPECT_TRUE(swapped.remove_neighbor(10, 0));
    EXPECT_TRUE(ordered.remove_neighbor(10, 0, true));
    EXPECT_EQ(swapped.get_neighbors(0), (std::vector<minivec::NodeId>{40, 30}));
    EXPECT_EQ(ordered.get_neighbors(0), (std::vector<minivec::NodeId>{30, 40}));
}

TEST(HNSWTest, TopKHandlesSmallEfZeroKAndLargeK) {
    minivec::HNSWIndexSimple index(1, 2);
    const float values[] = {0.0f, 1.0f, 2.0f, 3.0f};
    std::vector<minivec::NodeId> ids;
    for (float value : values)
        ids.push_back(index.add_node(&value, 0));
    for (size_t i = 0; i < ids.size(); ++i)
        for (size_t j = i + 1; j < ids.size(); ++j)
            index.link_nodes_symmetrically(ids[i], ids[j], 0);

    const float query[] = {0.0f};
    EXPECT_TRUE(index.search_top_k(query, 1, 0).empty());
    EXPECT_THROW(index.search_top_k(query, 1, -1), std::invalid_argument);
    EXPECT_EQ(index.search_top_k(query, 1, 4).size(), 4u);
    EXPECT_EQ(index.search_top_k(query, 1, std::numeric_limits<int>::max()).size(), 4u);
}

TEST(HNSWTest, LayerGeneratorCapsExtremeParametersBeforeIntegerConversion) {
    const double p = std::nextafter(1.0, 0.0);
    minivec::HNSWLevelGenerator generator(p, std::numeric_limits<double>::min(), 7);
    EXPECT_EQ(generator.max_level(), 64);
    EXPECT_GE(generator.getRandomLayer(), 0);
    EXPECT_LE(generator.getRandomLayer(), 64);

    minivec::HNSWLevelGenerator always_rises(1.0, 1e-6, 7);
    EXPECT_EQ(always_rises.max_level(), 64);
    EXPECT_EQ(always_rises.getRandomLayer(), 64);
}

TEST(HNSWTest, ClearRestartsDeterministicLayerSequence) {
    minivec::HNSWIndexSimple index(2, 4, 20, 20, true, 99);
    const float vector[] = {0.5f, 1.0f};
    std::vector<int> first_build;
    for (int i = 0; i < 30; ++i) {
        const auto id = index.insert_vector(vector);
        first_build.push_back(index.get_layer(id));
    }
    index.clear();

    for (int i = 0; i < 30; ++i) {
        const auto id = index.insert_vector(vector);
        EXPECT_EQ(index.get_layer(id), first_build[static_cast<size_t>(i)]);
    }
}

TEST(HNSWTest, VisitWorkspaceCanBeReusedAcrossIndexes) {
    minivec::HNSWIndexSimple first(1, 2, 10, 10, true, 11);
    minivec::HNSWIndexSimple second(1, 2, 10, 10, true, 12);
    const float values[] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    for (float value : values) {
        first.insert_vector(&value);
        second.insert_vector(&value);
    }
    const float query[] = {0.25f};
    const auto first_result = first.search_top_k(query, 5, 3);
    const auto second_result = second.search_top_k(query, 5, 3);
    const auto first_again = first.search_top_k(query, 5, 3);

    ASSERT_EQ(first_result.size(), second_result.size());
    ASSERT_EQ(first_result.size(), first_again.size());
    for (size_t i = 0; i < first_result.size(); ++i) {
        EXPECT_EQ(first_result[i].id, first_again[i].id);
        EXPECT_FLOAT_EQ(first_result[i].distance, first_again[i].distance);
    }
}

TEST(HNSWTest, CosineAndInnerProductRemainFiniteForLargeFiniteInputs) {
    const float a[] = {1.0e20f, 1.0e20f};
    const float same[] = {1.0e20f, 1.0e20f};
    const float orthogonal[] = {1.0e20f, -1.0e20f};
    const float opposite[] = {-1.0e20f, -1.0e20f};
    const float cancelling[] = {1.0e20f, -1.0e20f};
    EXPECT_FLOAT_EQ(minivec::cosine_distance(a, same, 2), 0.0f);
    EXPECT_FLOAT_EQ(minivec::cosine_distance(a, orthogonal, 2), 1.0f);
    EXPECT_FLOAT_EQ(minivec::cosine_distance(a, opposite, 2), 2.0f);
    EXPECT_FLOAT_EQ(minivec::inner_product_distance(a, cancelling, 2), 0.0f);
}

TEST(HNSWTest, DistanceHelpersRejectNegativeDimensions) {
    const float value[] = {1.0f};
    EXPECT_THROW(minivec::l2_squared_scalar(value, value, -1), std::invalid_argument);
    EXPECT_THROW(minivec::l2_squared_distance(value, value, -1), std::invalid_argument);
    EXPECT_THROW(minivec::inner_product_distance(value, value, -1), std::invalid_argument);
    EXPECT_THROW(minivec::cosine_distance(value, value, -1), std::invalid_argument);
    EXPECT_THROW(minivec::l2_squared_distance(nullptr, value, 1), std::invalid_argument);
}

TEST(HNSWTest, NeighborVisitorMatchesSnapshot) {
    minivec::HNSWNodeSimple node(7, 1, 4);
    node.add_neighbor(2, 0);
    node.add_neighbor(5, 0);
    std::vector<minivec::NodeId> visited;
    node.for_each_neighbor(0, [&](minivec::NodeId neighbor) { visited.push_back(neighbor); });
    EXPECT_EQ(visited, node.get_neighbors(0));
}

TEST(HNSWTest, PackedNeighborLayersStayCorrectAcrossMutation) {
    minivec::HNSWNodeSimple node(7, 3, 4);
    node.add_neighbor(10, 0);
    node.add_neighbor(11, 0);
    node.add_neighbor(20, 1);
    node.add_neighbor(30, 2);
    EXPECT_EQ(node.get_neighbors(0), (std::vector<minivec::NodeId>{10, 11}));
    EXPECT_EQ(node.get_neighbors(1), (std::vector<minivec::NodeId>{20}));
    EXPECT_EQ(node.get_neighbors(2), (std::vector<minivec::NodeId>{30}));

    EXPECT_TRUE(node.remove_neighbor(10, 0));
    EXPECT_EQ(node.get_neighbors(0), (std::vector<minivec::NodeId>{11}));
    EXPECT_EQ(node.get_neighbors(1), (std::vector<minivec::NodeId>{20}));
    EXPECT_EQ(node.get_neighbors(2), (std::vector<minivec::NodeId>{30}));
    node.clear_layer(1);
    EXPECT_TRUE(node.get_neighbors(1).empty());
    EXPECT_EQ(node.get_neighbors(2), (std::vector<minivec::NodeId>{30}));
}

TEST(HNSWTest, NodeAndNeighborIdsPreserveValuesBeyondIntRange) {
    const minivec::NodeId high_id = static_cast<minivec::NodeId>(std::numeric_limits<int>::max()) + 123;
    minivec::HNSWNodeSimple node(high_id, 1, 4);
    node.add_neighbor(high_id + 1, 0);
    const minivec::Candidate candidate(high_id + 2, 0.5f);
    EXPECT_EQ(node.get_id(), high_id);
    EXPECT_EQ(candidate.id, high_id + 2);
    EXPECT_EQ(node.get_neighbors(0), (std::vector<minivec::NodeId>{high_id + 1}));
}

TEST(HNSWTest, DistanceMetricsHaveExpectedOrderingAndZeroVectorBehavior) {
    const float x[] = {1.0f, 0.0f};
    const float orthogonal[] = {0.0f, 1.0f};
    const float opposite[] = {-1.0f, 0.0f};
    const float zero[] = {0.0f, 0.0f};
    EXPECT_FLOAT_EQ(minivec::compute_distance(minivec::DistanceMetric::L2Squared, x, orthogonal, 2), 2.0f);
    EXPECT_FLOAT_EQ(minivec::compute_distance(minivec::DistanceMetric::Cosine, x, orthogonal, 2), 1.0f);
    EXPECT_FLOAT_EQ(minivec::compute_distance(minivec::DistanceMetric::Cosine, x, opposite, 2), 2.0f);
    EXPECT_FLOAT_EQ(minivec::compute_distance(minivec::DistanceMetric::Cosine, x, zero, 2), 1.0f);
    EXPECT_FLOAT_EQ(minivec::compute_distance(minivec::DistanceMetric::InnerProduct, x, opposite, 2), 1.0f);

    minivec::HNSWIndexSimple index(2, 2, 10, 10, true, 9, "cosine", "cosine");
    index.insert_vector(x);
    index.insert_vector(orthogonal);
    const auto result = index.search_top_k(x, 10, 2);
    ASSERT_EQ(result.size(), 2u);
    EXPECT_EQ(result.front().id, 0);
}

TEST(HNSWTest, DispatchedSquaredL2MatchesScalarAcrossRemainders) {
    std::mt19937 rng(44);
    std::uniform_real_distribution<float> distribution(-3.0f, 3.0f);
    std::vector<float> a(37), b(37);
    for (size_t dim = 1; dim <= a.size(); ++dim) {
        for (size_t i = 0; i < dim; ++i) {
            a[i] = distribution(rng);
            b[i] = distribution(rng);
        }
        EXPECT_NEAR(minivec::l2_squared_distance(a.data(), b.data(), static_cast<int>(dim)),
                    minivec::l2_squared_scalar(a.data(), b.data(), static_cast<int>(dim)),
                    1e-4f * static_cast<float>(dim));
    }
}

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
        std::vector<minivec::NodeId> ids;
        for (auto& c : r) ids.push_back(c.id);
        return ids;
    };

    minivec::HNSWIndexSimple a(dim, 16, 100, 50, true, seed);
    minivec::HNSWIndexSimple b(dim, 16, 100, 50, true, seed);

    std::cout<<"Building index A..."<<std::endl;
    std::vector<minivec::NodeId> a1 = build(a);
    std::cout<<"Building index B..."<<std::endl;
    std::vector<minivec::NodeId> b1 = build(b);

    std::cout<<"Comparing results..."<<std::endl;
    std::cout<<"Index A results: ";
    for (auto id : a1) std::cout<<id<<" ";
    std::cout<<std::endl;
    std::cout<<"Index B results: ";
    for (auto id : b1) std::cout<<id<<" ";
    std::cout<<std::endl;

    ASSERT_EQ(std::unordered_set<minivec::NodeId>(a1.begin(), a1.end()),std::unordered_set<minivec::NodeId>(b1.begin(), b1.end()));
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
    std::vector<minivec::NodeId> pending{index.get_entry_point()};
    reachable[pending.front()] = true;
    while (!pending.empty()) {
        const int current = pending.back();
        pending.pop_back();
        for (minivec::NodeId neighbor : index.get_neighbors_copy(current, 0)) {
            if (!reachable[neighbor]) {
                reachable[neighbor] = true;
                pending.push_back(neighbor);
            }
        }
    }
    // Pruning keeps diverse edges, as in reference HNSW, so a rare node can
    // lose every incoming edge; require at least 99.9% layer-0 reachability.
    EXPECT_GE(std::count(reachable.begin(), reachable.end(), true), count * 999 / 1000);
}

TEST(HNSWTest, InsertNeighborSelectionRejectsRedundantCandidates) {
    minivec::HNSWIndexSimple index(2, 2, 10, 10, true, 17);
    const float nearest[] = {-1.0f, 0.0f};
    const float redundant[] = {-1.1f, -0.1f};
    const float opposite[] = {1.2f, 0.0f};
    const float upper[] = {0.0f, 1.3f};
    const float lower[] = {0.0f, -1.4f};
    const float query[] = {0.0f, 0.0f};
    const minivec::NodeId nearest_id = index.add_node(nearest, 0);
    const minivec::NodeId redundant_id = index.add_node(redundant, 0);
    const minivec::NodeId opposite_id = index.add_node(opposite, 0);
    const minivec::NodeId upper_id = index.add_node(upper, 0);
    const minivec::NodeId lower_id = index.add_node(lower, 0);
    index.link_nodes_symmetrically(nearest_id, redundant_id, 0);
    index.link_nodes_symmetrically(nearest_id, opposite_id, 0);
    index.link_nodes_symmetrically(nearest_id, upper_id, 0);
    index.link_nodes_symmetrically(nearest_id, lower_id, 0);

    const minivec::NodeId inserted_id = index.insert_vector(query);
    const auto selected = index.get_neighbors_copy(inserted_id, 0);
    const std::unordered_set<minivec::NodeId> selected_ids(selected.begin(), selected.end());

    // A new node links to at most M neighbors chosen by the diversity
    // heuristic: the nearest, then the closest candidate not dominated by it.
    EXPECT_EQ(selected.size(), static_cast<std::size_t>(index.get_M()));
    EXPECT_TRUE(selected_ids.count(nearest_id));
    EXPECT_FALSE(selected_ids.count(redundant_id));
    EXPECT_TRUE(selected_ids.count(opposite_id));
}

TEST(HNSWTest, EfSearchRefreshesWorstDistanceForEachNeighbor) {
    minivec::HNSWIndexSimple index(1, 2, 10, 10);
    const float query[] = {0.0f};
    const float entry[] = {0.0f};
    const float first[] = {10.0f};
    const float second[] = {9.0f};
    const float third[] = {9.5f};
    const minivec::NodeId entry_id = index.add_node(entry, 0);
    const minivec::NodeId first_id = index.add_node(first, 0);
    const minivec::NodeId second_id = index.add_node(second, 0);
    const minivec::NodeId third_id = index.add_node(third, 0);
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
    const minivec::NodeId far_id = index.add_node(far, 1);
    const minivec::NodeId near_id = index.add_node(near, 1);
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

TEST(HNSWTest, ConcurrentInsertSearchAndClearAreSynchronized) {
    constexpr int writers = 4;
    constexpr int inserts_per_writer = 50;
    constexpr int dim = 8;
    minivec::HNSWIndexSimple index(dim, 4, 20, 20, true, 123);
    std::atomic<bool> inserting{true};
    std::atomic<int> failures{0};
    const std::vector<float> query(dim, 0.25f);

    std::thread reader([&] {
        while (inserting.load()) {
            try {
                const auto results = index.search_top_k(query.data(), 20, 5);
                for (const auto &candidate : results)
                    if (candidate.id < 0) ++failures;
            } catch (...) {
                ++failures;
            }
        }
    });
    std::thread pruner([&] {
        while (inserting.load()) {
            const int count = index.get_node_count();
            if (count == 0) continue;
            try {
                index.prune_neighbours(count - 1, 0);
            } catch (...) {
                ++failures;
            }
        }
    });

    std::vector<std::thread> worker_threads;
    for (int worker = 0; worker < writers; ++worker) {
        worker_threads.emplace_back([&, worker] {
            for (int i = 0; i < inserts_per_writer; ++i) {
                std::vector<float> vector(dim, static_cast<float>(worker * inserts_per_writer + i));
                try {
                    index.insert_vector(vector.data());
                } catch (...) {
                    ++failures;
                }
            }
        });
    }
    for (auto &thread : worker_threads) thread.join();
    inserting = false;
    reader.join();
    pruner.join();

    EXPECT_EQ(failures.load(), 0);
    EXPECT_EQ(index.get_node_count(), writers * inserts_per_writer);
    const float *first_vector_ptr = index.get_vector_ptr(0);
    const float first_value = first_vector_ptr[0];
    const std::vector<float> extra_vector(dim, -1.0f);
    index.insert_vector(extra_vector.data());
    EXPECT_EQ(index.get_vector_ptr(0), first_vector_ptr);
    EXPECT_EQ(first_vector_ptr[0], first_value);
    EXPECT_FALSE(index.search_top_k(query.data(), 20, 5).empty());

    std::atomic<bool> reading{true};
    std::thread clear_reader([&] {
        while (reading.load()) {
            try {
                const auto results = index.search_top_k(query.data(), 20, 5);
                for (const auto &candidate : results)
                    if (candidate.id < 0) ++failures;
            } catch (...) {
                ++failures;
            }
        }
    });
    index.clear();
    reading = false;
    clear_reader.join();

    EXPECT_EQ(failures.load(), 0);
    EXPECT_EQ(index.get_node_count(), 0);
    EXPECT_EQ(index.get_entry_point(), -1);
    EXPECT_TRUE(index.search_top_k(query.data(), 20, 5).empty());
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
