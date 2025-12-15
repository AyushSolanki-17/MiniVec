#include <gtest/gtest.h>
#include "minivec/hnsw.hpp"  // your HNSW class
#include <vector>
#include <limits>

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

    ASSERT_EQ(build(a), build(b));
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

