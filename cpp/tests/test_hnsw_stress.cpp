// tests/test_hnsw_stress.cpp
//
// Concurrency stress: 8 writer threads insert while 8 reader threads search the
// same index. Intended to run under ThreadSanitizer (ctest label "stress").
#include <gtest/gtest.h>
#include "minivec/hnsw.hpp"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <random>
#include <thread>
#include <vector>

namespace {

constexpr int kDim = 32;
constexpr int kWriters = 8;
constexpr int kReaders = 8;
constexpr int kInsertsPerWriter = 500;
constexpr int kTotal = kWriters * kInsertsPerWriter;
constexpr int kK = 10;
constexpr int kRecallQueries = 100;

std::vector<float> random_vector(std::mt19937 &rng) {
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> v(kDim);
    for (auto &x : v) x = dist(rng);
    return v;
}

float squared_l2(const float *a, const float *b) {
    float sum = 0.0f;
    for (int i = 0; i < kDim; ++i) {
        const float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

}  // namespace

TEST(HNSWStress, EightWritersEightReaders) {
    minivec::HNSWIndexSimple index(kDim, /*M=*/16, /*efConstruction=*/100, /*efSearch=*/64);
    std::atomic<int> writers_done{0};
    std::atomic<int> failures{0};
    std::atomic<long> searches{0};

    std::vector<std::thread> threads;
    for (int w = 0; w < kWriters; ++w) {
        threads.emplace_back([&, w] {
            std::mt19937 rng(1000 + w);
            for (int i = 0; i < kInsertsPerWriter; ++i) {
                const auto v = random_vector(rng);
                try {
                    index.insert_vector(v.data());
                } catch (...) {
                    ++failures;
                }
            }
            ++writers_done;
        });
    }
    for (int r = 0; r < kReaders; ++r) {
        threads.emplace_back([&, r] {
            std::mt19937 rng(2000 + r);
            while (writers_done.load() < kWriters) {
                const auto q = random_vector(rng);
                try {
                    for (const auto &c : index.search_top_k(q.data(), 64, kK)) {
                        if (c.id < 0 || c.id >= kTotal) ++failures;
                    }
                    ++searches;
                } catch (...) {
                    ++failures;
                }
            }
        });
    }
    for (auto &t : threads) t.join();

    ASSERT_EQ(failures.load(), 0);
    ASSERT_EQ(index.get_node_count(), kTotal);
    EXPECT_GT(searches.load(), 0);

    // Recall against brute force over the stored vectors.
    std::mt19937 rng(3000);
    double total = 0.0;
    for (int q = 0; q < kRecallQueries; ++q) {
        const auto query = random_vector(rng);
        std::vector<std::pair<float, minivec::NodeId>> exact;
        exact.reserve(kTotal);
        for (minivec::NodeId id = 0; id < kTotal; ++id) {
            exact.emplace_back(squared_l2(query.data(), index.get_vector_ptr(id)), id);
        }
        std::partial_sort(exact.begin(), exact.begin() + kK, exact.end());
        std::vector<minivec::NodeId> truth;
        for (int i = 0; i < kK; ++i) truth.push_back(exact[i].second);

        int hits = 0;
        for (const auto &c : index.search_top_k(query.data(), 64, kK)) {
            ASSERT_GE(c.id, 0);
            ASSERT_LT(c.id, kTotal);
            if (std::find(truth.begin(), truth.end(), c.id) != truth.end()) ++hits;
        }
        total += static_cast<double>(hits) / kK;
    }
    EXPECT_GE(total / kRecallQueries, 0.9);
}
