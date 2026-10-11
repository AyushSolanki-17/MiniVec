#include <gtest/gtest.h>

#include "binturong/hnsw.hpp"
#include "binturong/dist.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace
{
    using binturong::Candidate;
    using binturong::HNSWIndexSimple;

    bool bitmap_has(const std::vector<std::uint64_t> &bitmap, binturong::NodeId id)
    {
        const auto node_id = static_cast<std::size_t>(id);
        return node_id / 64 < bitmap.size() &&
               (bitmap[node_id / 64] & (std::uint64_t{1} << (node_id % 64))) != 0;
    }

    TEST(HNSWFilteredSearch, StrategiesRespectBitmapAndReportSelectivityRecall)
    {
        constexpr int count = 10000;
        constexpr int dimension = 16;
        constexpr int k = 10;
        HNSWIndexSimple index(dimension, 12, 60, 40, true, 11);
        std::mt19937 rng(11);
        std::uniform_real_distribution<float> distribution(-1.0F, 1.0F);
        std::vector<std::vector<float>> vectors(count, std::vector<float>(dimension));
        for (int id = 0; id < count; ++id)
        {
            for (float &value : vectors[id])
                value = distribution(rng);
            std::unordered_map<std::string, std::int64_t> attrs{
                {"s50", id % 2 == 0},
                {"s05", id % 20 == 0},
                {"s005", id % 200 == 0},
            };
            ASSERT_EQ(index.insert_vector(vectors[id].data(), attrs), id);
        }

        std::vector<float> query(dimension);
        for (float &value : query)
            value = distribution(rng);

        for (const auto &scenario : std::vector<std::pair<std::string, double>>{
                 {"s50", 0.5}, {"s05", 0.05}, {"s005", 0.005}})
        {
            const auto filter = index.attribute_eq(scenario.first, 1);
            std::vector<Candidate> exact;
            for (int id = 0; id < count; ++id)
                if (bitmap_has(filter.words, id))
                    exact.emplace_back(id, binturong::compute_distance(
                        binturong::get_distance_metric("l2"), query.data(), vectors[id].data(), dimension));
            std::sort(exact.begin(), exact.end(), binturong::candidate_distance_less);
            exact.erase(exact.begin() + k, exact.end());

            binturong::SearchStats brute_stats;
            const auto brute = index.search_filtered_top_k(
                query.data(), 40, k, filter.words, "brute_force", 4, &brute_stats);
            ASSERT_EQ(brute.size(), exact.size());
            for (size_t i = 0; i < exact.size(); ++i)
                EXPECT_EQ(brute[i].id, exact[i].id);
            EXPECT_EQ(brute_stats.search_strategy, "brute_force");
            EXPECT_EQ(brute_stats.filter_rejected_candidates, count - filter.count);

            for (const auto &strategy : std::vector<std::string>{"filtered_traversal", "post_filter"})
            {
                binturong::SearchStats stats;
                const auto found = index.search_filtered_top_k(
                    query.data(), 40, k, filter.words, strategy, 8, &stats);
                size_t hits = 0;
                for (const auto &result : found)
                {
                    EXPECT_TRUE(bitmap_has(filter.words, result.id));
                    if (std::any_of(exact.begin(), exact.end(), [&](const Candidate &truth) {
                            return truth.id == result.id;
                        }))
                        ++hits;
                }
                const double recall = static_cast<double>(hits) / exact.size();
                std::cout << scenario.first << " selectivity=" << scenario.second
                          << " strategy=" << strategy << " recall=" << recall
                          << " returned=" << found.size()
                          << " rejected=" << stats.filter_rejected_candidates << '\n';
                EXPECT_EQ(stats.search_strategy, strategy);
                EXPECT_GT(stats.filter_rejected_candidates, 0);
            }
        }
    }
}
