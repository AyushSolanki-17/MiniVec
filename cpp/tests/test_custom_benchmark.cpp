#include <gtest/gtest.h>
#include "minivec/hnsw.hpp"
#include "minivec/dist.hpp"

#include <vector>
#include <random>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <map>

/* ----------------------- Brute Force KNN ----------------------- */

static std::vector<int> brute_force_knn(
    const std::vector<float>& query,
    const std::vector<std::vector<float>>& db,
    int k)
{
    std::vector<std::pair<float,int>> dists;
    dists.reserve(db.size());

    for (int i = 0; i < (int)db.size(); ++i) {
        // CHange function according to distance metric
        float d = minivec::l2_squared_distance(
            query.data(), db[i].data(), query.size());
        dists.emplace_back(d, i);
    }

    std::nth_element(dists.begin(), dists.begin() + k, dists.end());
    std::sort(dists.begin(), dists.begin() + k);

    std::vector<int> out;
    out.reserve(k);
    for (int i = 0; i < k; ++i)
        out.push_back(dists[i].second);

    return out;
}
double brute_force_latency(
    const std::vector<std::vector<float>>& queries,
    const std::vector<std::vector<float>>& db,
    int K)
{
    auto t0 = std::chrono::high_resolution_clock::now();
    for (const auto& q : queries)
        brute_force_knn(q, db, K);
    auto t1 = std::chrono::high_resolution_clock::now();

    return std::chrono::duration<double, std::milli>(t1 - t0).count()
           / queries.size();
}
double percentile(std::vector<double>& v, double p) {
    std::sort(v.begin(), v.end());
    size_t idx = static_cast<size_t>(p * v.size());
    if (idx >= v.size()) idx = v.size() - 1;
    return v[idx];
}

static float recall_at_k(
    const std::vector<int>& gt,
    const std::vector<int>& pred)
{
    int found = 0;
    for (int id : pred)
        if (std::find(gt.begin(), gt.end(), id) != gt.end())
            ++found;
    return float(found) / gt.size();
}

/* ----------------------- Parameter Grid ----------------------- */

struct HNSWParams {
    int M;
    int efC;
    int efS;
};


const std::vector<int> Ms  = {16, 32, 64};
const std::vector<int> efCs = {200, 400};
const std::vector<int> efSs = {100, 200};
const std::vector<int> Ks   = {1, 5, 10, 25};


#include <fstream>

struct CSVWriter {
    std::ofstream out;
    explicit CSVWriter(const std::string& path) : out(path) {
        out << "M,efConstruction,efSearch,K,Recall,AvgLatencyMs,P50Ms,P95Ms,BruteForceMs\n";
    }
    void write(int M, int efC, int efS, int K,
               float recall, double avg, double p50, double p95, double bf) {
        out << M << "," << efC << "," << efS << ","
            << K << "," << recall << ","
            << avg << "," << p50 << "," << p95 << ","
            << bf << "\n";
    }
};


/* ----------------------- Main Benchmark Test ----------------------- */

 TEST(HNSWBenchmark, RecallLatencySweep)
{
    const int dim = 128;
    const int N   = 10'000;
    const int Q   = 100;

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    /* Generate database */
    std::vector<std::vector<float>> db(N, std::vector<float>(dim));
    for (auto& v : db)
        for (float& x : v)
            x = dist(rng);

    /* Generate queries */
    std::vector<std::vector<float>> queries(Q, std::vector<float>(dim));
    for (auto& q : queries)
        for (float& x : q)
            x = dist(rng);

    CSVWriter csv("hnsw_benchmark.csv");

    for (int M : Ms)
    for (int efC : efCs)
    for (int efS : efSs)
    {
        minivec::HNSWIndexSimple index(dim, M, efC, efS);
        for (int i = 0; i < N; ++i)
            index.insert_vector(db[i].data());

        for (int K : Ks)
        {
            std::vector<double> latencies;
            float recall_sum = 0.0f;

            for (int qi = 0; qi < Q; ++qi)
            {
                auto gt = brute_force_knn(queries[qi], db, K);

                auto t0 = std::chrono::high_resolution_clock::now();
                auto out = index.search_top_k(queries[qi].data(), efS, K);
                auto t1 = std::chrono::high_resolution_clock::now();

                double ms =
                    std::chrono::duration<double, std::milli>(t1 - t0).count();
                latencies.push_back(ms);

                std::vector<int> pred;
                for (auto& c : out) pred.push_back(c.id);
                recall_sum += recall_at_k(gt, pred);
            }

            float mean_recall = recall_sum / Q;
            double avg = std::accumulate(latencies.begin(), latencies.end(), 0.0)
                         / latencies.size();
            double p50 = percentile(latencies, 0.50);
            double p95 = percentile(latencies, 0.95);
            double bf  = brute_force_latency(queries, db, K);

            csv.write(M, efC, efS, K, mean_recall, avg, p50, p95, bf);
            if (mean_recall < 0.75f) {
                std::cout << "Skipping low-recall config M=" << M
                          << " efC=" << efC
                          << " efS=" << efS
                          << " K=" << K
                          << " Recall=" << mean_recall << std::endl;
            }
            /* ---- Sanity checks (non-flaky) ---- */
            if (K == 1)  EXPECT_GE(mean_recall, 0.60f);
            else EXPECT_GE(mean_recall, 0.60f);
        }
    }
}
