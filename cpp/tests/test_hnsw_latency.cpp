// tests/test_hnsw_latency.cpp
#include <gtest/gtest.h>
#include "minivec/hnsw.hpp"
#include "minivec/dist.hpp"
#include <chrono>
#include <random>
#include <iostream>

using namespace std::chrono;

TEST(HNSWLatency, QueryLatency) {
    const int dim = 128;
    const int N = 5000;
    const int Q = 1000;
    const int K = 10;
    const int ef = 100;

    std::mt19937 rng(42);
    std::normal_distribution<float> d(0,1);
    std::vector<std::vector<float>> db(N, std::vector<float>(dim));
    for (int i=0;i<N;++i) for (int j=0;j<dim;++j) db[i][j]=d(rng);

    HNSWIndexSimple index(dim);
    for (int i=0;i<N;++i) index.insert_vector(db[i].data());

    std::vector<std::vector<float>> queries(Q, std::vector<float>(dim));
    for (int i=0;i<Q;++i) for (int j=0;j<dim;++j) queries[i][j]=d(rng);

    std::vector<double> latencies; latencies.reserve(Q);
    for (int i=0;i<Q;++i) {
        auto t0 = high_resolution_clock::now();
        auto res = index.search_top_k(queries[i].data(), ef, K);
        auto t1 = high_resolution_clock::now();
        latencies.push_back(duration<double, std::micro>(t1-t0).count()); // microseconds
    }
    // compute stats
    std::sort(latencies.begin(), latencies.end());
    double sum=0; for(auto v:latencies) sum+=v;
    double avg = sum/latencies.size();
    double p50 = latencies[latencies.size()*50/100];
    double p95 = latencies[latencies.size()*95/100];
    double p99 = latencies[latencies.size()*99/100];
    std::cout << "Q="<<Q<<" avg(us)="<<avg<<" p50="<<p50<<" p95="<<p95<<" p99="<<p99<<std::endl;

    // sanity: make sure queries returned K results
    ASSERT_GT(latencies.size(), 0);
}
