#include <benchmark/benchmark.h>
#include "minivec/hnsw.hpp"

static void BM_HNSW_Search(benchmark::State& state)
{
    const int dim = 128;
    const int N = 10'000;
    const int efSearch = state.range(0);

    HNSWIndexSimple index(dim, 32, 200, efSearch);

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0,1);

    std::vector<std::vector<float>> db(N, std::vector<float>(dim));
    for (auto& v : db)
        for (auto& x : v) x = dist(rng);

    for (int i = 0; i < N; ++i)
        index.insert_vector(db[i].data());

    std::vector<float> query(dim);
    for (auto& x : query) x = dist(rng);

    for (auto _ : state) {
        benchmark::DoNotOptimize(
            index.search_top_k(query.data(), efSearch, 10)
        );
    }
}

BENCHMARK(BM_HNSW_Search)
    ->Arg(50)
    ->Arg(100)
    ->Arg(200);

BENCHMARK_MAIN();
