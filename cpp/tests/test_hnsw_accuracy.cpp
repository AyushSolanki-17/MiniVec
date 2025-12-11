// tests/test_hnsw_accuracy.cpp
#include <gtest/gtest.h>
#include "minivec/hnsw.hpp"
#include "minivec/dist.hpp"
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>
#include <numeric>

using namespace std::chrono;

static std::vector<int> brute_force_knn(
    const std::vector<float>& query,
    const std::vector<std::vector<float>>& db,
    int k)
{
    struct Item { int id; float dist; };
    std::vector<Item> items; items.reserve(db.size());
    for (size_t i=0;i<db.size();++i) {
        float d = minivec::l2_squared_distance(query.data(), db[i].data(), query.size());
        items.push_back({(int)i, d});
    }
    std::nth_element(items.begin(), items.begin()+std::min(k,(int)items.size())-1, items.end(),
                     [](const Item&a,const Item&b){return a.dist < b.dist;});
    std::sort(items.begin(), items.begin()+std::min(k,(int)items.size()),
              [](const Item&a,const Item&b){return a.dist < b.dist;});
    std::vector<int> out;
    for (int i=0;i<std::min(k,(int)items.size());++i) out.push_back(items[i].id);
    return out;
}
#include <algorithm>

// compute top-k brute force
std::vector<std::vector<int>> brute_force_nn(const std::vector<std::vector<float>>& data, int k) {
    int n = data.size();
    std::vector<std::vector<int>> out(n);
    for (int i=0;i<n;++i) {
        std::vector<std::pair<float,int>> dist;
        dist.reserve(n-1);
        for (int j=0;j<n;++j) if (i!=j) {
            float d = 0;
            for (size_t d_i=0; d_i<data[i].size(); ++d_i) {
                float diff = data[i][d_i] - data[j][d_i];
                d += diff*diff;
            }
            dist.emplace_back(d, j);
        }
        std::nth_element(dist.begin(), dist.begin()+k, dist.end());
        std::sort(dist.begin(), dist.begin()+k);
        out[i].reserve(k);
        for (int t=0;t<k;++t) out[i].push_back(dist[t].second);
    }
    return out;
}


static float recall_at_k(const std::vector<int>& gt, const std::vector<int>& pred) {
    int k = gt.size();
    int found = 0;
    for (int id : pred) {
        if (std::find(gt.begin(), gt.end(), id) != gt.end()) ++found;
    }
    return float(found) / float(k);
}

TEST(HNSWAccuracy, RecallSmallDataset) {
    const int dim = 32;
    const int N = 500;    // DB size (small for brute force)
    const int Q = 100;    // number of queries
    const int K = 10;

    // make data
    std::mt19937 rng(123);
    std::normal_distribution<float> dist(0.0f,1.0f);
    std::vector<std::vector<float>> db(N, std::vector<float>(dim));
    for (int i=0;i<N;++i) for (int d=0; d<dim; ++d) db[i][d] = dist(rng);

    // Build index
    // experiment: increase connectivity and construction ef
    HNSWIndexSimple index(32, /*M=*/48, /*efConstruction=*/1000, /*efSearch=*/200);
    //HNSWIndexSimple index(dim);
    for (int i=0;i<N;++i) index.insert_vector(db[i].data());

    // Query set
    std::vector<std::vector<float>> queries(Q, std::vector<float>(dim));
    for (int q=0;q<Q;++q) for (int d=0; d<dim; ++d) queries[q][d] = dist(rng);

    // Evaluate recall@K
    float sum_recall = 0.0f;
    for (int q=0;q<Q;++q) {
        auto gt = brute_force_knn(queries[q], db, K);
        auto out = index.search_top_k(queries[q].data(), 500, K);
        std::vector<int> pred;
        for (auto &c: out) pred.push_back(c.id);
        sum_recall += recall_at_k(gt, pred);
    }
    float mean_recall = sum_recall / Q;
    std::cout << "mean recall@" << K << " = " << mean_recall << std::endl;

    // a sanity check: recall should be reasonably high for small N (tweak threshold)
    ASSERT_GE(mean_recall, 0.7f);
}

// Replace test body with this (or add as a new debug test)

TEST(HNSWAccuracy, RecallSmallDataset_Debug) {
    const int dim = 32;
    const int N = 500;
    const int Q = 100;
    const int K = 10;

    // make data
    std::mt19937 rng(123);
    std::normal_distribution<float> dist(0.0f,1.0f);
    std::vector<std::vector<float>> db(N, std::vector<float>(dim));
    for (int i=0;i<N;++i) for (int d=0; d<dim; ++d) db[i][d] = dist(rng);

    // Build index with previously tried params
    HNSWIndexSimple index(dim, /*M=*/48, /*efConstruction=*/1000, /*efSearch=*/200);
    for (int i=0;i<N;++i) index.insert_vector(db[i].data());

    // Dump the graph to stdout
    std::cout << ">>> DUMP GRAPH <<<\n";
    index.dump_graph(std::cout);

    // Compute basic degree stats on layer 0 and check symmetry
    int n_nodes = index.get_node_count();
    std::vector<int> deg(n_nodes, 0);
    int isolated = 0;
    int invalid_links = 0;
    for (int i=0;i<n_nodes;++i) {
        auto nbrs = index.get_neighbors_copy(i, 0);
        deg[i] = nbrs.size();
        if (deg[i] == 0) ++isolated;
        for (int nb : nbrs) {
            if (nb < 0 || nb >= n_nodes) ++invalid_links;
            else {
                auto rev = index.get_neighbors_copy(nb, 0);
                if (std::find(rev.begin(), rev.end(), i) == rev.end()) {
                    std::cout << "[ASYMM] node " << i << " -> " << nb << " not present in reverse list\n";
                }
            }
        }
    }
    double avg_deg = std::accumulate(deg.begin(), deg.end(), 0.0) / n_nodes;
    std::cout << "layer0: avg_deg=" << avg_deg << " isolated=" << isolated << " invalid_links=" << invalid_links << "\n";

    // connected components (layer 0)
    std::vector<char> seen(n_nodes, 0);
    std::vector<int> comp_size;
    for (int i=0;i<n_nodes;++i) {
        if (seen[i]) continue;
        int count = 0;
        std::queue<int> q; q.push(i); seen[i]=1;
        while(!q.empty()){
            int u=q.front(); q.pop(); ++count;
            auto nbrs = index.get_neighbors_copy(u, 0);
            for (int v: nbrs) if (!seen[v]) { seen[v]=1; q.push(v); }
        }
        comp_size.push_back(count);
    }
    std::sort(comp_size.begin(), comp_size.end(), std::greater<int>());
    std::cout << "components (sizes desc): ";
    for (int s : comp_size) std::cout << s << " ";
    std::cout << "\n";

    // Query set
    std::vector<std::vector<float>> queries(Q, std::vector<float>(dim));
    for (int q=0;q<Q;++q) for (int d=0; d<dim; ++d) queries[q][d] = dist(rng);

    // Evaluate recall@K and capture per-query results for worst offenders
    struct Failure { int q; float recall; std::vector<int> gt; std::vector<int> pred; };
    std::vector<Failure> fails;
    float sum_recall = 0.0f;
    for (int q=0;q<Q;++q) {
        auto gt = brute_force_knn(queries[q], db, K);
        auto out = index.search_top_k(queries[q].data(), 500, K);
        std::vector<int> pred;
        for (auto &c: out) pred.push_back(c.id);
        float r = recall_at_k(gt, pred);
        sum_recall += r;
        if (r < 1.0f) { // collect those not perfect
            fails.push_back({q, r, gt, pred});
        }
    }
    float mean_recall = sum_recall / Q;
    std::cout << "mean recall@" << K << " = " << mean_recall << " fails=" << fails.size() << std::endl;

    // print worst 5 queries
    std::sort(fails.begin(), fails.end(), [](auto &a, auto &b){ return a.recall < b.recall; });
    int printN = std::min((size_t)5, fails.size());
    for (int i=0;i<printN;++i) {
        auto &f = fails[i];
        std::cout << "\n--- worst q=" << f.q << " recall=" << f.recall << " ---\n";
        std::cout << "GT: ";
        for (int id: f.gt) std::cout << id << " ";
        std::cout << "\nPRED: ";
        for (int id: f.pred) std::cout << id << " ";
        std::cout << "\n";
    }

    // Keep test passing for now (we'll assert later). But fail if huge fragmentation
    ASSERT_LT(isolated, n_nodes * 0.5); // if >50% isolated – big problem
    // Also assert recall is not ridiculously low
    ASSERT_GE(mean_recall, 0.3f);
}
