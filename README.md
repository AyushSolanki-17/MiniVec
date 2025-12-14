# MiniVec
A mini version of FAISS

## Custom Benchmarking Results
We benchmarked HNSW over M ∈ {8,16,32,64}.
We observed that M < 16 leads to sharp recall degradation, validating FAISS defaults.
At M=32, efSearch=100 we achieve Recall@10 ≈ 0.90 with ~0.18ms P50 latency,
which is ~2× faster than brute force at N=10k.
Increasing efSearch improves recall but introduces tail latency (P95),
demonstrating the classic recall–latency tradeoff.
