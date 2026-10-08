# Reproducible recall and latency benchmark

Run the benchmark from a checkout after building MiniVec's Python extension:

```bash
python -m pip install .
python scripts/benchmark_ann.py --output benchmark-results/run.json
```

The script creates seeded Gaussian vectors and independent queries, builds a
deterministic index for each `ef_search` value, and compares top-k results with
exact squared-L2 neighbors. It reports build time, recall@k, and per-query p50,
p95, and mean latency for MiniVec and the NumPy exact-search baseline. The JSON
output records the source revision and software environment. Change the
dataset size, dimensions, query count, seed, or HNSW parameters with the
command-line options; compare runs only when those settings and the hardware
match.

## Recorded example

The checked-in [JSON result](benchmark-results/2026-10-08-local.json) is one
local run on Apple silicon using 5,000 vectors of 128 dimensions and 100
queries. MiniVec used `M=32` and `ef_construction=200`; data and level
generation used seed 42.

| ef_search | Recall@10 | MiniVec p50 (ms/query) | MiniVec p95 (ms/query) | Build (s) |
| ---: | ---: | ---: | ---: | ---: |
| 50 | 0.958 | 0.048 | 0.055 | 3.468 |
| 100 | 0.985 | 0.076 | 0.085 | 3.490 |
| 200 | 0.997 | 0.117 | 0.124 | 3.507 |

The exact NumPy baseline in this run had 0.233 ms p50 and 0.289 ms p95 per
query. This is a small synthetic example, not a general performance claim:
results depend on the dataset, Python and NumPy versions, compiler, and machine.
The baseline also measures NumPy's vectorized CPU implementation, while MiniVec
reports the complete Python-to-C++ call for one query.
