# Reproducible recall and latency benchmark

`scripts/benchmark_ann.py` builds one deterministic index, sweeps `ef_search`
over the same index, and compares top-k results with exact neighbors. It reports
build time, peak process memory, recall@k, per-query p50/p95/p99 latency and
single-thread QPS. The JSON output records the source revision and software
environment; `--csv` writes one row per `ef_search` value.

## Real datasets (ann-benchmarks)

```bash
python -m pip install ".[bench]"          # adds h5py
python scripts/benchmark_ann.py --dataset sift-128-euclidean \
    --M 16 --ef-construction 200 --ef-search 16 32 64 128 256 \
    --output data/bench/sift1m.json --csv data/bench/sift1m.csv
```

Supported datasets are `sift-128-euclidean` and `glove-100-angular` (vectors
are L2-normalized so squared-L2 ranking equals cosine ranking). Files are
downloaded once from ann-benchmarks.com into `data/ann/` (about 500 MB each)
and recall uses the published ground truth. `--limit-queries N` uses the first
N queries; `--limit-train N` indexes the first N vectors and recomputes exact
ground truth for that subset.

### SIFT1M result

One run on 2026-10-10: Apple M5 (16 GB), macOS 26.6, Apple clang 21, Release
build, Python 3.14, NumPy 2.5. 1,000,000 base vectors, 10,000 queries, k=10,
`M=16`, `ef_construction=200`, deterministic levels (seed 42), single-threaded
build and queries. Raw data:
[2026-10-10-sift1m.json](benchmark-results/2026-10-10-sift1m.json).

| ef_search | Recall@10 | p50 (ms) | p95 (ms) | p99 (ms) | QPS (1 thread) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 0.773 | 0.048 | 0.070 | 0.082 | 20,471 |
| 32 | 0.868 | 0.077 | 0.105 | 0.121 | 12,992 |
| 64 | 0.932 | 0.135 | 0.177 | 0.215 | 7,457 |
| 128 | 0.966 | 0.239 | 0.305 | 0.329 | 4,343 |
| 256 | 0.983 | 0.416 | 0.536 | 0.578 | 2,483 |

Index construction took 525 s (about 1,900 inserts/s on one thread). Peak
process memory grew from 561 MiB (dataset loaded) to 1,314 MiB after the build.
Latency is the full Python-to-C++ call for one query. No comparison with other
libraries is implied by this table; compare only runs with the same hardware,
dataset and parameters.

## Synthetic data

Without `--dataset`, the script creates seeded Gaussian vectors and independent
queries (`--count`, `--dim`, `--queries`, `--seed`) and measures a NumPy
exact-search baseline alongside MiniVec:

```bash
python -m pip install .
python scripts/benchmark_ann.py --output benchmark-results/run.json
```

### Recorded synthetic example

The checked-in [JSON result](benchmark-results/2026-10-09-local.json) is one
local run on Apple silicon using 5,000 vectors of 128 dimensions and 100
queries. MiniVec used `M=32` and `ef_construction=200`; data and level
generation used seed 42.

| ef_search | Recall@10 | MiniVec p50 (ms/query) | MiniVec p95 (ms/query) |
| ---: | ---: | ---: | ---: |
| 50 | 0.958 | 0.047 | 0.051 |
| 100 | 0.985 | 0.074 | 0.078 |
| 200 | 0.997 | 0.114 | 0.121 |

Index construction took 3.422 seconds. The exact NumPy baseline in this run had 0.283 ms p50 and 0.322 ms p95 per
query. This is a small synthetic example, not a general performance claim:
results depend on the dataset, Python and NumPy versions, compiler, and machine.
The baseline also measures NumPy's vectorized CPU implementation, while MiniVec
reports the complete Python-to-C++ call for one query.
