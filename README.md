# MiniVec

**A research-oriented, high-performance C++ vector search engine with Python bindings**

![C++](https://img.shields.io/badge/C++-17-blue.svg)
![Python](https://img.shields.io/badge/Python-3.11+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Build](https://img.shields.io/badge/Build-CMake-success.svg)

---

## 🔍 Overview

**MiniVec** is a **from-scratch implementation of the Hierarchical Navigable Small World (HNSW)** approximate nearest neighbor (ANN) algorithm, written in modern **C++17** and exposed to Python via **pybind11**.

Unlike projects that wrap existing ANN libraries, MiniVec is intentionally built to:

- expose the **internal mechanics of HNSW**
- emphasize **determinism and correctness**
- support **research experimentation**
- demonstrate **systems-level engineering practices**

The goal of this project is not to replace FAISS or hnswlib, but to provide a **clear, inspectable, and extensible reference implementation** of ANN search.

---

## 🚀 Quick Start

Clone the repository:

```bash
git clone https://github.com/<your-username>/minivec
cd minivec
````

---

## ⚙️ Installation

### Requirements

* C++17 compiler (GCC / Clang)
* CMake ≥ 3.16
* Python ≥ 3.11 (optional for bindings)
* pybind11

---

### Build C++ Library

```bash
mkdir build
cd build
cmake ..
make -j
```

---

### Install Python Bindings

```bash
pip install -e .
```

This exposes the MiniVec index to Python.

---

## 📐 Architecture

![MiniVec Architecture](./docs/images/HNSW.png)

High-level workflow:

### Index Construction

```
Vector Insertion
   → Layer Assignment
   → Greedy Descent
   → efConstruction Candidate Search
   → Neighbor Selection & Pruning
   → Hierarchical Graph Update
```

### Query Search

```
Query Vector
   → Greedy Descent (top layers)
   → efSearch Exploration (bottom layer)
   → Candidate Heap
   → Top-K Nearest Neighbors
```

The algorithm follows the **standard HNSW design**, maintaining a **multi-layer proximity graph** that enables logarithmic search complexity.

---

## ✨ Key Features

### Core Functionality

* ⚡ **Approximate Nearest Neighbor Search**
* 🧠 **Full HNSW implementation from first principles**
* 🔧 Configurable parameters:

  * `M` — maximum neighbors per node
  * `efConstruction` — graph build search width
  * `efSearch` — query search width

---

### Systems Engineering Focus

* 📊 **Search instrumentation**

  * visited nodes
  * distance computations
  * layer traversal statistics

* 🧵 **Thread-safe graph updates**

  * fine-grained locking
  * deadlock-safe neighbor linking

* 📦 **CMake-based build system**

* 🐍 **Zero-copy Python bindings via pybind11**

---

### Design Philosophy

MiniVec intentionally avoids hidden optimizations.

Goals:

* transparency
* reproducibility
* inspectability
* algorithmic correctness

Everything is measurable and observable.

---

## 🔁 Deterministic Graph Construction

MiniVec supports **deterministic index builds**.

When enabled:

* layer assignment is seeded
* insertion order is preserved
* graph structure becomes reproducible
* search results are deterministic

This is useful for:

* benchmarking
* regression testing
* academic experimentation
* debugging ANN behavior

---

## 📊 Search Instrumentation

MiniVec exposes runtime search metrics:

```cpp
struct SearchStats {
    uint64_t visited_nodes;
    uint64_t distance_calls;
    std::unordered_map<int, uint64_t> layer_visits;
};
```

These statistics enable:

* recall vs latency analysis
* algorithm debugging
* performance comparisons against FAISS / hnswlib
* tuning search parameters

Instrumentation is available in **C++** and can be exposed through **Python bindings**.

---

## 📈 Benchmarking Results

MiniVec was evaluated on synthetic and large-scale datasets.

Configuration:

| Parameter      | Value            |
| -------------- | ---------------- |
| Dimensionality | 128              |
| Distance       | L2               |
| Dataset sizes  | 10K / 1M vectors |

Hardware: modern x86 CPU

---

### Small Dataset (10K vectors)

| Configuration      | Recall@10 | P50 Latency |
| ------------------ | --------- | ----------- |
| M=32 efSearch=100  | ≈ 0.90    | ≈ 0.18 ms   |
| Brute Force Search | 1.00      | ≈ 0.36 ms   |

---

### Large Dataset (1M vectors)

| Configuration              | Recall@10 | P50 Latency |
| -------------------------- | --------- | ----------- |
| HNSW (M=32 tuned efSearch) | 0.89      | 11.7 ms     |
| Brute Force                | 1.00      | ~30 ms      |

Performance scales **logarithmically with dataset size**, consistent with theoretical HNSW expectations.

Results are comparable to default FAISS HNSW configurations.

---

## 🐍 Python Usage

Example usage from Python:

```python
import minivec

index = minivec.HNSWIndex(
    dim=128,
    M=32,
    ef_construction=200,
    ef_search=100
)

index.add(vectors)

results, stats = index.search(query, k=10, return_stats=True)
```

---

## 📂 Project Structure

```
minivec/
├── include/        # HNSW graph data structures
├── src/            # core algorithm implementation
├── python/         # pybind11 bindings
├── benchmarks/     # performance experiments
├── tests/          # deterministic & correctness tests
├── examples/       # minimal usage examples
└── docs/           # algorithm documentation
```

---

## 🧠 Why This Project Exists

MiniVec is not intended to compete with FAISS or hnswlib.

Instead it exists to:

* deeply understand ANN algorithms
* explore algorithm–systems tradeoffs
* demonstrate performance-oriented C++ design
* provide a transparent reference implementation

It is designed to be **read, studied, modified, and extended**.

---

## 🔮 Future Work

Planned improvements:

* adaptive `efSearch`
* memory-mapped indices
* deletion and update support
* SIMD distance optimizations
* GPU search backend
* research experiments on deterministic vs stochastic graph builds

---

## 📜 License

MIT License

---

## 🙌 Acknowledgements

This project is inspired by the original **HNSW paper**:

> *Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs*
> Yu. A. Malkov, D. A. Yashunin (2018)

and by open-source implementations such as:

* FAISS
* hnswlib

MiniVec aims to provide a **minimal, transparent implementation for learning and experimentation**.

