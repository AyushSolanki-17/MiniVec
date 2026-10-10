#!/usr/bin/env python3
"""Download and load ann-benchmarks HDF5 datasets.

Each file holds ``train`` (indexed vectors), ``test`` (queries), ``neighbors``
(exact top-100 ids per query) and ``distances``. Files are cached locally and
never committed.
"""

import sys
import urllib.request
from pathlib import Path

import numpy as np

BASE_URL = "http://ann-benchmarks.com"
DATASETS = {
    "sift-128-euclidean": "l2",
    "glove-100-angular": "cosine",
}


def _download(url, dest):
    tmp = dest.with_suffix(dest.suffix + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    # The dataset host rejects urllib's default user agent.
    request = urllib.request.Request(url, headers={"User-Agent": "minivec-benchmark/1.0"})
    with urllib.request.urlopen(request) as response, open(tmp, "wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        while chunk := response.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"\rdownloading {url}: {done / total:6.1%}", end="", file=sys.stderr)
    print(file=sys.stderr)
    if tmp.stat().st_size == 0:
        tmp.unlink()
        raise RuntimeError(f"downloaded file is empty: {url}")
    tmp.rename(dest)


def load(name, cache_dir="data/ann"):
    """Return ``(train, test, neighbors, metric)`` for an ann-benchmarks dataset.

    Angular datasets are L2-normalized so that squared-L2 ranking matches
    cosine ranking.
    """
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; choose from {sorted(DATASETS)}")
    try:
        import h5py
    except ImportError as exc:
        raise SystemExit("h5py is required: pip install 'minivec-ann[bench]'") from exc

    path = Path(cache_dir) / f"{name}.hdf5"
    if not path.exists():
        _download(f"{BASE_URL}/{name}.hdf5", path)

    with h5py.File(path, "r") as f:
        train = np.asarray(f["train"], dtype=np.float32)
        test = np.asarray(f["test"], dtype=np.float32)
        neighbors = np.asarray(f["neighbors"], dtype=np.int64)

    metric = DATASETS[name]
    if metric == "cosine":
        train /= np.maximum(np.linalg.norm(train, axis=1, keepdims=True), 1e-12)
        test /= np.maximum(np.linalg.norm(test, axis=1, keepdims=True), 1e-12)
    return train, test, neighbors, metric
