# minivec/core.py
"""
Python-facing wrapper for the MiniVec HNSW index.

This module provides a thin, explicit wrapper over the C++ implementation
exposed via pybind11. The goal is:
- zero hidden magic
- explicit data conversions
- predictable performance characteristics
"""

from typing import List, Tuple, Dict
from operator import index as integer_index
import numpy as np
import minivec_cpp

_C_INT_MAX = 2**31 - 1


def _non_negative_k(k: int) -> int:
    """Return k as an integer, rejecting bools and lossy conversions."""
    if isinstance(k, (bool, np.bool_)):
        raise TypeError("k must be a non-negative integer")
    try:
        value = integer_index(k)
    except TypeError as exc:
        raise TypeError("k must be a non-negative integer") from exc
    if value < 0:
        raise ValueError("k must be non-negative")
    if value > _C_INT_MAX:
        raise ValueError(f"k must be at most {_C_INT_MAX}")
    return value


class MiniVecIndex:
    """
    MiniVecIndex is a lightweight Python wrapper over the C++ HNSWIndexSimple.

    It owns:
    - dimensionality validation
    - numpy ↔ C++ boundary handling
    - a minimal Pythonic API

    It does NOT:
    - reimplement search logic
    - store vectors redundantly
    - hide algorithmic behavior
    """

    def __init__(
        self,
        dim: int,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 200,
        distance: str = "l2_squared",
        final_distance: str = "l2",
    ):
        """
        Create a new HNSW index.

        Parameters
        ----------
        dim : int
            Dimensionality of vectors.
        M : int
            Maximum number of neighbors per node (graph degree).
        ef_construction : int
            Candidate list size during index construction.
        ef_search : int
            Candidate list size during search.
        distance : str
            Distance function used internally (e.g. "l2_squared").
        final_distance : str
            Distance used for final re-ranking.
        """
        if isinstance(dim, (bool, np.bool_)) or not isinstance(dim, (int, np.integer)):
            raise TypeError("dim must be a positive integer")
        if dim <= 0:
            raise ValueError("dim must be a positive integer")
        if isinstance(M, (bool, np.bool_)) or not isinstance(M, (int, np.integer)):
            raise TypeError("M must be a positive integer")
        if M <= 0:
            raise ValueError("M must be a positive integer")
        if M > _C_INT_MAX // 2:
            raise ValueError(f"M must be at most {_C_INT_MAX // 2}")
        for name, value in (("ef_construction", ef_construction), ("ef_search", ef_search)):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise TypeError(f"{name} must be a positive integer")
            if value <= 0:
                raise ValueError(f"{name} must be a positive integer")
            if value > _C_INT_MAX:
                raise ValueError(f"{name} must be at most {_C_INT_MAX}")
        if dim > _C_INT_MAX:
            raise ValueError(f"dim must be at most {_C_INT_MAX}")
        self.dim = int(dim)
        self.M = int(M)
        self._index = minivec_cpp.HNSWIndexSimple(
            self.dim,
            self.M,
            int(ef_construction),
            int(ef_search),
            False,          # deterministic_levelgen
            42,             # seed
            distance,
            final_distance,
        )

    # ------------------------------------------------------------------
    # Insertion
    # ------------------------------------------------------------------

    def add(self, vector: np.ndarray) -> int:
        """
        Insert a vector into the index.

        Parameters
        ----------
        vector : np.ndarray
            1D float32 array of shape (dim,).

        Returns
        -------
        int
            Internal ID assigned by the index.
        """
        vec = np.asarray(vector, dtype=np.float32)

        if vec.ndim != 1 or vec.shape[0] != self.dim:
            raise ValueError(
                f"Expected vector of shape ({self.dim},), got {vec.shape}"
            )
        if not np.isfinite(vec).all():
            raise ValueError("Vector values must all be finite")

        return int(self._index.insert_vector(vec))

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: np.ndarray, k: int) -> List[Tuple[int, float]]:
        """
        Search for the top-k nearest neighbors.

        Parameters
        ----------
        query : np.ndarray
            Query vector of shape (dim,).
        k : int
            Number of neighbors to retrieve. Must be a non-negative integer.
            Zero returns an empty list; values larger than the index size
            return all available neighbors.

        Returns
        -------
        List[(int, float)]
            List of (id, distance) pairs sorted by distance.
        """
        q = np.asarray(query, dtype=np.float32)

        if q.ndim != 1 or q.shape[0] != self.dim:
            raise ValueError(
                f"Expected query of shape ({self.dim},), got {q.shape}"
            )

        if not np.isfinite(q).all():
            raise ValueError("Query values must all be finite")
        return self._index.search(q, _non_negative_k(k))

    def search_with_stats(
        self, query: np.ndarray, k: int
    ) -> Tuple[List[Tuple[int, float]], Dict[str, int]]:
        """
        Search with instrumentation enabled.

        Returns both neighbors and internal search statistics.

        Useful for:
        - benchmarking
        - research experiments
        - debugging traversal behavior
        """
        q = np.asarray(query, dtype=np.float32)

        if q.ndim != 1 or q.shape[0] != self.dim:
            raise ValueError(
                f"Expected query of shape ({self.dim},), got {q.shape}"
            )
        if not np.isfinite(q).all():
            raise ValueError("Query values must all be finite")

        results, stats = self._index.search_with_stats(q, _non_negative_k(k))
        return results, dict(stats)

    def add_many(self, vectors: np.ndarray) -> List[int]:
        """Insert finite vectors of shape (n, dim) and return their IDs."""
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != self.dim:
            raise ValueError(
                f"Expected vectors with shape (n, {self.dim}), got {matrix.shape}"
            )
        if not np.isfinite(matrix).all():
            raise ValueError("Vector values must all be finite")
        return list(self._index.insert_vectors(matrix))

    def search_many(
        self, queries: np.ndarray, k: int
    ) -> List[List[Tuple[int, float]]]:
        """Search finite queries of shape (n, dim), returning one top-k list per row."""
        matrix = np.asarray(queries, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != self.dim:
            raise ValueError(
                f"Expected queries with shape (n, {self.dim}), got {matrix.shape}"
            )
        if not np.isfinite(matrix).all():
            raise ValueError("Query values must all be finite")
        return self._index.search_batch(matrix, _non_negative_k(k))

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Number of vectors currently stored."""
        return self._index.node_count()

    @property
    def entry_point(self) -> int:
        """Current entry point of the HNSW graph."""
        return self._index.entry_point()

    @property
    def max_level(self) -> int:
        """Maximum level present in the index."""
        return self._index.max_level()
