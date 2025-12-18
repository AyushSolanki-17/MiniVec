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
import numpy as np
import minivec_cpp


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
        self.dim = dim
        self.M = M
        self._index = minivec_cpp.HNSWIndexSimple(
            dim,
            M,
            ef_construction,
            ef_search,
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
            Number of neighbors to retrieve.

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

        return self._index.search(q, k)

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

        results, stats = self._index.search_with_stats(q, k)
        return results, dict(stats)

    # ------------------------------------------------------------------
    # Uncertainty control (research feature)
    # ------------------------------------------------------------------

    def set_beta(self, beta: float) -> None:
        """
        Set uncertainty scaling parameter.

        beta = 0.0  → standard HNSW behavior
        beta > 0.0  → uncertainty-aware pruning/search
        """
        self._index.set_beta(float(beta))

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
