# binturong/core.py
"""
Python-facing wrapper for the Binturong HNSW index.

This module provides a thin, explicit wrapper over the C++ implementation
exposed via pybind11. The goal is:
- zero hidden magic
- explicit data conversions
- predictable performance characteristics
"""

from __future__ import annotations

from typing import List, Tuple, Dict
from collections.abc import Mapping, Sequence
from operator import index as integer_index
import numpy as np
import binturong_cpp

_C_INT_MAX = 2**31 - 1
_I64_MIN = -(2**63)
_I64_MAX = 2**63 - 1


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


def _attribute_value(value: int) -> int:
    """Normalize an integer attribute while rejecting bools and overflow."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("attribute values must be integers, not bools")
    try:
        normalized = integer_index(value)
    except TypeError as exc:
        raise TypeError("attribute values must be integers") from exc
    if normalized < _I64_MIN or normalized > _I64_MAX:
        raise ValueError("attribute values must fit in signed 64-bit integers")
    return normalized


def _attributes(attrs: Mapping[str, int] | None) -> Dict[str, int]:
    """Validate one vector's optional integer attributes."""
    if attrs is None:
        return {}
    if not isinstance(attrs, Mapping):
        raise TypeError("attrs must be a mapping from non-empty names to integers")
    normalized = {}
    for name, value in attrs.items():
        if not isinstance(name, str) or not name:
            raise ValueError("attribute names must be non-empty strings")
        normalized[name] = _attribute_value(value)
    return normalized


class BinturongIndex:
    """
    BinturongIndex is a lightweight Python wrapper over the C++ HNSWIndexSimple.

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
        deterministic: bool = False,
        seed: int = 42,
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
        deterministic : bool
            Use seeded serial level generation for repeatable graph builds.
        seed : int
            Seed used when ``deterministic`` is true.
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
        self._index = binturong_cpp.HNSWIndexSimple(
            self.dim,
            self.M,
            int(ef_construction),
            int(ef_search),
            deterministic,
            seed,
            distance,
            final_distance,
        )

    # ------------------------------------------------------------------
    # Insertion
    # ------------------------------------------------------------------

    def add(self, vector: np.ndarray, attrs: Mapping[str, int] | None = None) -> int:
        """
        Insert a vector into the index.

        Parameters
        ----------
        vector : np.ndarray
            1D float32 array of shape (dim,).
        attrs : Mapping[str, int], optional
            Integer attributes to store alongside this vector.

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

        return int(self._index.insert_vector(vec, _attributes(attrs)))

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
    ) -> Tuple[List[Tuple[int, float]], Dict[str, object]]:
        """
        Search with instrumentation enabled.

        Returns both neighbors and internal search statistics.

        Useful for:
        - benchmarking
        - research experiments
        - debugging traversal behavior

        Stats include distance calls, greedy hops by layer, candidate-queue
        stability, and adjacent gaps between sorted result distances.
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

    def search_filtered(
        self,
        query: np.ndarray,
        k: int,
        bitmap: Tuple[List[int], int],
        strategy: str = "filtered_traversal",
        expansion_factor: int = 4,
    ) -> Tuple[List[Tuple[int, float]], Dict[str, object]]:
        """Search a predicate bitmap with brute force, filtered traversal, or post-filtering.

        ``bitmap`` is the ``(words, count)`` tuple returned by an attribute predicate.
        ``post_filter`` searches with ``ef_search * expansion_factor`` before filtering.
        """
        q = np.asarray(query, dtype=np.float32)
        if q.ndim != 1 or q.shape[0] != self.dim:
            raise ValueError(f"Expected query of shape ({self.dim},), got {q.shape}")
        if not np.isfinite(q).all():
            raise ValueError("Query values must all be finite")
        if strategy not in {"brute_force", "filtered_traversal", "post_filter"}:
            raise ValueError("strategy must be brute_force, filtered_traversal, or post_filter")
        if isinstance(expansion_factor, (bool, np.bool_)) or not isinstance(
            expansion_factor, (int, np.integer)
        ):
            raise TypeError("expansion_factor must be a positive integer")
        if expansion_factor <= 0 or expansion_factor > _C_INT_MAX:
            raise ValueError("expansion_factor must be a positive integer")
        if (not isinstance(bitmap, tuple) or len(bitmap) != 2 or
                not isinstance(bitmap[0], Sequence)):
            raise TypeError("bitmap must be the (words, count) result of an attribute predicate")
        words = []
        for word in bitmap[0]:
            if isinstance(word, (bool, np.bool_)):
                raise TypeError("bitmap words must be unsigned 64-bit integers")
            try:
                normalized = integer_index(word)
            except TypeError as exc:
                raise TypeError("bitmap words must be unsigned 64-bit integers") from exc
            if normalized < 0 or normalized >= 2**64:
                raise ValueError("bitmap words must fit in unsigned 64-bit integers")
            words.append(normalized)
        results, stats = self._index.search_filtered(
            q,
            _non_negative_k(k),
            words,
            strategy,
            0,
            int(expansion_factor),
        )
        return results, dict(stats)

    def add_many(
        self,
        vectors: np.ndarray,
        attrs: Sequence[Mapping[str, int] | None] | None = None,
    ) -> List[int]:
        """Insert finite vectors of shape (n, dim), with optional per-row attributes."""
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != self.dim:
            raise ValueError(
                f"Expected vectors with shape (n, {self.dim}), got {matrix.shape}"
            )
        if not np.isfinite(matrix).all():
            raise ValueError("Vector values must all be finite")
        if attrs is None:
            normalized_attrs = [{} for _ in range(len(matrix))]
        else:
            if isinstance(attrs, (str, bytes)) or not isinstance(attrs, Sequence):
                raise TypeError("attrs must be a sequence with one mapping per vector")
            if len(attrs) != len(matrix):
                raise ValueError("attrs must contain one mapping per vector")
            normalized_attrs = [_attributes(row) for row in attrs]
        return list(self._index.insert_vectors(matrix, normalized_attrs))

    def set_attribute(self, vector_id: int, name: str, value: int) -> None:
        """Set or replace one signed 64-bit integer attribute on a vector."""
        if isinstance(name, bool) or not isinstance(name, str) or not name:
            raise ValueError("attribute name must be a non-empty string")
        self._index.set_attribute(vector_id, name, _attribute_value(value))

    def attribute_eq(self, name: str, value: int) -> Tuple[List[int], int]:
        """Return (bitmap words, match count) for an attribute equality predicate."""
        return self._index.attribute_eq(name, _attribute_value(value))

    def attribute_in(self, name: str, values: Sequence[int]) -> Tuple[List[int], int]:
        """Return (bitmap words, match count) for an attribute membership predicate."""
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise TypeError("values must be a sequence of integers")
        normalized = [_attribute_value(value) for value in values]
        return self._index.attribute_in(name, normalized)

    def attribute_range(
        self, name: str, minimum: int, maximum: int
    ) -> Tuple[List[int], int]:
        """Return an inclusive integer-range bitmap and match count."""
        return self._index.attribute_range(
            name, _attribute_value(minimum), _attribute_value(maximum)
        )

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

    def set_ef_search(self, ef_search: int) -> None:
        """Set the default search breadth used by subsequent queries."""
        if isinstance(ef_search, (bool, np.bool_)) or not isinstance(
            ef_search, (int, np.integer)
        ):
            raise TypeError("ef_search must be a positive integer")
        if ef_search <= 0:
            raise ValueError("ef_search must be a positive integer")
        if ef_search > _C_INT_MAX:
            raise ValueError(f"ef_search must be at most {_C_INT_MAX}")
        self._index.set_ef_search(int(ef_search))

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
