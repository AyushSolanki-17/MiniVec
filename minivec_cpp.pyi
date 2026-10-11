# stubs/minivec_cpp.pyi
from typing import Dict, List, Tuple
import numpy as np


# ---------------------------------------------------------
# Distance helpers
# ---------------------------------------------------------

def l2(a: np.ndarray, b: np.ndarray) -> np.float32: ...
def l2_squared(a: np.ndarray, b: np.ndarray) -> np.float32: ...


# ---------------------------------------------------------
# Search statistics (for research / benchmarking)
# ---------------------------------------------------------

class SearchStats:
    visited_nodes: int
    distance_calls: int
    greedy_hops: int
    hops_per_layer: Dict[int, int]
    layer_visits: Dict[int, int]
    candidate_queue_stability: float
    result_distance_gaps: List[float]
    search_strategy: str
    filter_rejected_candidates: int

    def __init__(self) -> None: ...


# ---------------------------------------------------------
# Core HNSW index
# ---------------------------------------------------------

class HNSWIndexSimple:
    """
    Low-level C++ HNSW index exposed via pybind11.

    This class mirrors the C++ API closely and is not intended
    to be user-facing directly. Use MiniVecIndex instead.
    """

    # -------- constructor --------
    def __init__(
        self,
        dim: int,
        M: int = 16,
        efConstruction: int = 200,
        efSearch: int = 200,
        deterministic_levelgen: bool = False,
        levelgen_seed: int = 42,
        distance: str = "l2_squared",
        final_distance: str = "l2",
    ) -> None: ...

    # -------- insertion --------
    def insert_vector(self, vector: np.ndarray, attrs: Dict[str, int] = ...) -> int: ...
    def insert_vectors(
        self, vectors: np.ndarray, attrs: List[Dict[str, int]] = ...
    ) -> List[int]: ...
    def set_attribute(self, id: int, name: str, value: int) -> None: ...
    def attribute_eq(self, name: str, value: int) -> Tuple[List[int], int]: ...
    def attribute_in(self, name: str, values: List[int]) -> Tuple[List[int], int]: ...
    def attribute_range(
        self, name: str, minimum: int, maximum: int
    ) -> Tuple[List[int], int]: ...

    # -------- search --------
    def search(
        self,
        query: np.ndarray,
        k: int,
    ) -> List[Tuple[int, float]]: ...

    def search_with_stats(
        self,
        query: np.ndarray,
        k: int,
    ) -> Tuple[List[Tuple[int, float]], Dict[str, object]]: ...

    def search_filtered(
        self,
        query: np.ndarray,
        k: int,
        bitmap: List[int],
        strategy: str,
        ef: int = 0,
        expansion_factor: int = 4,
    ) -> Tuple[List[Tuple[int, float]], Dict[str, object]]: ...

    def search_batch(
        self, queries: np.ndarray, k: int
    ) -> List[List[Tuple[int, float]]]: ...

    # -------- getters --------
    def node_count(self) -> int: ...
    def entry_point(self) -> int: ...
    def max_level(self) -> int: ...
    def vector_dim(self) -> int: ...
    def M(self) -> int: ...
    def efConstruction(self) -> int: ...
    def efSearch(self) -> int: ...
    def set_ef_search(self, ef_search: int) -> None: ...

    # -------- vector access --------
    def get_vector(self, id: int) -> np.ndarray: ...
