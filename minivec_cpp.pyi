# stubs/minivec_cpp.pyi
from typing import List, Tuple, Dict
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
    layer_visits: Dict[int, int]

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
    def insert_vector(self, vector: np.ndarray) -> int: ...

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
    ) -> Tuple[List[Tuple[int, float]], Dict[str, int]]: ...

    # -------- uncertainty control --------
    def set_beta(self, beta: float) -> None: ...

    # -------- getters --------
    def node_count(self) -> int: ...
    def entry_point(self) -> int: ...
    def max_level(self) -> int: ...
    def vector_dim(self) -> int: ...
    def M(self) -> int: ...
    def efConstruction(self) -> int: ...
    def efSearch(self) -> int: ...

    # -------- vector access --------
    def get_vector(self, id: int) -> np.ndarray: ...
