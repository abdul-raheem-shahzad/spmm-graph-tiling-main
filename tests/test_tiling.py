"""Sanity tests.  Run with `python tests/test_tiling.py` or `pytest`."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spmm_tiling import executor as E  # noqa: E402
from spmm_tiling import tiling as T  # noqa: E402


def toy() -> sp.csr_matrix:
    # rows 0 and 2 use columns {0, 1}; rows 1 and 3 use columns {2, 3}
    return sp.csr_matrix(np.array([
        [1, 1, 0, 0],
        [0, 0, 1, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
    ], dtype=float))


def random_matrix(n=300, m=200, density=0.03, seed=1) -> sp.csr_matrix:
    a = sp.random(n, m, density=density, format="csr", random_state=seed)
    a.sort_indices()
    return a


def test_toy_example_by_hand():
    a = toy()
    # contiguous tiles {0,1} and {2,3}: each needs 4 distinct B rows -> 8 fetches, 0% reuse
    assert E.tile_stats(a, T.contiguous(T.order_natural(a), 2)).reuse == 0.0
    # graph tiling pairs rows {0,2} and {1,3}: 2 + 2 fetches for 8 accesses -> 50% reuse
    tiles = T.noncontiguous_greedy(a, 2)
    assert sorted(sorted(t.tolist()) for t in tiles) == [[0, 2], [1, 3]]
    assert E.tile_stats(a, tiles).reuse == 0.5
    assert E.reuse_upper_bound(a, 2) == 0.5  # so the greedy result is optimal here


def test_tilings_are_partitions():
    a = random_matrix()
    for r in (1, 7, 32, 1000):
        for score in ("overlap", "ratio"):
            assert T.is_valid_tiling(T.noncontiguous_greedy(a, r, score), a.shape[0], r)
        assert T.is_valid_tiling(T.contiguous(T.order_random(a), r), a.shape[0], r)


def test_tiled_spmm_matches_reference():
    a = random_matrix()
    b = np.random.default_rng(0).standard_normal((a.shape[1], 8))
    for tiles in (T.contiguous(T.order_natural(a), 16), T.noncontiguous_greedy(a, 16)):
        c, stats = E.tiled_spmm(a, b, tiles)
        assert np.allclose(c, a @ b)
        assert stats.fetches == E.tile_stats(a, tiles).fetches


def test_reuse_never_exceeds_upper_bound():
    a = random_matrix(density=0.08)
    for r in (4, 16, 64):
        bound = E.reuse_upper_bound(a, r)
        for tiles in (T.contiguous(T.order_natural(a), r),
                      T.noncontiguous_greedy(a, r, "overlap"),
                      T.noncontiguous_greedy(a, r, "ratio")):
            assert E.tile_stats(a, tiles).reuse <= bound + 1e-12


def test_extreme_tile_sizes():
    a = random_matrix()
    # one row per tile: nothing can be reused
    assert E.tile_stats(a, T.noncontiguous_greedy(a, 1)).reuse == 0.0
    # one tile holding everything: each used column is fetched exactly once
    used_cols = len(np.unique(a.indices))
    s = E.tile_stats(a, T.noncontiguous_greedy(a, a.shape[0]))
    assert s.fetches == used_cols


def test_lru_with_huge_buffer_fetches_each_column_once():
    a = random_matrix()
    used_cols = len(np.unique(a.indices))
    reuse = E.lru_reuse(a, np.arange(a.shape[0]), capacity=10**9)
    assert abs(reuse - (1 - used_cols / a.nnz)) < 1e-12


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok  ", t.__name__)
    print(f"{len(tests)} tests passed")
