"""Tile-by-tile SpMM executor and data-reuse metrics.

The executor really computes C = A @ B one tile at a time and counts how many
rows of B it had to fetch, so the reuse number is measured on the actual
computation rather than estimated.

Definitions (row-wise / Gustavson dataflow):
  access : one non-zero A[i, j] needs row B[j, :] once          -> nnz(A) accesses
  fetch  : loading row B[j, :] from off-chip memory into the tile buffer
  reuse  : an access served by a row already in the buffer

      data reuse = 1 - fetches / accesses
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from .tiling import Tiling


@dataclass
class TileStats:
    accesses: int  # nnz(A)
    fetches: int  # rows of B loaded, summed over tiles
    max_buffer_rows: int  # largest tile buffer (rows of B)
    mean_buffer_rows: float

    @property
    def reuse(self) -> float:
        return 1.0 - self.fetches / self.accesses


def tile_stats(a: sp.csr_matrix, tiles: Tiling) -> TileStats:
    """Count fetches for a tiling without doing the arithmetic (fast)."""
    sizes = np.empty(len(tiles), dtype=np.int64)
    for t, rows in enumerate(tiles):
        sizes[t] = len(np.unique(a[rows].indices))
    return TileStats(int(a.nnz), int(sizes.sum()), int(sizes.max()), float(sizes.mean()))


def tiled_spmm(a: sp.csr_matrix, b: np.ndarray, tiles: Tiling) -> tuple[np.ndarray, TileStats]:
    """Compute C = A @ B tile by tile, counting B-row fetches as it goes."""
    c = np.zeros((a.shape[0], b.shape[1]), dtype=b.dtype)
    fetches, sizes = 0, []
    for rows in tiles:
        sub = a[rows]  # the tile's rows of A
        cols = np.unique(sub.indices)  # distinct B rows this tile needs
        buffer = b[cols]  # <-- the fetch: |cols| rows of B
        fetches += len(cols)
        sizes.append(len(cols))
        c[rows] = sub[:, cols] @ buffer  # every access is served from buffer
    stats = TileStats(int(a.nnz), fetches, max(sizes), float(np.mean(sizes)))
    return c, stats


def reuse_upper_bound(a: sp.csr_matrix, tile_rows: int) -> float:
    """Best reuse any tiling with <= tile_rows rows per tile can reach.

    A column with d non-zeros is spread over at least ceil(d / R) tiles, and
    must be fetched once in each of them.
    """
    col_deg = np.bincount(a.indices, minlength=a.shape[1])
    min_fetches = np.ceil(col_deg / tile_rows).sum()
    return 1.0 - min_fetches / a.nnz


def lru_reuse(a: sp.csr_matrix, row_order: np.ndarray, capacity: int) -> float:
    """Reuse with a fixed-size B-row buffer (LRU replacement).

    Cross-check for the tile model: rows of A are processed in `row_order`
    and the buffer holds at most `capacity` rows of B, whatever the tile
    shape.  Returns hits / accesses.
    """
    cache: OrderedDict[int, None] = OrderedDict()
    hits = 0
    indptr, indices = a.indptr, a.indices
    for r in row_order:
        for j in indices[indptr[r] : indptr[r + 1]].tolist():
            if j in cache:
                hits += 1
                cache.move_to_end(j)
            else:
                cache[j] = None
                if len(cache) > capacity:
                    cache.popitem(last=False)
    return hits / a.nnz
