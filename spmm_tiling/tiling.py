"""Row-tiling strategies for SpMM  C = A @ B  (A sparse, B dense).

A *tiling* is a partition of the rows of A into tiles of at most R rows.
The accelerator processes one tile at a time: it loads every row of B that
the tile needs (one per distinct column index in the tile) into an on-chip
buffer, then computes the tile's R output rows.

* Contiguous tiling: a tile is R consecutive rows in some row order.
* Non-contiguous tiling: a tile is any set of R rows, chosen so that the rows
  share as many column indices as possible.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import reverse_cuthill_mckee

Tiling = list[np.ndarray]


# --------------------------------------------------------------------------
# Row orders for contiguous tiling
# --------------------------------------------------------------------------
def order_natural(a: sp.csr_matrix) -> np.ndarray:
    return np.arange(a.shape[0])


def order_random(a: sp.csr_matrix, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).permutation(a.shape[0])


def order_degree(a: sp.csr_matrix) -> np.ndarray:
    """Rows sorted by number of non-zeros (a common cheap reordering)."""
    return np.argsort(-np.diff(a.indptr), kind="stable")


def order_rcm(a: sp.csr_matrix) -> np.ndarray:
    """Reverse Cuthill-McKee: classic bandwidth-reducing reordering."""
    return np.asarray(reverse_cuthill_mckee(a, symmetric_mode=True))


def contiguous(order: np.ndarray, tile_rows: int) -> Tiling:
    """Cut a row order into consecutive tiles of `tile_rows` rows."""
    return [order[i : i + tile_rows] for i in range(0, len(order), tile_rows)]


# --------------------------------------------------------------------------
# Graph-based non-contiguous tiling
# --------------------------------------------------------------------------
def noncontiguous_greedy(
    a: sp.csr_matrix, tile_rows: int, score: str = "ratio"
) -> Tiling:
    """Greedy non-contiguous tiling on the row-overlap graph.

    View A as a bipartite graph (rows -- columns).  Two rows are connected in
    the *row-overlap graph* when they share a column, weighted by how many
    columns they share.  Rows that share columns reuse the same rows of B, so
    we want heavily connected rows in the same tile.

    Each tile is grown greedily:
      1. seed it with the unassigned row with the most non-zeros;
      2. repeatedly add the unassigned row that overlaps most with the
         columns already in the tile;
      3. stop at `tile_rows` rows.

    score = "overlap": pick the row with the most non-zeros already covered.
    score = "ratio"  : pick the row with the largest *fraction* of its
                       non-zeros already covered (adds the fewest new B rows
                       per reused one).

    The overlap graph is never materialised: when a column enters the tile we
    bump the score of every row touching that column (one CSC slice).
    """
    n, m = a.shape
    csc = a.tocsc()
    deg = np.diff(a.indptr).astype(np.float64)
    weight = np.ones(n) if score == "overlap" else 1.0 / np.maximum(deg, 1.0)
    # tiny static tie-break: among equal scores prefer rows with fewer non-zeros
    base = -deg / (deg.max() + 1.0) * 1e-6
    gain = base.copy()
    in_tile_col = np.zeros(m, dtype=bool)
    assigned = np.zeros(n, dtype=bool)
    seed_order = np.argsort(-deg, kind="stable")
    seed_ptr = 0
    tiles: Tiling = []
    remaining = n

    while remaining:
        while assigned[seed_order[seed_ptr]]:
            seed_ptr += 1
        row = int(seed_order[seed_ptr])
        tile = []
        tile_cols = []
        for _ in range(min(tile_rows, remaining)):
            if tile:
                row = int(np.argmax(gain))
            tile.append(row)
            assigned[row] = True
            base[row] = -np.inf
            gain[row] = -np.inf
            cols = a.indices[a.indptr[row] : a.indptr[row + 1]]
            new_cols = cols[~in_tile_col[cols]]
            in_tile_col[new_cols] = True
            tile_cols.append(new_cols)
            for j in new_cols:
                rows_j = csc.indices[csc.indptr[j] : csc.indptr[j + 1]]
                gain[rows_j] += weight[rows_j]
        remaining -= len(tile)
        tiles.append(np.asarray(tile))
        in_tile_col[np.concatenate(tile_cols)] = False
        np.copyto(gain, base)
    return tiles


def is_valid_tiling(tiles: Tiling, n_rows: int, tile_rows: int) -> bool:
    """Every row appears exactly once and no tile exceeds the row budget."""
    flat = np.concatenate(tiles)
    return (
        len(flat) == n_rows
        and np.array_equal(np.sort(flat), np.arange(n_rows))
        and max(len(t) for t in tiles) <= tile_rows
    )
