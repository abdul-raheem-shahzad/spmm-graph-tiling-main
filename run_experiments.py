"""Run all experiments and write CSV results to results/.

    python run_experiments.py

Experiment 1  tile model : data reuse vs tile height R, every method + upper bound
Experiment 2  LRU buffer : data reuse with a fixed-capacity B-row buffer
Experiment 3  correctness: tiled SpMM output equals the reference A @ B
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from spmm_tiling import datasets
from spmm_tiling import executor as E
from spmm_tiling import tiling as T

OUT = Path(__file__).parent / "results"
TILE_ROWS = [8, 16, 32, 64, 128, 256]
LRU_TILE_ROWS = 64  # tile height used to build the row orders for experiment 2
LRU_CAPACITY = [64, 256, 1024, 4096]
FEATURE_DIM = 16

CONTIGUOUS = {
    "contiguous (random order)": T.order_random,
    "contiguous (natural order)": T.order_natural,
    "contiguous (degree sort)": T.order_degree,
    "contiguous (RCM reorder)": T.order_rcm,
}
GRAPH = {
    "non-contiguous graph (overlap)": "overlap",
    "non-contiguous graph (ratio)": "ratio",
}


def main() -> None:
    OUT.mkdir(exist_ok=True)
    info, tile_rows_out, lru_rows_out, check_rows = [], [], [], []
    rng = np.random.default_rng(0)

    for name in datasets.DATASETS:
        a = datasets.load(name)
        n = a.shape[0]
        deg = np.diff(a.indptr)
        info.append(dict(dataset=name, rows=n, nnz=a.nnz, avg_nnz_per_row=a.nnz / n,
                         max_nnz_per_row=int(deg.max()), density=a.nnz / n**2))
        print(f"== {name}: {n} rows, {a.nnz} non-zeros")

        orders = {}
        t0 = time.perf_counter()
        for label, fn in CONTIGUOUS.items():
            t = time.perf_counter()
            orders[label] = (fn(a), time.perf_counter() - t)

        b = rng.standard_normal((n, FEATURE_DIM))
        reference = a @ b

        for r in TILE_ROWS:
            tilings = {}
            for label, (order, secs) in orders.items():
                tilings[label] = (T.contiguous(order, r), secs)
            for label, score in GRAPH.items():
                t = time.perf_counter()
                tl = T.noncontiguous_greedy(a, r, score)
                tilings[label] = (tl, time.perf_counter() - t)

            for label, (tl, secs) in tilings.items():
                assert T.is_valid_tiling(tl, n, r), (name, label, r)
                s = E.tile_stats(a, tl)
                tile_rows_out.append(dict(
                    dataset=name, method=label, tile_rows=r, reuse=s.reuse,
                    fetches=s.fetches, accesses=s.accesses,
                    mean_buffer_rows=s.mean_buffer_rows,
                    max_buffer_rows=s.max_buffer_rows, tiling_seconds=secs))

                if r == LRU_TILE_ROWS:
                    # Experiment 3: really execute the tiled SpMM and compare.
                    c, s_exec = E.tiled_spmm(a, b, tl)
                    check_rows.append(dict(
                        dataset=name, method=label,
                        max_abs_error=float(np.abs(c - reference).max()),
                        matches_reference=bool(np.allclose(c, reference)),
                        executor_fetches=s_exec.fetches,
                        counter_matches=bool(s_exec.fetches == s.fetches)))
                    # Experiment 2: fixed-capacity LRU buffer over the same row order.
                    order = np.concatenate(tl)
                    for cap in LRU_CAPACITY:
                        lru_rows_out.append(dict(
                            dataset=name, method=label, buffer_rows=cap,
                            reuse=E.lru_reuse(a, order, cap)))

            tile_rows_out.append(dict(
                dataset=name, method="upper bound", tile_rows=r,
                reuse=E.reuse_upper_bound(a, r)))
        print(f"   done in {time.perf_counter() - t0:.1f}s")

    pd.DataFrame(info).to_csv(OUT / "datasets.csv", index=False)
    pd.DataFrame(tile_rows_out).to_csv(OUT / "tile_reuse.csv", index=False)
    pd.DataFrame(lru_rows_out).to_csv(OUT / "lru_reuse.csv", index=False)
    pd.DataFrame(check_rows).to_csv(OUT / "correctness.csv", index=False)

    df = pd.DataFrame(tile_rows_out)
    table = (df[df.tile_rows == 64].pivot(index="method", columns="dataset", values="reuse")
             [datasets.DATASETS])
    table["average"] = table.mean(axis=1)
    print("\nData reuse at tile height R = 64")
    print((table * 100).round(1).sort_values("average").to_string())
    chk = pd.DataFrame(check_rows)
    print(f"\nCorrectness: {chk.matches_reference.sum()}/{len(chk)} tiled runs match A @ B, "
          f"{chk.counter_matches.sum()}/{len(chk)} fetch counters agree, "
          f"max abs error {chk.max_abs_error.max():.2e}")


if __name__ == "__main__":
    main()
