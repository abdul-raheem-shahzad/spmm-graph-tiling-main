"""Load real graphs as GCN-normalised sparse adjacency matrices.

Every loader returns the matrix a GCN layer multiplies by its dense feature
matrix:  A_hat = D^-1/2 (A + I) D^-1/2   (symmetric, with self loops).

Rows are numbered in the dataset's *natural order* (sorted original node id),
which is the order a contiguous tiling scheme would see.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.sparse as sp

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"

DATASETS = ["cora", "citeseer", "pubmed", "ca-HepPh", "email-Enron"]


def _edges_to_adj(src: list, dst: list) -> sp.csr_matrix:
    ids = sorted(set(src) | set(dst))
    index = {v: i for i, v in enumerate(ids)}
    s = np.fromiter((index[v] for v in src), dtype=np.int64, count=len(src))
    d = np.fromiter((index[v] for v in dst), dtype=np.int64, count=len(dst))
    n = len(ids)
    a = sp.coo_matrix((np.ones(len(s)), (s, d)), shape=(n, n)).tocsr()
    a = a + a.T + sp.identity(n, format="csr")  # symmetrise + self loops
    a.data[:] = 1.0
    a.sum_duplicates()
    return a


def gcn_normalise(a: sp.csr_matrix) -> sp.csr_matrix:
    deg = np.asarray(a.sum(axis=1)).ravel()
    d_inv_sqrt = sp.diags(1.0 / np.sqrt(deg))
    out = (d_inv_sqrt @ a @ d_inv_sqrt).tocsr()
    out.sort_indices()
    return out


def _read_pairs(path: Path, skip_prefix: str = "#") -> tuple[list, list]:
    src, dst = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.startswith(skip_prefix):
                continue
            a, b = line.split()[:2]
            src.append(a)
            dst.append(b)
    return src, dst


def _read_pubmed(path: Path) -> tuple[list, list]:
    src, dst = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            # data lines look like:  <id>  paper:123  |  paper:456
            if len(parts) == 4 and parts[2] == "|":
                src.append(int(parts[1].split(":")[1]))
                dst.append(int(parts[3].split(":")[1]))
    return src, dst


def load(name: str) -> sp.csr_matrix:
    if name == "cora":
        s, d = _read_pairs(RAW / "cora/cora/cora.cites")
        s, d = [int(x) for x in s], [int(x) for x in d]
    elif name == "citeseer":
        s, d = _read_pairs(RAW / "citeseer/citeseer/citeseer.cites")
    elif name == "pubmed":
        s, d = _read_pubmed(
            RAW / "pubmed/Pubmed-Diabetes/data/Pubmed-Diabetes.DIRECTED.cites.tab"
        )
    elif name in ("ca-HepPh", "email-Enron"):
        s, d = _read_pairs(RAW / name / f"{name}.txt")
        s, d = [int(x) for x in s], [int(x) for x in d]
    else:
        raise ValueError(f"unknown dataset {name!r}; choose from {DATASETS}")
    return gcn_normalise(_edges_to_adj(s, d))
