"""Draw the figures in figures/ from the CSV files in results/.

    python make_figures.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).parent
RES, FIG = ROOT / "results", ROOT / "figures"
DATASETS = ["cora", "citeseer", "pubmed", "ca-HepPh", "email-Enron"]

# method -> (short label, colour, marker).  Fixed colour per method in every figure.
SERIES = {
    "contiguous (random order)": ("Contiguous, random order", "#898781", "v"),
    "contiguous (natural order)": ("Contiguous, natural order", "#2a78d6", "o"),
    "contiguous (RCM reorder)": ("Contiguous, RCM reorder", "#eda100", "s"),
    "non-contiguous graph (ratio)": ("Graph-based, ratio score", "#1baf7a", "D"),
    "non-contiguous graph (overlap)": ("Graph-based, overlap score", "#eb6834", "^"),
}
INK, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#fcfcfb"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "text.color": INK,
    "axes.edgecolor": AXIS, "axes.labelcolor": MUTED, "axes.titlesize": 10,
    "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "legend.frameon": False,
})


def pct(ax) -> None:
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")


def fig_headline(tile: pd.DataFrame) -> None:
    d = tile[tile.tile_rows == 64]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    groups = DATASETS + ["average"]
    width = 0.15
    for k, (method, (label, colour, _)) in enumerate(SERIES.items()):
        vals = d[d.method == method].set_index("dataset").reuse.reindex(DATASETS) * 100
        vals = list(vals) + [vals.mean()]
        xs = [g + (k - 2) * (width + 0.015) for g in range(len(groups))]
        ax.bar(xs, vals, width, color=colour, label=label)
        if method in ("contiguous (natural order)", "non-contiguous graph (overlap)"):
            for x, v in zip(xs, vals):
                ax.text(x, v + 1.2, f"{v:.0f}", ha="center", va="bottom", fontsize=7.5, color=INK)
    ub = d[d.method == "upper bound"].set_index("dataset").reuse.reindex(DATASETS) * 100
    ub = list(ub) + [ub.mean()]
    for g, v in enumerate(ub):
        ax.hlines(v, g - 0.42, g + 0.42, colors=INK, linestyles=(0, (4, 3)), linewidth=1,
                  label="Upper bound for any tiling" if g == 0 else None)
    ax.axvline(len(DATASETS) - 0.5, color=AXIS, linewidth=0.8)
    ax.set_xticks(range(len(groups)), groups)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Data reuse (share of B-row accesses served from buffer)")
    ax.grid(axis="x", visible=False)
    pct(ax)
    ax.set_title("Data reuse at tile height R = 64 rows", loc="left")
    ax.legend(ncols=3, loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_reuse_r64.png", dpi=200)
    plt.close(fig)


def small_multiples(df: pd.DataFrame, xcol: str, xlabel: str, title: str, fname: str,
                    with_bound: bool) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(13, 3.4), sharey=True)
    for ax, name in zip(axes, DATASETS):
        d = df[df.dataset == name]
        for method, (label, colour, marker) in SERIES.items():
            s = d[d.method == method].sort_values(xcol)
            ax.plot(s[xcol], s.reuse * 100, color=colour, marker=marker, markersize=4.5,
                    linewidth=1.8, label=label)
        if with_bound:
            s = d[d.method == "upper bound"].sort_values(xcol)
            ax.plot(s[xcol], s.reuse * 100, color=INK, linestyle=(0, (4, 3)), linewidth=1,
                    label="Upper bound for any tiling")
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(d[xcol].unique()))
        ax.xaxis.set_major_formatter(lambda v, _: f"{int(v)}")
        ax.tick_params(axis="x", labelsize=7.5)
        ax.minorticks_off()
        ax.set_title(name, loc="left")
        ax.set_xlabel(xlabel)
        ax.set_ylim(0, 100)
        pct(ax)
    axes[0].set_ylabel("Data reuse")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncols=6, loc="lower center", fontsize=8)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0.08, 1, 0.97))
    fig.savefig(FIG / fname, dpi=200)
    plt.close(fig)


def fig_buffer(tile: pd.DataFrame) -> None:
    d = tile[(tile.tile_rows == 64) & (tile.method.isin(SERIES))]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, col, title in [(axes[0], "mean_buffer_rows", "Mean buffer per tile (rows of B)"),
                           (axes[1], "max_buffer_rows", "Largest buffer of any tile (rows of B)")]:
        width = 0.15
        for k, (method, (label, colour, _)) in enumerate(SERIES.items()):
            vals = d[d.method == method].set_index("dataset")[col].reindex(DATASETS)
            xs = [g + (k - 2) * (width + 0.015) for g in range(len(DATASETS))]
            ax.bar(xs, vals, width, color=colour, label=label)
        ax.set_xticks(range(len(DATASETS)), DATASETS, fontsize=8)
        ax.set_yscale("log")
        ax.grid(axis="x", visible=False)
        ax.set_title(title, loc="left")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncols=5, loc="lower center", fontsize=8)
    fig.suptitle("Buffer needed per tile at R = 64 (log scale; lower is better)", x=0.01,
                 ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(FIG / "fig4_buffer_r64.png", dpi=200)
    plt.close(fig)


def main() -> None:
    FIG.mkdir(exist_ok=True)
    tile = pd.read_csv(RES / "tile_reuse.csv")
    lru = pd.read_csv(RES / "lru_reuse.csv")
    fig_headline(tile)
    small_multiples(tile, "tile_rows", "Tile height R (rows of A)",
                    "Data reuse vs tile height", "fig2_reuse_vs_tile_height.png", True)
    small_multiples(lru, "buffer_rows", "Buffer capacity (rows of B)",
                    "Cross-check: data reuse with a fixed-size LRU buffer (row order from R = 64 tiling)",
                    "fig3_reuse_lru_buffer.png", False)
    fig_buffer(tile)
    print("wrote", ", ".join(p.name for p in sorted(FIG.glob("*.png"))))


if __name__ == "__main__":
    main()
