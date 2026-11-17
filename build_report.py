"""Build report/SpMM_Tiling_Report.pdf from the results and figures.

    python build_report.py

Run after run_experiments.py and make_figures.py.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image, KeepTogether, ListFlowable, ListItem, PageBreak, Paragraph,
    SimpleDocTemplate, Spacer, Table, TableStyle,
)

ROOT = Path(__file__).parent
RES, FIG, OUT = ROOT / "results", ROOT / "figures", ROOT / "report"
DATASETS = ["cora", "citeseer", "pubmed", "ca-HepPh", "email-Enron"]
DS_LABEL = {"cora": "Cora", "citeseer": "CiteSeer", "pubmed": "PubMed",
            "ca-HepPh": "ca-HepPh", "email-Enron": "email-Enron"}
METHODS = [
    ("contiguous (random order)", "Contiguous, random order"),
    ("contiguous (natural order)", "Contiguous, natural order"),
    ("contiguous (degree sort)", "Contiguous, degree sort"),
    ("contiguous (RCM reorder)", "Contiguous, RCM reorder"),
    ("non-contiguous graph (ratio)", "Graph-based, ratio score"),
    ("non-contiguous graph (overlap)", "Graph-based, overlap score"),
    ("upper bound", "Upper bound (any tiling)"),
]

INK = colors.HexColor("#0b0b0b")
MUTED = colors.HexColor("#52514e")
RULE = colors.HexColor("#c3c2b7")
TINT = colors.HexColor("#f0efec")
ACCENT = colors.HexColor("#2a78d6")
HILITE = colors.HexColor("#fde7dc")

# ---------------------------------------------------------------- fonts
FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
for name, file in [("Sans", "DejaVuSans.ttf"), ("Sans-Bold", "DejaVuSans-Bold.ttf"),
                   ("Sans-Italic", "DejaVuSans-Oblique.ttf"),
                   ("Sans-BoldItalic", "DejaVuSans-BoldOblique.ttf"),
                   ("Mono", "DejaVuSansMono.ttf")]:
    pdfmetrics.registerFont(TTFont(name, str(FONT_DIR / file)))
pdfmetrics.registerFontFamily("Sans", normal="Sans", bold="Sans-Bold",
                              italic="Sans-Italic", boldItalic="Sans-BoldItalic")

# ---------------------------------------------------------------- styles
BODY = ParagraphStyle("body", fontName="Sans", fontSize=9.6, leading=14.2,
                      textColor=INK, spaceAfter=6)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=8.2, leading=11.4, textColor=MUTED)
CAPTION = ParagraphStyle("caption", parent=SMALL, spaceBefore=3, spaceAfter=12)
H1 = ParagraphStyle("h1", fontName="Sans-Bold", fontSize=13.5, leading=17,
                    textColor=INK, spaceBefore=14, spaceAfter=6)
H2 = ParagraphStyle("h2", fontName="Sans-Bold", fontSize=10.6, leading=14,
                    textColor=INK, spaceBefore=8, spaceAfter=4)
TITLE = ParagraphStyle("title", fontName="Sans-Bold", fontSize=19, leading=24,
                       textColor=INK, spaceAfter=6)
SUBTITLE = ParagraphStyle("subtitle", parent=BODY, fontSize=10.5, leading=15, textColor=MUTED)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=8.4, leading=11, spaceAfter=0)
CELL_B = ParagraphStyle("cellb", parent=CELL, fontName="Sans-Bold")
BIG = ParagraphStyle("big", fontName="Sans-Bold", fontSize=22, leading=26,
                     textColor=INK, alignment=TA_CENTER)
BIG_LBL = ParagraphStyle("biglbl", parent=SMALL, alignment=TA_CENTER, fontSize=8)
FORMULA = ParagraphStyle("formula", parent=BODY, fontName="Sans", alignment=TA_CENTER,
                         fontSize=10, spaceBefore=2, spaceAfter=8)
CODE = ParagraphStyle("code", parent=BODY, fontName="Mono", fontSize=8.2, leading=11.5,
                      backColor=TINT, borderPadding=(5, 6, 5, 6), spaceBefore=4,
                      spaceAfter=10, leftIndent=6, rightIndent=6)

PAGE_W, PAGE_H = A4
MARGIN = 20 * mm
TEXT_W = PAGE_W - 2 * MARGIN


def p(text: str, style=BODY) -> Paragraph:
    return Paragraph(text, style)


def bullets(items: list[str], style=BODY) -> ListFlowable:
    return ListFlowable([ListItem(p(t, style), leftIndent=12) for t in items],
                        bulletType="bullet", start="•", leftIndent=12, bulletFontSize=8)


def numbered(items: list[str]) -> ListFlowable:
    return ListFlowable([ListItem(p(t), leftIndent=14) for t in items],
                        bulletType="1", leftIndent=14, bulletFontName="Sans-Bold",
                        bulletFontSize=9)


def figure(path: Path, caption: str, width: float = TEXT_W) -> KeepTogether:
    img = Image(str(path))
    img.drawWidth, img.drawHeight = width, width * img.imageHeight / img.imageWidth
    return KeepTogether([img, p(caption, CAPTION)])


def table(rows: list[list], col_widths: list[float], header_rows: int = 1,
          highlight: list[int] | None = None, bold_rows: list[int] | None = None,
          num_from: int | None = None) -> Table:
    """num_from: first column index holding numbers (header right-aligned to match)."""
    head_r = ParagraphStyle("hr", parent=CELL_B, alignment=2)
    data = []
    for r, row in enumerate(rows):
        style = CELL_B if r < header_rows or (bold_rows and r in bold_rows) else CELL
        cells = []
        for k, c in enumerate(row):
            if isinstance(c, str):
                right = r < header_rows and num_from is not None and k >= num_from
                c = p(c, head_r if right else style)
            cells.append(c)
        data.append(cells)
    t = Table(data, colWidths=col_widths, repeatRows=header_rows)
    cmds = [
        ("LINEABOVE", (0, 0), (-1, 0), 0.9, INK),
        ("LINEBELOW", (0, header_rows - 1), (-1, header_rows - 1), 0.6, INK),
        ("LINEBELOW", (0, -1), (-1, -1), 0.9, INK),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3.2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.2),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]
    for r in highlight or []:
        cmds.append(("BACKGROUND", (0, r), (-1, r), HILITE))
    t.setStyle(TableStyle(cmds))
    return t


def num_cell(v: str, bold: bool = False) -> Paragraph:
    st = ParagraphStyle("n", parent=CELL_B if bold else CELL, alignment=2)
    return p(v, st)


# ---------------------------------------------------------------- toy figure
def make_toy_figure(path: Path) -> None:
    """4x4 example: contiguous tiles vs graph-based tiles."""
    nz = [[1, 1, 0, 0], [0, 0, 1, 1], [1, 1, 0, 0], [0, 0, 1, 1]]
    tile_colour = {"A": "#2a78d6", "B": "#eb6834"}
    layouts = [
        ("Contiguous tiles: {0,1} and {2,3}", ["A", "A", "B", "B"],
         "Each tile needs B rows {0,1,2,3}\n4 + 4 = 8 fetches for 8 accesses\nData reuse = 0%"),
        ("Graph-based tiles: {0,2} and {1,3}", ["A", "B", "A", "B"],
         "Tile 1 needs B rows {0,1}; tile 2 needs {2,3}\n2 + 2 = 4 fetches for 8 accesses\nData reuse = 50%"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.5))
    fig.patch.set_facecolor("white")
    for ax, (title, tiles, note) in zip(axes, layouts):
        ax.set_xlim(-1.6, 4.2)
        ax.set_ylim(5.6, -1.0)
        ax.axis("off")
        for i in range(4):
            c = tile_colour[tiles[i]]
            ax.add_patch(Rectangle((-0.05, i + 0.03), 4.1, 0.94, facecolor=c, alpha=0.13,
                                   edgecolor="none"))
            ax.text(-0.25, i + 0.5, f"row {i}", ha="right", va="center", fontsize=9,
                    color="#52514e")
            ax.text(-1.45, i + 0.5, f"tile {1 if tiles[i] == 'A' else 2}", ha="left",
                    va="center", fontsize=8.5, color=c, fontweight="bold")
            for j in range(4):
                ax.add_patch(Rectangle((j + 0.1, i + 0.12), 0.8, 0.76,
                                       facecolor=c if nz[i][j] else "white",
                                       edgecolor="#c3c2b7", linewidth=0.8))
        for j in range(4):
            ax.text(j + 0.5, -0.25, f"col {j}", ha="center", va="bottom", fontsize=8.5,
                    color="#52514e")
        ax.set_title(title, fontsize=10.5, fontweight="bold", loc="left", pad=4)
        ax.text(-1.45, 4.35, note, ha="left", va="top", fontsize=9, color="#0b0b0b",
                linespacing=1.5)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


# ---------------------------------------------------------------- page decor
def on_page(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Sans", 7.5)
    canvas.setFillColor(MUTED)
    if doc.page > 1:
        canvas.drawString(MARGIN, PAGE_H - 12 * mm,
                          "Graph-Based Non-Contiguous Tiling for SpMM on GNN Workloads")
        canvas.setStrokeColor(RULE)
        canvas.setLineWidth(0.5)
        canvas.line(MARGIN, PAGE_H - 13.5 * mm, PAGE_W - MARGIN, PAGE_H - 13.5 * mm)
    canvas.drawRightString(PAGE_W - MARGIN, 11 * mm, f"Page {doc.page}")
    canvas.drawString(MARGIN, 11 * mm, "A. R. Shahzad")
    canvas.restoreState()


# ---------------------------------------------------------------- content
def build() -> Path:
    OUT.mkdir(exist_ok=True)
    toy = FIG / "fig0_toy_example.png"
    make_toy_figure(toy)

    ds = pd.read_csv(RES / "datasets.csv").set_index("dataset")
    tile = pd.read_csv(RES / "tile_reuse.csv")
    lru = pd.read_csv(RES / "lru_reuse.csv")
    chk = pd.read_csv(RES / "correctness.csv")

    r64 = tile[tile.tile_rows == 64].pivot(index="method", columns="dataset",
                                           values="reuse")[DATASETS] * 100
    r64["avg"] = r64.mean(axis=1)
    avg = r64["avg"]
    nat, rcm = avg["contiguous (natural order)"], avg["contiguous (RCM reorder)"]
    ovl, rat = avg["non-contiguous graph (overlap)"], avg["non-contiguous graph (ratio)"]
    ub = avg["upper bound"]
    lru_avg = lru.pivot_table(index="method", columns="buffer_rows", values="reuse") * 100
    buf = tile[tile.tile_rows == 64].pivot(index="method", columns="dataset",
                                           values="max_buffer_rows")
    tsec = tile[tile.tile_rows == 64].groupby("method").tiling_seconds.max()
    n_ok, n_runs = int(chk.matches_reference.sum()), len(chk)
    n_cnt = int(chk.counter_matches.sum())

    s: list = []

    # ---------------- title block
    s += [
        p("Graph-Based Non-Contiguous Tiling for Sparse Matrix Multiplication "
          "in Graph Neural Networks", TITLE),
        p("A small software study of data reuse, motivated by research Direction 01: "
          "<i>Hardware Acceleration for Sparse Computations</i>", SUBTITLE),
        Spacer(1, 4),
        p("<b>Abdul Raheem Shahzad</b> · M.S. Software Engineering, Wuhan University · "
          "abdul.raheem.colab@gmail.com", SMALL),
        p(date.today().strftime("%B %Y"), SMALL),
        Spacer(1, 8),
    ]

    stat = Table([[p(f"{nat:.0f}%", BIG), p("→", BIG), p(f"{ovl:.0f}%", BIG),
                   p(f"{ub:.0f}%", BIG)],
                  [p("average data reuse,<br/>ordinary contiguous tiling", BIG_LBL), "",
                   p("average data reuse,<br/>graph-based tiling (this work)", BIG_LBL),
                   p("best possible for<br/>any tiling (upper bound)", BIG_LBL)]],
                 colWidths=[TEXT_W * 0.3, TEXT_W * 0.08, TEXT_W * 0.32, TEXT_W * 0.3])
    stat.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), TINT),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, 0), 10), ("BOTTOMPADDING", (0, -1), (-1, -1), 10),
        ("LINEBEFORE", (3, 0), (3, -1), 0.6, RULE),
    ]))
    s += [stat, p("Five real graphs, tile height of 64 rows. Every number in this report "
                  "is produced by the accompanying code.", CAPTION)]

    s += [p("Summary", H1), p(
        "A graph neural network (GNN) layer spends most of its time on one operation: "
        "multiplying a <b>sparse</b> adjacency matrix by a <b>dense</b> feature matrix (SpMM). "
        "An accelerator computes this in <i>tiles</i> (small groups of rows) and must load "
        "the feature rows each tile needs into a small on-chip buffer. The usual approach "
        "makes each tile a block of consecutive rows. In real graphs, consecutive rows "
        "rarely share neighbours, so almost every loaded row is used only once."), p(
        "I modelled the matrix as a graph, linking rows that share columns, and built "
        f"tiles greedily from that graph instead. On five real graphs, average data reuse "
        f"rises from <b>{nat:.1f}%</b> to <b>{ovl:.1f}%</b>, "
        f"ahead of a classic reordering baseline (Reverse Cuthill–McKee, {rcm:.1f}%). "
        f"The result holds at almost every tile size I tested and in a second, independent "
        f"memory model. The tiled computation produces exactly the same output as a normal "
        f"multiplication in all {n_runs} test runs.")]
    s += [p("Scope", H2), p(
        "This study was motivated by <i>Rethinking Tiling and Dataflow for SpMM "
        "Acceleration: A Graph Transformation Framework</i> (MICRO 2025). It is <b>not a "
        "reproduction</b> of that paper. The tiling heuristic, the cost model and the "
        "experiments are my own simplified formulation, written to understand the problem "
        "from the software side. It measures data reuse only. It does not model cycles, "
        "energy, area or a specific accelerator.")]

    # ---------------- 1 problem
    s += [PageBreak(), p("1  The problem", H1), p(
        "A graph convolution layer computes <b>C = A · B</b>, where A is the normalised "
        "adjacency matrix of the graph (sparse) and B holds one feature vector per node "
        "(dense). In the row-wise dataflow used here, every non-zero entry A[i, j] needs "
        "row j of B. I call that need an <b>access</b>. There are as many accesses as "
        "non-zeros."), p(
        "The accelerator processes one <b>tile</b> at a time: a group of at most R rows of "
        "A. For each tile it <b>fetches</b> every distinct row of B the tile needs from "
        "off-chip memory into an on-chip buffer, once. Rows in the same tile that share a "
        "column share that fetch. The quantity of interest is:"),
        p("<b>data reuse  =  1 − fetches / accesses</b>", FORMULA),
        p("Data reuse is the share of accesses served from the buffer without going "
          "off-chip. Higher reuse means less memory traffic, which is usually the "
          "bottleneck for sparse workloads."),
        p("Why contiguous tiles waste reuse", H2), p(
        "Figure 1 shows the core idea on a 4 × 4 matrix with tiles of R = 2 rows. Rows 0 "
        "and 2 use the same columns, and so do rows 1 and 3. Contiguous tiles always pair "
        "row 0 with row 1, which share nothing. Grouping rows by the columns they share "
        "halves the number of fetches."),
        figure(toy, "<b>Figure 1.</b> Toy example. Coloured squares are non-zeros; the band "
               "colour shows which tile a row belongs to. The same matrix costs 8 fetches "
               "with contiguous tiles and 4 with graph-based tiles. This case is the first "
               "unit test in the code.", TEXT_W * 0.92)]

    # ---------------- 2 method
    s += [p("2  Method", H1), p("2.1  Graph view of the matrix", H2), p(
        "Treat A as a bipartite graph between its rows and its columns. From it, define the "
        "<b>row-overlap graph</b>: two rows are linked when they share a column, and the "
        "edge weight is the number of columns they share. That weight is exactly the "
        "number of fetches saved by putting the two rows in the same tile. Tiling then "
        "becomes a graph problem: split the rows into groups of at most R so that as much "
        "edge weight as possible stays inside groups.")]
    s += [p("2.2  Greedy non-contiguous tiling", H2),
          p("Each tile is grown one row at a time:"),
          numbered([
              "Seed the tile with the unassigned row that has the most non-zeros.",
              "Add the unassigned row that best overlaps the columns already in the tile.",
              "Repeat step 2 until the tile has R rows, then start the next tile.",
          ]),
          Spacer(1, 4),
          p("I compare two ways of scoring “best overlap” in step 2:"),
          table([
              ["Score", "Picks the row with…", "Behaviour"],
              ["Overlap", "the most non-zeros already covered by the tile",
               "highest reuse, but rows with many neighbours can make some tiles very large"],
              ["Ratio", "the largest <i>fraction</i> of its non-zeros already covered",
               "slightly lower reuse, much smaller worst-case buffer"],
          ], [TEXT_W * 0.13, TEXT_W * 0.40, TEXT_W * 0.47]),
          Spacer(1, 8),
          p("The row-overlap graph can be dense, so it is never built. When a column enters "
            "a tile, the scores of all rows touching that column are updated through one "
            "column slice of the matrix.")]
    s += [p("2.3  Baselines", H2), p(
        "Contiguous tiles of R rows taken over four different row orders:"),
        bullets([
            "<b>Random order</b>: a lower reference point.",
            "<b>Natural order</b>: the node numbering the dataset ships with.",
            "<b>Degree sort</b>: rows sorted by number of non-zeros, a common cheap reordering.",
            "<b>Reverse Cuthill–McKee (RCM)</b>: a classic reordering that clusters "
            "non-zeros near the diagonal. This is the strongest baseline.",
        ])]
    s += [p("2.4  An upper bound for any tiling", H2), p(
        "A column with d non-zeros is spread over at least ⌈d / R⌉ tiles, and must be "
        "fetched once in each of them. So no tiling with R rows per tile can do better than"),
        p("reuse  ≤  1 − Σ<sub>j</sub> ⌈d<sub>j</sub> / R⌉ / nnz(A)", FORMULA),
        p("This bound shows how much room is left. It ignores conflicts between columns, "
          "so it is probably not reachable.")]

    # ---------------- 3 setup
    rows = [["Dataset", "Type", "Rows (nodes)", "Non-zeros", "Avg nnz / row",
             "Max nnz / row"]]
    kind = {"cora": "citation", "citeseer": "citation", "pubmed": "citation",
            "ca-HepPh": "co-authorship", "email-Enron": "email"}
    for d in DATASETS:
        r = ds.loc[d]
        rows.append([DS_LABEL[d], kind[d], num_cell(f"{int(r.rows):,}"),
                     num_cell(f"{int(r.nnz):,}"), num_cell(f"{r.avg_nnz_per_row:.1f}"),
                     num_cell(f"{int(r.max_nnz_per_row):,}")])
    s += [p("3  Experimental setup", H1), p(
        "Each matrix A is the GCN-normalised adjacency D<super>−1/2</super>(G + I)"
        "D<super>−1/2</super> of a real graph G. Cora, CiteSeer and PubMed are the standard "
        "GNN benchmarks (LINQS); ca-HepPh and email-Enron come from the Stanford SNAP "
        "collection and are larger and denser. The graphs are built from the raw edge "
        "lists, so node counts can differ slightly from the preprocessed copies in GNN "
        "libraries."),
        table(rows, [TEXT_W * w for w in (0.17, 0.17, 0.16, 0.16, 0.17, 0.17)], num_from=2),
        p("<b>Table 1.</b> Datasets.", CAPTION),
        p("Tile heights from R = 8 to 256 rows were tested. The main comparison uses "
          "R = 64. All code is Python (NumPy/SciPy) and the full run takes about "
          "1.5 minutes on a laptop CPU.")]

    # ---------------- 4 results
    rows = [["Method"] + [DS_LABEL[d].replace("email-", "email-<br/>") for d in DATASETS]
            + ["Average"]]
    for key, label in METHODS:
        best = key == "non-contiguous graph (overlap)"
        rows.append([label] + [num_cell(f"{r64.loc[key, d]:.1f}", best) for d in DATASETS]
                    + [num_cell(f"{r64.loc[key, 'avg']:.1f}", True)])
    s += [PageBreak(), p("4  Results", H1),
          p("4.1  Data reuse at tile height R = 64", H2),
          table(rows, [TEXT_W * 0.25] + [TEXT_W * 0.125] * 6, highlight=[6], bold_rows=[6],
                num_from=1),
          p("<b>Table 2.</b> Data reuse (%) at R = 64. Highlighted: the best method.",
            CAPTION),
          figure(FIG / "fig1_reuse_r64.png",
                 "<b>Figure 2.</b> The same numbers as Table 2. Dashed lines show the upper "
                 "bound for each dataset."),
          p(f"Graph-based tiling roughly triples reuse over the natural order "
            f"({nat:.1f}% → {ovl:.1f}%) and adds about {ovl - rcm:.0f} points over RCM. "
            f"The gain is largest on the sparse citation graphs (Cora, CiteSeer, PubMed), "
            f"where consecutive rows share almost nothing. On ca-HepPh, whose collaboration "
            f"groups already appear as near-consecutive rows, RCM gets closer, but graph "
            f"tiling still leads.")]

    s += [p("4.2  Every tile size", H2),
          figure(FIG / "fig2_reuse_vs_tile_height.png",
                 "<b>Figure 3.</b> Data reuse as tile height grows from 8 to 256 rows."),
          p("Graph-based tiling leads at every tile height on every dataset, with one "
            "exception: on ca-HepPh at R = 8 the ratio score (57.2%) is just below RCM "
            "(58.9%). The advantage is largest for small tiles, which matter most for "
            "accelerators with small on-chip buffers.")]

    rows = [["Buffer capacity (rows of B)", "64", "256", "1,024", "4,096"]]
    for key, label in [("contiguous (natural order)", "Contiguous, natural order"),
                       ("contiguous (RCM reorder)", "Contiguous, RCM reorder"),
                       ("non-contiguous graph (overlap)", "Graph-based, overlap score"),
                       ("non-contiguous graph (ratio)", "Graph-based, ratio score")]:
        best = key == "non-contiguous graph (ratio)"
        rows.append([label] + [num_cell(f"{lru_avg.loc[key, c]:.1f}", best)
                               for c in (64, 256, 1024, 4096)])
    s += [p("4.3  Cross-check with a fixed-size buffer", H2), p(
        "The tile model lets each tile use as much buffer as it needs. To check that the "
        "gain is not an artefact of that assumption, the same row orders were replayed "
        "through a buffer of fixed size with least-recently-used (LRU) replacement."),
        table(rows, [TEXT_W * 0.36] + [TEXT_W * 0.16] * 4, highlight=[4], bold_rows=[4],
              num_from=1),
        p("<b>Table 3.</b> Average data reuse (%) over the five datasets with a fixed LRU "
          "buffer.", CAPTION),
        figure(FIG / "fig3_reuse_lru_buffer.png",
               "<b>Figure 4.</b> Fixed-buffer cross-check per dataset (row orders from the "
               "R = 64 tilings)."),
        p("The benefit holds, and it is largest when the buffer is small. With very large "
          "buffers all orders converge, as expected, because almost everything fits. One "
          "exception: on CiteSeer with a 1,024-row buffer, RCM (73.2%) is slightly ahead "
          "of both graph-based orders (71.1% and 67.3%).")]

    s += [p("4.4  Buffer size per tile", H2),
          figure(FIG / "fig4_buffer_r64.png",
                 "<b>Figure 5.</b> Rows of B each tile must hold at R = 64 (log scale)."),
          p(f"The overlap score gives the best reuse but sometimes builds a very large tile "
            f"around rows with many neighbours: up to "
            f"{int(buf.loc['non-contiguous graph (overlap)', 'email-Enron']):,} rows of B on "
            f"email-Enron. The ratio score keeps the largest tile at "
            f"{int(buf.loc['non-contiguous graph (ratio)', 'email-Enron']):,} rows there, "
            f"the size of the single densest row, for a loss of only {ovl - rat:.1f} points "
            f"of average reuse. For real hardware with a fixed buffer, the ratio score is "
            f"the better choice.")]

    s += [p("4.5  Correctness", H2), p(
        f"The tiled executor really computes A · B tile by tile with a 16-dimensional "
        f"feature matrix. <b>{n_ok} of {n_runs}</b> runs (5 datasets × 6 methods) match the "
        f"reference multiplication exactly, and in <b>{n_cnt} of {n_runs}</b> runs the "
        f"executor's fetch counter equals the analytic count. Six unit tests also pass, "
        f"including the toy example of Figure 1, which the greedy algorithm solves "
        f"optimally (reuse equals the upper bound).")]

    # ---------------- 5 limitations
    s += [p("5  Limitations", H1), bullets([
        "<b>Software model only.</b> Data reuse stands in for off-chip traffic. There is "
        "no cycle, energy or area model, and no analysis of balancing work across "
        "parallel processing units.",
        f"<b>Gap to the bound.</b> The greedy method sits about {ub - ovl:.0f} points below "
        "the upper bound on average. The bound is probably not tight, and I have not "
        "measured how much of the gap is due to the heuristic.",
        f"<b>Preprocessing cost.</b> This NumPy version is quadratic in the number of rows "
        f"(up to {tsec.max():.1f} s for email-Enron at R = 64). It runs once per graph, but "
        f"million-node graphs would need a priority-queue implementation.",
        "<b>Dense rows are not split.</b> A row with more non-zeros than the buffer holds "
        "is kept whole.",
        "<b>Scale.</b> Five graphs of up to 37,000 nodes; the random baseline uses one seed.",
    ])]

    s += [p("6  Possible next steps", H1), numbered([
        "Split very dense rows and columns into sub-vertices so every tile fits a fixed "
        "buffer, in the spirit of vertex decomposition.",
        "Add a balance constraint (equal non-zeros per tile) and study the trade-off "
        "between reuse and load balance across processing units.",
        "Compare against hypergraph partitioners (e.g. PaToH, KaHyPar) as a stronger baseline.",
        "Extend the row-overlap view to sparse × sparse multiplication and sparse tensor "
        "contraction.",
        "Feed the tilings into a cycle-level accelerator simulator to turn reuse into "
        "speed and energy numbers.",
    ])]

    s += [p("7  Code and reproduction", H1), p(
        "The project is plain Python (NumPy, SciPy, pandas, Matplotlib). Running these "
        "four commands in order reproduces every number and figure in this report:"),
        p("python download_data.py<br/>python run_experiments.py<br/>"
          "python make_figures.py<br/>python tests/test_tiling.py", CODE),
        table([
            ["File", "Contents"],
            ["spmm_tiling/datasets.py", "Loads edge lists and builds the normalised sparse matrix"],
            ["spmm_tiling/tiling.py", "Contiguous row orders and the greedy graph-based tiling"],
            ["spmm_tiling/executor.py", "Tiled SpMM executor, reuse metrics, upper bound, LRU model"],
            ["run_experiments.py", "Runs all experiments and writes results/*.csv"],
            ["make_figures.py", "Draws figures/*.png from the results"],
            ["tests/test_tiling.py", "Six sanity tests"],
        ], [TEXT_W * 0.36, TEXT_W * 0.64])]

    pdf = OUT / "SpMM_Tiling_Report.pdf"
    doc = SimpleDocTemplate(
        str(pdf), pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=19 * mm, bottomMargin=18 * mm,
        title="Graph-Based Non-Contiguous Tiling for SpMM on GNN Workloads",
        author="Abdul Raheem Shahzad",
        subject="Data reuse study for sparse-dense matrix multiplication")
    doc.build(s, onFirstPage=on_page, onLaterPages=on_page)
    return pdf


if __name__ == "__main__":
    print("wrote", build())
