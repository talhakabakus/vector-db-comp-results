"""High-level architecture overview of the study (S1-S5 pipeline, systems under test, experiments E1-E6, RQ1-RQ4).

Usage: python architecture_overview.py  ->  Fig01_architecture_overview.png (600 dpi) next to this script.
Palette: Okabe-Ito (systems take the first three slots: Qdrant, Weaviate, pgvector).
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OI = ["#0072B2", "#D55E00", "#E69F00", "#009E73", "#56B4E9", "#F0E442", "#CC79A7", "#000000"]
QD, WV, PG, DATA, BENCH, YEL, RAG, INK = OI
STAT = "#555555"
OUT = Path(__file__).with_name("Fig01_architecture_overview.png")

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "savefig.dpi": 600})
fig, ax = plt.subplots(figsize=(19.5, 11))
ax.set_xlim(0, 200)
ax.set_ylim(3, 102)
ax.axis("off")


def tint(hex_color, a):
    """Blend a colour with white (a = share of the colour)."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return "#%02x%02x%02x" % tuple(round(255 - (255 - c) * a) for c in (r, g, b))


def box(x, y, w, h, edge, fill_a=0.12, lw=2.2, ls="-", r=1.6, z=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=tint(edge, fill_a) if fill_a else "white", ec=edge, lw=lw, ls=ls, zorder=z))


def header(x, y, w, text, color, size=15.5):
    """Coloured title strip at the top of a panel (y = top edge)."""
    ax.add_patch(FancyBboxPatch((x, y - 6.2), w, 6.2, boxstyle="round,pad=0,rounding_size=1.6",
                                fc=color, ec=color, lw=0, zorder=3))
    ax.text(x + w / 2, y - 3.1, text, ha="center", va="center", color="white", fontsize=size,
            fontweight="bold", zorder=4)


def lines(x, y, items, size=12.5, gap=4.3, color=INK, bullet="•"):
    for i, t in enumerate(items):
        cont = t.startswith("   ")            # continuation line: no bullet
        ax.text(x, y - i * gap, f"{bullet} {t}" if bullet and not cont else t, ha="left", va="top", fontsize=size,
                color=color, zorder=4)


def arrow(p, q, color=INK, label=None, lpos=0.5, loff=(0, 1.6), rad=0.0, lw=2.4, size=11.5, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=22, lw=lw, color=color, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", zorder=5, shrinkA=2, shrinkB=2))
    if label:
        lx = p[0] + (q[0] - p[0]) * lpos + loff[0]
        ly = p[1] + (q[1] - p[1]) * lpos + loff[1]
        ax.text(lx, ly, label, ha="center", va="bottom", fontsize=size, color=color, style="italic",
                zorder=6, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9))


# ------------------------------------------------------------------ column 1: S1 data preparation (GPU)
X1, W1 = 1, 40
box(X1, 12, W1, 89, DATA)
header(X1, 101, W1, "S1 · Data preparation", DATA)
ax.text(X1 + W1 / 2, 92.2, "computed on GPU", ha="center", va="top", fontsize=12, color=DATA, style="italic")

sub = [("Corpora", ["5 BEIR: SciFact, NFCorpus,", "   FiQA, TREC-COVID, Quora", "2 QA: SQuAD v1.1,", "   HotpotQA-sub",
                     "3.6K – 523K documents"]),
       ("Embeddings", ["bge-small-en-v1.5 (384-d)", "bge-base-en-v1.5 (768-d)", "computed once, L2-normalized"]),
       ("Ground truth", ["exact top-100 (GPU)", "+ exact filtered top-100", "human relevance judgments"]),
       ("Filter attributes", ["selectivity 1/2 … 1/1,000"])]
y = 87.5
for title, items in sub:
    h = 4.6 + 4.1 * len(items)
    box(X1 + 1.6, y - h, W1 - 3.2, h, DATA, fill_a=0, lw=1.4, r=1.0, z=2)
    ax.text(X1 + 3.2, y - 1.0, title, ha="left", va="top", fontsize=13.5, fontweight="bold", color=DATA, zorder=4)
    lines(X1 + 3.2, y - 5.0, items, size=12, gap=4.1, bullet="")
    y -= h + 1.6

# ------------------------------------------------------------------ column 2: systems under test + hardware
X2, W2 = 48, 51
box(X2, 22, W2, 79, INK, fill_a=0.04)
header(X2, 101, W2, "Systems under test", INK)
box(X2 + 2, 83.4, W2 - 4, 9.8, INK, fill_a=0, lw=1.4, r=1.0, z=2)                 # hardware (Kaggle)
ax.text(X2 + 4, 90.9, "CPU", ha="left", va="center", fontsize=12.5, fontweight="bold", zorder=4)
ax.text(X2 + 11.5, 90.9, "4 vCPU, 31 GB RAM: all databases", ha="left", va="center", fontsize=11.8, zorder=4)
ax.text(X2 + 4, 85.8, "GPU", ha="left", va="center", fontsize=12.5, fontweight="bold", zorder=4)
ax.text(X2 + 11.5, 85.8, "2 × NVIDIA T4: embeddings, LLM", ha="left", va="center", fontsize=11.8, zorder=4)
systems = [("Qdrant 1.20", "native · Rust · gRPC", "payload-filtered HNSW; sparse vectors", QD),
           ("Weaviate 1.40", "native · Go · gRPC", "ACORN filtering; native BM25 + fusion", WV),
           ("PostgreSQL 16 + pgvector 0.8.7", "extended · SQL wire protocol", "cost-based planner; B-tree; sparsevec", PG)]
sy = [69.2, 54.6, 40.0]
for (name, kind, feat, c), y0 in zip(systems, sy):
    box(X2 + 2, y0, W2 - 4, 12.6, c, fill_a=0.16, lw=2.4, z=2)
    ax.text(X2 + 4, y0 + 10.0, name, ha="left", va="center", fontsize=13.2, fontweight="bold", color=c, zorder=4)
    ax.text(X2 + 4, y0 + 6.0, kind, ha="left", va="center", fontsize=12.2, color=INK, zorder=4)
    ax.text(X2 + 4, y0 + 2.4, feat, ha="left", va="center", fontsize=11.8, color="#333333", zorder=4)
box(X2 + 2, 23.6, W2 - 4, 14.8, INK, fill_a=0, lw=1.4, ls=(0, (4, 2)), r=1.0, z=2)
ax.text(X2 + W2 / 2, 37.4, "Common configuration", ha="center", va="top", fontsize=13.5, fontweight="bold", zorder=4)
lines(X2 + 4, 33.3, ["HNSW, M = 16, efConstruction = 128", "cosine distance; ef swept at query time",
                     "identical vectors for all systems"], size=11.4, gap=3.6)

# ------------------------------------------------------------------ column 3: experiments
X3, W3 = 106, 45
exps = [("S2 · System benchmark", BENCH, 101, 30.5,
         ["E1  ingest time, memory (PSS), disk", "E2  recall–QPS sweep, iso-recall,", "      concurrency (1–8), null query",
          "E2b each system's default config", "3 builds × 2 passes, k = 10 / 100"]),
       ("S3 · Retrieval experiments", QD, 68.5, 26.5,
        ["E3  recall & nDCG@10 vs ef", "E4  filtered search (1/2 … 1/1,000)", "E5  hybrid: dense + BM25, RRF",
         "per-query result lists saved"]),
       ("S4 · End-to-end RAG", RAG, 37.5, 25.5,
        ["E6  top-5 passages → Qwen2.5-7B", "      (4-bit, greedy) on GPU", "2 QA datasets × 1,000 questions",
         "EM, F1, hit@5 · 96 configurations"])]
for title, c, top, h, items in exps:
    box(X3, top - h, W3, h, c, fill_a=0.10)
    header(X3, top, W3, title, c, size=14.5)
    lines(X3 + 2.2, top - 8.3, items, size=12, gap=4.1, bullet="")

# ------------------------------------------------------------------ column 4: statistics + outputs
X4, W4 = 158, 41
box(X4, 60, W4, 41, STAT, fill_a=0.08)
header(X4, 101, W4, "S5 · Statistics", STAT, size=14.5)
lines(X4 + 2, 92.8, ["Friedman + Nemenyi CD", "Wilcoxon (Pratt) · McNemar", "Holm per family",
                     "paired bootstrap 95 % CI", "TOST equivalence (±2 pts)", "logistic EM ~ hit@5 + system",
                     "   (query-clustered SEs)"], size=12, gap=4.6)
rqs = [("RQ1", "system performance", BENCH), ("RQ2", "retrieval quality", QD),
       ("RQ3", "answer quality", RAG), ("RQ4", "filtered & hybrid search", QD)]
for i, (rq, t, c) in enumerate(rqs):
    y0 = 50.0 - i * 6.6
    box(X4, y0, W4, 5.4, c, fill_a=0.18, lw=1.8, r=1.0)
    ax.text(X4 + 2, y0 + 2.7, rq, ha="left", va="center", fontsize=13.5, fontweight="bold", color=c, zorder=4)
    ax.text(X4 + 8.5, y0 + 2.7, t, ha="left", va="center", fontsize=12.2, color=INK, zorder=4)

# ------------------------------------------------------------------ arrows
arrow((X1 + W1, 62), (X2, 62), DATA, "identical\nvectors", loff=(0, 1.2))
arrow((X2 + W2, 80), (X3, 80), INK)
arrow((X2 + W2, 55), (X3, 55), INK)
arrow((X3 + W3 / 2, 42.0), (X3 + W3 / 2, 37.6), QD, lw=2.6)          # S3 -> S4 (lists)
ax.text(X3 + W3 / 2 + 1.8, 39.8, "retrieved lists", ha="left", va="center", fontsize=11.5, color=QD,
        style="italic", zorder=6)
arrow((X1 + W1, 17.5), (X3, 17.5), DATA, "questions, gold answers & gold passages", lpos=0.42, loff=(0, -4.6))
for yy, ty in ((80, 82), (55, 73), (27, 64)):
    arrow((X3 + W3, yy), (X4, ty), STAT)
arrow((X4 + W4 / 2, 60), (X4 + W4 / 2, 55.6), STAT)

# ------------------------------------------------------------------ bottom legend strip
for x0, (lab, c) in zip((1, 21, 43, 64, 80, 101), [("Qdrant", QD), ("Weaviate", WV), ("pgvector", PG),
                                                     ("data", DATA), ("benchmark", BENCH), ("RAG", RAG)]):
    ax.add_patch(FancyBboxPatch((x0, 4.4), 3.2, 3.2, boxstyle="round,pad=0,rounding_size=0.5", fc=c, ec=c))
    ax.text(x0 + 4.4, 6.0, lab, ha="left", va="center", fontsize=12.5)

fig.savefig(OUT, bbox_inches="tight", pad_inches=0.04, facecolor="white")
print("saved", OUT)
