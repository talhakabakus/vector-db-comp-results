#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S5 — Statistical analysis, tables and figures for the paper (no database or GPU needed; runs in ~1-2 min).

Inputs (extracted result archives; defaults = <project>/results/{s2,s3,s4}, override with --s2/--s3/--s4):
  s2: S2 rows.jsonl (A1 + A2 + SMOKE rows, cumulative)          -> throughput / latency / resources
  s3: S3 rows.jsonl + q_*.npz (Q1 + Q2 + SMOKE rows)             -> E3 quality, E4 filtering, E5 hybrid
  s4: S4 rows.jsonl + s4_*.npz (SMOKE + FULL rows)               -> end-to-end RAG
Outputs (<project>/analysis/ or /kaggle/working/analysis/): tables/*.csv|md, figures/*.png (600 dpi), stats.json,
REPORT.md (+ analysis.zip on Kaggle). On Kaggle add the S2, S3 and S4 result datasets as inputs; they are found
automatically (paste the whole file into one cell).

Data cleaning (documented in the study plan, sections 3.6-3.11):
  S2  * rows without `preset` are A1, except fiqa nq=500 (SMOKE, dropped) and quora/bge-base (A2 attempt 1):
        there Qdrant/Weaviate rep 0 is valid A2 data, pgvector rep 0/1 ran the old parameterised-LIMIT code (dropped)
      * A2 pgvector in-process sweeps ran unforced (stale worker module) -> identical conditions to A1; the
        k=10 / ef=512 point is the planner's switch to an exact scan and is excluded from HNSW curves and
        iso-recall, reported separately ("planner cliff")
      * error rows dropped; `default` config analysed separately
  S3  * only Q1/Q2 rows; a Qdrant unit rerun with the lowered full_scan_threshold supersedes its earlier rows
      * if QNP rows exist (Qdrant rerun without payload indexes), they replace Qdrant's E3/E5 rows; E4 keeps Q2.
        The q_<dataset>_<model>_qdrant.npz files must then hold the QNP arrays (merge with 06_merge_qnp.py)
  S4  * FULL rows only, last row per (dataset, model, config); QNP rows replace the Qdrant configs of FULL
Statistics: iso-recall QPS = linear interpolation of log(QPS) in recall between adjacent ef points, per
repetition, then mean +- SD over the 3 repetitions; Friedman + Nemenyi critical difference across blocks;
paired Wilcoxon signed-rank with Pratt zero handling (per-query nDCG@10) and exact McNemar (per-query EM) with Holm correction per family;
paired bootstrap 95% CIs (10 000 resamples, seed 0); TOST equivalence on paired differences (margin 0.02);
logistic regression EM ~ hit@5 + system with query-clustered SEs ("does the system matter beyond recall?").
"""

import argparse, json, math, sys, warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore", category=RuntimeWarning)
try:                                   # script: <project>/scripts/05_...py ; notebook cell: no __file__
    ROOT = Path(__file__).resolve().parent.parent
except NameError:
    ROOT = Path.cwd()
KAGGLE = Path("/kaggle/working").exists()
SEED, B, DPI = 0, 10_000, 600
DBS = ["qdrant", "weaviate", "pgvector"]
NAME = {"qdrant": "Qdrant", "weaviate": "Weaviate", "pgvector": "pgvector"}
# Okabe–Ito palette, fixed order; systems take slots 1-3, markers are the secondary (CVD/print) encoding
OI = ["#0072B2", "#D55E00", "#E69F00", "#009E73", "#56B4E9", "#F0E442", "#CC79A7", "#000000"]
COL = {"qdrant": OI[0], "weaviate": OI[1], "pgvector": OI[2]}
MRK = {"qdrant": "o", "weaviate": "s", "pgvector": "^"}
INK, INK2, GRID, REF, SHADE = "#000000", "#4d4d4d", "#e6e6e6", "#4d4d4d", "#f1efe6"
DS_LABEL = {"fiqa": "FiQA", "hotpotqa_sub": "HotpotQA-sub", "quora": "Quora", "scifact": "SciFact",
            "nfcorpus": "NFCorpus", "trec-covid": "TREC-COVID", "squad": "SQuAD"}
STATS = {}


# ============================================================================ generic helpers
def read_rows(p):
    return pd.DataFrame([json.loads(l) for l in open(p)])


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj, run = np.empty_like(p), 0.0
    for i, j in enumerate(order):
        run = max(run, (len(p) - i) * p[j])
        adj[j] = min(run, 1.0)
    return adj


def mcnemar_exact(a, b):
    a, b = np.asarray(a) > 0.5, np.asarray(b) > 0.5
    n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
    n = n01 + n10
    p = 1.0 if n == 0 else min(1.0, 2 * stats.binom.cdf(min(n01, n10), n, 0.5))
    return p, n01, n10


def boot_ci(d, rng, level=0.95):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    m = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
    a = (1 - level) / 2
    return float(np.quantile(m, a)), float(np.quantile(m, 1 - a))


def tost_paired(d, margin):
    """Two one-sided paired t-tests; equivalence if p < 0.05 (i.e. the 90% CI lies inside +-margin)."""
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    n, m, s = len(d), d.mean(), d.std(ddof=1)
    if s == 0:
        return 0.0 if abs(m) < margin else 1.0
    se = s / math.sqrt(n)
    p_lo = 1 - stats.t.cdf((m + margin) / se, n - 1)
    p_hi = stats.t.cdf((m - margin) / se, n - 1)
    return float(max(p_lo, p_hi))


def nemenyi_cd(k, n, alpha=0.05):
    q = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728}[k]       # q_alpha (Demsar 2006, alpha = 0.05)
    return q * math.sqrt(k * (k + 1) / (6 * n))


def friedman(table):
    """table: rows = blocks, cols = systems (higher = better). Returns dict with mean ranks, p, CD."""
    t = table.dropna()
    if len(t) < 2:
        return None
    ranks = t.rank(axis=1, ascending=False).mean()
    chi, p = stats.friedmanchisquare(*[t[c].values for c in t.columns])
    return dict(n_blocks=len(t), mean_rank={c: round(float(r), 3) for c, r in ranks.items()},
                chi2=round(float(chi), 3), p=float(p), nemenyi_cd=round(nemenyi_cd(t.shape[1], len(t)), 3))


def save_table(df, name, out, floatfmt=None, index=False):
    df.to_csv(out / "tables" / f"{name}.csv", index=index)
    try:
        md = df.to_markdown(index=index, floatfmt=floatfmt or "g")
    except ImportError:
        md = df.to_string(index=index)
    (out / "tables" / f"{name}.md").write_text(md + "\n")
    return md


# ============================================================================ S2: throughput, latency, resources
def load_s2(d):
    df = read_rows(d / "rows.jsonl")
    df["preset"] = df["preset"].fillna("-")
    smoke = (df.preset == "SMOKE") | ((df.preset == "-") & (df.dataset == "fiqa") & (df.nq == 500))
    qb = (df.dataset == "quora") & (df.model == "bge-base")
    old_pg = (df.preset == "-") & qb & (df.db == "pgvector")
    df = df[~smoke & ~old_pg & (df.phase != "error")].copy()
    df["run"] = np.where((df.dataset == "quora") & (df.model == "bge-base"), "A2", "A1")
    df["unit"] = df.dataset.map(DS_LABEL) + "/" + df.model
    sw = df[df.phase == "sweep"].copy()
    sw["k"] = sw.k.astype(int)
    cliff = (sw.run == "A2") & (sw.db == "pgvector") & (sw.config == "common") & (sw.k == 10) & (sw.ef.astype(str) == "512")
    return df, sw[~cliff & (sw.config == "common")].copy(), sw[cliff].copy(), sw[sw.config == "default"].copy()


def curves(sw):
    """mean over passes per (unit, db, k, rep, ef) -> recall/qps/latency per ef."""
    sw = sw.assign(ef=sw.ef.astype(int))
    return sw.groupby(["unit", "dataset", "model", "db", "k", "rep", "ef"], as_index=False)[
        ["recall", "qps", "p50_ms", "p99_ms"]].mean()


def iso_at(c, target, col="qps"):
    c = c.sort_values("ef")
    r, v = c.recall.values, np.log(c[col].values)
    if target > r.max():
        return np.nan, "unreachable"
    if target < r.min():                   # already exceeded at the smallest ef: QPS there is a lower bound
        return float(np.exp(v[0])), "lower_bound"
    for i in range(len(r) - 1):
        if r[i] <= target <= r[i + 1] and r[i + 1] > r[i]:
            w = (target - r[i]) / (r[i + 1] - r[i])
            return float(np.exp(v[i] + w * (v[i + 1] - v[i]))), "ok"
    i = int(np.argmin(np.abs(r - target)))
    return float(np.exp(v[i])), "ok"


TARGETS = {10: [0.90, 0.95, 0.97, 0.98, 0.99], 100: [0.95, 0.99]}


def analyse_s2(d, out, rng):
    df, sw, cliff, dflt = load_s2(d)
    cv = curves(sw)
    rows = []
    for (unit, db, k, rep), c in cv.groupby(["unit", "db", "k", "rep"]):
        for t in TARGETS[k]:
            q, flag = iso_at(c, t)
            p50, _ = iso_at(c, t, "p50_ms")
            rows.append(dict(unit=unit, db=db, k=k, rep=rep, target=t, qps=q, p50_ms=p50, flag=flag))
    iso = pd.DataFrame(rows)
    agg = iso.groupby(["unit", "k", "target", "db"]).agg(qps=("qps", "mean"), sd=("qps", "std"), n=("qps", "count"),
                                                        p50_ms=("p50_ms", "mean"),
                                                        flag=("flag", lambda s: s.mode().iat[0])).reset_index()
    wide = agg.pivot_table(index=["unit", "k", "target"], columns="db", values="qps", dropna=False)
    sdw = agg.pivot_table(index=["unit", "k", "target"], columns="db", values="sd", dropna=False)
    flg = agg.pivot_table(index=["unit", "k", "target"], columns="db", values="flag", aggfunc="first", dropna=False)
    exact = agg.pivot_table(index=["unit", "k", "target"], columns="db", values="qps", dropna=False)
    exact[flg == "lower_bound"] = np.nan        # bounds are shown in the table but kept out of the rank tests
    exact = exact.reindex(wide.index)
    tab = []
    for idx in wide.index:
        r = dict(zip(["unit", "k", "target"], idx))
        vals = {db: wide.loc[idx].get(db, np.nan) for db in DBS}
        for db in DBS:
            v, s, f = vals[db], sdw.loc[idx].get(db, np.nan), flg.loc[idx].get(db, "")
            r[NAME[db]] = ("unreachable" if f == "unreachable" else "—") if not np.isfinite(v) else \
                (f"≥ {v:.0f}" if f == "lower_bound" else f"{v:.0f} ± {s:.0f}")
        fin = {db: v for db, v in vals.items() if np.isfinite(v)}
        r["best"] = (NAME[max(fin, key=fin.get)] + (" (only one reaching)" if len(fin) == 1 else "")) if fin else "none"
        bnd = [db for db in DBS if flg.loc[idx].get(db, "") == "lower_bound"]
        if bnd:   # a lower bound decides the winner only if it beats every other system's exact value
            top = max(fin, key=fin.get)
            others_exact = all(db not in bnd for db in fin if db != top)
            r["best"] = NAME[top] + " (bound)" if top in bnd and others_exact and len(fin) == 3 else "n/a (bounds only)"
        tab.append(r)
    tab = pd.DataFrame(tab)
    tab = tab[~(tab[[NAME[d] for d in DBS]] == "—").all(axis=1)]       # targets not in the grid for this k
    save_table(tab, "T_s2_iso_recall_qps", out)
    fr = friedman(exact[DBS])
    pw = []
    t = exact[DBS].dropna()
    for a, b in combinations(DBS, 2):
        w = stats.wilcoxon(t[a], t[b])
        pw.append(dict(a=a, b=b, median_ratio=float(np.median(t[a] / t[b])), p=float(w.pvalue)))
    for r, p in zip(pw, holm([r["p"] for r in pw])):
        r["p_holm"] = float(p)
    STATS["s2_iso_recall"] = dict(friedman=fr, pairwise_wilcoxon=pw,
                                  wins={NAME[db]: int((t.idxmax(axis=1) == db).sum()) for db in DBS},
                                  blocks=[f"{u}|k{k}|{tg}" for u, k, tg in t.index])
    # winner by recall regime (low/mid vs high) — the "it depends on the operating point" finding
    reg = t.reset_index()
    reg["regime"] = np.where((reg.k == 10) & (reg.target < 0.99), "k=10, recall<0.99", "recall=0.99 or k=100")
    STATS["s2_wins_by_regime"] = {g: {NAME[db]: int((x[DBS].idxmax(axis=1) == db).sum()) for db in DBS}
                                  for g, x in reg.groupby("regime")}

    # null query (protocol overhead), concurrency, resources, defaults, planner cliff
    nq = df[df.phase == "null_query"].groupby(["db"])[["p50_ms", "p95_ms"]].mean().round(3).reset_index()
    nq["db"] = nq.db.map(NAME)
    save_table(nq, "T_s2_null_query", out)
    STATS["s2_null_query_p50_ms"] = dict(zip(nq.db, nq.p50_ms))
    cc = df[df.phase == "concurrency"].groupby(["unit", "db", "recall_target", "clients"])[["qps", "p50_ms", "p99_ms"]].mean()
    cc = cc.reset_index()
    cct = cc.pivot_table(index=["unit", "recall_target", "clients"], columns="db", values="qps").round(0).reset_index()
    cct.columns = [NAME.get(c, c) for c in cct.columns]
    save_table(cct, "T_s2_concurrency_qps", out)
    ing = df[(df.phase == "ingest") & (df.config == "common")]
    mem = df[(df.phase == "post_search_memory") & (df.config == "common")]
    res = ing.groupby(["unit", "db"]).agg(ingest_s=("ingest_total_s", "mean"), ingest_sd=("ingest_total_s", "std"),
                                          build_s=("build_s", "mean"), pss_peak_mb=("pss_peak_ingest_mb", "mean"),
                                          disk_mb=("disk_mb", "mean"))
    res = res.join(mem.groupby(["unit", "db"]).pss_mb.mean().rename("pss_after_search_mb")).reset_index()
    res["db"] = res.db.map(NAME)
    save_table(res.round(0), "T_s2_resources", out)
    dsw = dflt.groupby(["unit", "db", "k"]).agg(recall=("recall", "mean"), qps=("qps", "mean"),
                                                 mean_returned=("mean_returned", "mean")).reset_index()
    dsw["db"] = dsw.db.map(NAME)
    save_table(dsw.round(3), "T_s2_default_config", out)
    cl = cliff.groupby(["unit", "rep"]).agg(recall=("recall", "mean"), qps=("qps", "mean"), p50_ms=("p50_ms", "mean")).reset_index()
    hn = cv[(cv.unit == "Quora/bge-base") & (cv.db == "pgvector") & (cv.k == 10)].groupby("ef")[["recall", "qps", "p50_ms"]].mean()
    STATS["s2_planner_cliff"] = dict(exact_scan_point=cl.mean(numeric_only=True).round(4).to_dict(),
                                     hnsw_ef256=hn.loc[256].round(4).to_dict() if 256 in hn.index else None)
    save_table(cl.round(4), "T_s2_planner_cliff", out)

    fig_s2_pareto(cv, cliff, out)
    fig_s2_iso(agg, out)
    fig_s2_concurrency(cc, out)
    fig_s2_resources(res, out)
    return tab


# ============================================================================ S3: quality, filtering, hybrid
def load_s3(d):
    df = read_rows(d / "rows.jsonl")
    qnp = df[df.preset == "QNP"].copy()                  # Qdrant rerun without payload indexes (E3, E5)
    df = df[df.preset.isin(["Q1", "Q2"])].copy()
    if "qdrant_fst_kb" in df:
        idx = df.set_index(["dataset", "model", "db"]).index
        rerun = idx[df.qdrant_fst_kb.notna().values].unique()
        df = df[~(idx.isin(rerun) & df.qdrant_fst_kb.isna().values)]
    if len(qnp):                                         # QNP replaces Qdrant's E3/E5 rows; E4 keeps the Q2 build
        done = qnp[qnp.exp == "load"].set_index(["dataset", "model"]).index.unique()   # units QNP completed
        unit = df.set_index(["dataset", "model"]).index.isin(done)
        df = df[~((df.db == "qdrant") & df.exp.isin(["E3", "E5", "load", "error"]) & unit)]
        qnp = qnp[qnp.set_index(["dataset", "model"]).index.isin(done) & (qnp.exp != "error")]
        STATS["s3_qnp_units"] = sorted(f"{d}/{m}" for d, m in done)
        df = pd.concat([df, qnp], ignore_index=True)
    return df


def analyse_s3(d, out, rng):
    df = load_s3(d)
    e3 = df[(df.exp == "E3") & (df.k == 10) & (df.db != "exact")].copy()
    e3["ef"] = e3.ef.astype(int)
    e3["unit"] = e3.dataset.map(DS_LABEL) + "/" + e3.model
    ex = df[(df.exp == "E3") & (df.db == "exact") & (df.k == 10)].set_index(["dataset", "model"]).ndcg10
    tab = e3.pivot_table(index=["unit", "ef"], columns="db", values=["recall_gt", "ndcg10"]).round(4)
    tab.columns = [f"{m}_{NAME[db]}" for m, db in tab.columns]
    save_table(tab.reset_index(), "T_s3_e3_quality", out)
    # Friedman over (dataset x model) blocks, per ef: recall_gt and nDCG@10
    STATS["s3_e3_friedman"] = {}
    for ef in sorted(e3.ef.unique()):
        for metric in ("recall_gt", "ndcg10"):
            t = e3[e3.ef == ef].pivot_table(index="unit", columns="db", values=metric)[DBS]
            STATS["s3_e3_friedman"][f"ef{ef}_{metric}"] = friedman(t)
    # per-query paired Wilcoxon on nDCG@10 (per dataset x model, ef in {16, 64, 128}), Holm within each ef
    pq = []
    for (ds, mdl), _ in e3.groupby(["dataset", "model"]):
        z = {db: np.load(d / f"q_{ds}_{mdl}_{db}.npz") for db in DBS}
        for ef in (16, 64, 128):
            for a, b in combinations(DBS, 2):
                x, y = z[a][f"E3_k10_ef{ef}_ndcg10"], z[b][f"E3_k10_ef{ef}_ndcg10"]
                diff = x - y
                p = stats.wilcoxon(x, y, zero_method="pratt").pvalue if np.any(diff != 0) else 1.0
                pq.append(dict(dataset=ds, model=mdl, ef=ef, a=NAME[a], b=NAME[b], mean_diff=float(diff.mean()),
                               p=float(p), tost_p_0p01=tost_paired(diff, 0.01)))
    pq = pd.DataFrame(pq)
    pq["p_holm"] = np.nan
    for ef, g in pq.groupby("ef"):
        pq.loc[g.index, "p_holm"] = holm(g.p.values)
    save_table(pq.round(5), "T_s3_e3_pairwise_ndcg", out)
    STATS["s3_e3_pairwise_ndcg"] = {f"ef{ef}": dict(n_tests=len(g), n_sig_holm=int((g.p_holm < 0.05).sum()),
                                                    n_equiv_0p01=int((g.tost_p_0p01 < 0.05).sum()),
                                                    max_abs_diff=float(g.mean_diff.abs().max()))
                                    for ef, g in pq.groupby("ef")}
    # ndcg gap to exact at each ef (how much quality ANN costs), averaged over units
    e3["gap"] = e3.apply(lambda r: r.ndcg10 - ex.get((r.dataset, r.model), np.nan), axis=1)
    STATS["s3_e3_ndcg_gap_to_exact"] = e3.groupby(["ef", "db"]).gap.mean().unstack().round(4).rename(columns=NAME).to_dict()

    # E4
    e4 = df[df.exp == "E4"].copy()
    e4["variant"] = e4.db.map(NAME) + e4.pg_iterative_scan.fillna("").map(lambda s: f" [{s}]" if s else "")
    t4 = e4.pivot_table(index=["dataset", "card", "ef"], columns="variant",
                        values=["recall_gt", "p50_ms", "mean_returned"]).round(3)
    t4.columns = [f"{m}|{v}" for m, v in t4.columns]
    plans = e4[e4.db == "pgvector"].pivot_table(index=["dataset", "card", "ef"], columns="pg_iterative_scan",
                                                 values="pg_plan", aggfunc="first")
    plans.columns = [f"pg_plan|{c}" for c in plans.columns]
    save_table(t4.join(plans).reset_index(), "T_s3_e4_filter", out)
    # E5
    e5 = df[df.exp == "E5"].copy()
    t5 = e5.pivot_table(index=["dataset", "method"], columns="db", values=["ndcg10", "p50_ms"]).round(4)
    t5.columns = [f"{m}|{NAME.get(db, db)}" for m, db in t5.columns]
    save_table(t5.reset_index(), "T_s3_e5_hybrid", out)
    hyb = []
    for ds in e5.dataset.unique():
        zx = np.load(d / f"q_{ds}_bge-base_exact.npz")
        dd = zx["E5_hybrid_ndcg10"] - zx["E5_dense_ndcg10"]
        lo, hi = boot_ci(dd, rng)
        hyb.append(dict(dataset=ds, hybrid_minus_dense=float(dd.mean()), ci_lo=lo, ci_hi=hi,
                        p=float(stats.wilcoxon(zx["E5_hybrid_ndcg10"], zx["E5_dense_ndcg10"], zero_method="pratt").pvalue)))
    hyb = pd.DataFrame(hyb)
    hyb["p_holm"] = holm(hyb.p.values)
    save_table(hyb.round(4), "T_s3_e5_hybrid_gain_exact", out)
    STATS["s3_e5_hybrid_gain_exact"] = hyb.round(4).to_dict("records")

    fig_s3_e3(e3, ex, out)
    fig_s3_e4(e4, out)
    fig_s3_e5(e5, out)


# ============================================================================ S4: end-to-end RAG
def load_s4(d):
    df = read_rows(d / "rows.jsonl")
    df = df[df.preset.isin(["FULL", "QNP"])]
    if (df.preset == "QNP").any():                       # QNP replaces the Qdrant configs of FULL, in their place
        full = df[df.preset == "FULL"].drop_duplicates(["dataset", "model", "config"], keep="last")
        pos = {k: i for i, k in enumerate(map(tuple, full[["dataset", "model", "config"]].values))}
        qnp = df[df.preset == "QNP"].drop_duplicates(["dataset", "model", "config"], keep="last")
        df = pd.concat([full[full.db != "qdrant"], qnp])
        df = df.assign(_o=[pos.get(k, len(pos)) for k in map(tuple, df[["dataset", "model", "config"]].values)])
        return df.sort_values("_o", kind="stable").drop(columns="_o").reset_index(drop=True)
    return df.drop_duplicates(["dataset", "model", "config"], keep="last").copy()


def analyse_s4(d, out, rng):
    df = load_s4(d)
    Z = {(ds, m): np.load(d / f"s4_{ds}_{m}.npz") for ds, m in df[["dataset", "model"]].drop_duplicates().values}
    get = lambda ds, m, cfg, met: Z[(ds, m)][f"{cfg.replace('/', '__')}__{met}"].astype(float)
    # main table with bootstrap CIs
    rows = []
    for _, r in df.iterrows():
        em, f1 = get(r.dataset, r.model, r.config, "em"), get(r.dataset, r.model, r.config, "f1")
        lo, hi = boot_ci(em, rng)
        flo, fhi = boot_ci(f1, rng)
        rows.append(dict(dataset=DS_LABEL[r.dataset], model=r.model, config=r.config, EM=em.mean(),
                         EM_ci=f"[{lo:.3f}, {hi:.3f}]", F1=f1.mean(), F1_ci=f"[{flo:.3f}, {fhi:.3f}]",
                         hit5=r.hit5, allgold5=r.allgold5, overlap5=r.overlap5, ans_in_ctx=r.ans_in_ctx))
    save_table(pd.DataFrame(rows).round(4), "T_s4_rag_main", out)

    # vs exact dense and between systems: McNemar (EM) + TOST (EM, F1; margin 0.02), Holm per family
    vs_ex, pair = [], []
    for (ds, m) in Z:
        ex_em, ex_f1 = get(ds, m, "exact/dense", "em"), get(ds, m, "exact/dense", "f1")
        for ef in (10, 16, 32, 64, 128, 256):
            for db in DBS:
                c = f"{db}/dense/ef{ef}"
                em, f1 = get(ds, m, c, "em"), get(ds, m, c, "f1")
                p, n01, n10 = mcnemar_exact(em, ex_em)
                lo, hi = boot_ci(em - ex_em, rng)
                vs_ex.append(dict(dataset=ds, model=m, ef=ef, db=NAME[db], dEM=float((em - ex_em).mean()), dEM_lo=lo,
                                  dEM_hi=hi, mcnemar_p=p, tost_em_p=tost_paired(em - ex_em, 0.02),
                                  tost_f1_p=tost_paired(f1 - ex_f1, 0.02)))
            for a, b in combinations(DBS, 2):
                ea, eb = get(ds, m, f"{a}/dense/ef{ef}", "em"), get(ds, m, f"{b}/dense/ef{ef}", "em")
                fa, fb = get(ds, m, f"{a}/dense/ef{ef}", "f1"), get(ds, m, f"{b}/dense/ef{ef}", "f1")
                p, n01, n10 = mcnemar_exact(ea, eb)
                lo, hi = boot_ci(ea - eb, rng)
                pair.append(dict(dataset=ds, model=m, ef=ef, a=NAME[a], b=NAME[b], dEM=float((ea - eb).mean()),
                                 dEM_lo=lo, dEM_hi=hi, discordant=f"{n01}/{n10}", mcnemar_p=p,
                                 tost_em_p=tost_paired(ea - eb, 0.02), tost_f1_p=tost_paired(fa - fb, 0.02)))
    vs_ex, pair = pd.DataFrame(vs_ex), pd.DataFrame(pair)
    for t in (vs_ex, pair):
        t["mcnemar_p_holm"] = np.nan
        for ef, g in t.groupby("ef"):
            t.loc[g.index, "mcnemar_p_holm"] = holm(g.mcnemar_p.values)
        t["equiv_0p02"] = (t.tost_em_p < 0.05) & (t.tost_f1_p < 0.05)
    save_table(vs_ex.round(5), "T_s4_vs_exact", out)
    save_table(pair.round(5), "T_s4_pairwise_systems", out)
    STATS["s4_vs_exact"] = {f"ef{ef}": dict(n=len(g), n_sig_holm=int((g.mcnemar_p_holm < 0.05).sum()),
                                            n_equiv=int(g.equiv_0p02.sum()), dEM_range=[round(float(g.dEM.min()), 4),
                                                                                        round(float(g.dEM.max()), 4)])
                            for ef, g in vs_ex.groupby("ef")}
    STATS["s4_pairwise"] = {f"ef{ef}": dict(n=len(g), n_sig_holm=int((g.mcnemar_p_holm < 0.05).sum()),
                                            n_equiv=int(g.equiv_0p02.sum()),
                                            max_abs_dEM=round(float(g.dEM.abs().max()), 4))
                            for ef, g in pair.groupby("ef")}

    # chain: config-level correlations + per-query logistic regression EM ~ hit5 + system (query-clustered SE)
    import statsmodels.formula.api as smf
    chain = {}
    for (ds, m) in Z:
        cf = [(db, ef) for db in DBS for ef in (10, 16, 32, 64, 128, 256)]
        agg = pd.DataFrame([dict(db=db, ef=ef, overlap5=np.nanmean(get(ds, m, f"{db}/dense/ef{ef}", "overlap5")),
                                 hit5=np.nanmean(get(ds, m, f"{db}/dense/ef{ef}", "hit5")),
                                 em=get(ds, m, f"{db}/dense/ef{ef}", "em").mean()) for db, ef in cf])
        long = pd.concat([pd.DataFrame(dict(q=np.arange(len(get(ds, m, f"{db}/dense/ef{ef}", "em"))),
                                            em=get(ds, m, f"{db}/dense/ef{ef}", "em"),
                                            hit=get(ds, m, f"{db}/dense/ef{ef}", "hit5"),
                                            ov=get(ds, m, f"{db}/dense/ef{ef}", "overlap5"), db=db))
                          for db, ef in cf], ignore_index=True)
        res = {}
        for name, formula in (("hit5", "em ~ hit + C(db, Treatment('qdrant'))"),
                              ("overlap5", "em ~ ov + C(db, Treatment('qdrant'))")):
            fit = smf.logit(formula, long).fit(disp=0, cov_type="cluster", cov_kwds={"groups": long.q})
            sysp = {k.split("T.")[-1].rstrip("]"): float(v) for k, v in fit.pvalues.items() if "C(db" in k}
            sysb = {k.split("T.")[-1].rstrip("]"): float(v) for k, v in fit.params.items() if "C(db" in k}
            res[name] = dict(system_coef=sysb, system_p=sysp, retrieval_coef=float(fit.params.iloc[1]),
                             retrieval_p=float(fit.pvalues.iloc[1]))
        chain[f"{ds}/{m}"] = dict(r_overlap_hit=float(np.corrcoef(agg.overlap5, agg.hit5)[0, 1]),
                                  r_hit_em=float(np.corrcoef(agg.hit5, agg.em)[0, 1]),
                                  slope_em_per_hit=float(np.polyfit(agg.hit5, agg.em, 1)[0]), logit=res)
    STATS["s4_chain"] = chain

    # effect sizes (paired, with bootstrap CIs) — what moves answer quality, and by how much
    eff = []
    def add(label, ds, m, a, b, m2=None):
        x, y = get(ds, m, a, "em"), get(ds, m2 or m, b, "em")
        lo, hi = boot_ci(x - y, rng)
        eff.append(dict(effect=label, dataset=DS_LABEL[ds], dEM=float((x - y).mean()), lo=lo, hi=hi,
                        p=mcnemar_exact(x, y)[0]))
    for ds in ("squad", "hotpotqa_sub"):
        add("Retrieval ceiling (oracle − exact dense)", ds, "bge-base", "gold/oracle", "exact/dense")
        add("Context vs none (exact dense − closed book)", ds, "bge-base", "exact/dense", "none/closed_book")
        add("Embedding model (bge-base − bge-small)", ds, "bge-base", "exact/dense", "exact/dense", m2="bge-small")
        add("Hybrid vs dense (exact)", ds, "bge-base", "exact/hybrid", "exact/dense")
        add("ANN at ef=10 (Weaviate − exact)", ds, "bge-base", "weaviate/dense/ef10", "exact/dense")
        g = pair[(pair.dataset == ds) & (pair.model == "bge-base") & (pair.ef == 128)]
        r = g.loc[g.dEM.abs().idxmax()]
        add(f"DB choice at ef=128 (largest pair: {r.a} − {r.b})", ds, "bge-base",
            f"{r.a.lower()}/dense/ef128", f"{r.b.lower()}/dense/ef128")
    eff = pd.DataFrame(eff)
    eff["p_holm"] = holm(eff.p.values)
    save_table(eff.round(4), "T_s4_effect_sizes", out)
    STATS["s4_effect_sizes"] = eff.round(4).to_dict("records")

    fig_s4_chain(df, out)
    fig_s4_effects(eff, out)


# ============================================================================ figures
def style():
    plt.rcParams.update({
        "figure.dpi": 100, "savefig.dpi": DPI, "font.size": 12.5, "axes.titlesize": 13.5, "axes.labelsize": 13,
        "xtick.labelsize": 11.5, "ytick.labelsize": 11.5, "legend.fontsize": 12, "axes.titleweight": "bold",
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "lines.linewidth": 2.2, "lines.markersize": 7.5, "font.family": "DejaVu Sans",
        "axes.titlepad": 8, "axes.facecolor": "white"})


def savefig(fig, name, out):
    fig.savefig(out / "figures" / f"{name}.png", dpi=DPI, bbox_inches="tight", pad_inches=0.03, facecolor="white")
    plt.close(fig)


def sys_legend(fig, extra=(), ncol=None, y=1.0):
    h = [Line2D([], [], color=COL[d], marker=MRK[d], lw=2.2, ms=8, label=NAME[d]) for d in DBS] + list(extra)
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, y), ncol=ncol or len(h), handlelength=2.2,
               columnspacing=1.6)


def label(ax, x, y, text, color=INK, dx=0, dy=7, size=9.5, ha="center", va="bottom", weight="normal", box=False):
    kw = dict(bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.85)) if box else {}
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, fontsize=size,
                color=color, fontweight=weight, **kw)


def fmt_k(v):
    return f"{v/1000:.1f}k" if v >= 1000 else f"{v:.0f}"


UNITS = ["FiQA/bge-base", "HotpotQA-sub/bge-base", "Quora/bge-small", "Quora/bge-base"]


def fig_s2_pareto(cv, cliff, out):
    m = cv.groupby(["unit", "db", "k", "ef"], as_index=False)[["recall", "qps"]].mean()
    fig, axes = plt.subplots(2, 4, figsize=(19, 9), constrained_layout=True)
    for j, u in enumerate(UNITS):
        for i, k in enumerate((10, 100)):
            ax = axes[i, j]
            ax.axvspan(0.99, 1.0, color=SHADE, zorder=0)
            ends = []
            for db in DBS:
                c = m[(m.unit == u) & (m.db == db) & (m.k == k)].sort_values("ef")
                ax.plot(c.recall, c.qps, color=COL[db], marker=MRK[db], ms=6.5, lw=2.2, mec="white", mew=0.8)
                lo = c.iloc[0]                     # fastest point labelled on the curve, most accurate in the box
                label(ax, lo.recall, lo.qps, fmt_k(lo.qps), COL[db], dy=-17 if db == "weaviate" else 6, size=10,
                      weight="bold")
                ends.append((db, c.iloc[-1]))
            ax.text(0.03, 0.035, "max recall (QPS)", transform=ax.transAxes, fontsize=9.5, color=INK2)
            for r, (db, hi) in enumerate(ends):
                ax.text(0.03, 0.10 + 0.075 * (2 - r), f"{NAME[db]}: {hi.recall:.3f} ({hi.qps:.0f})",
                        transform=ax.transAxes, fontsize=10, color=COL[db], fontweight="bold")
            if k == 10 and u == "Quora/bge-base" and len(cliff):
                ax.text(0.03, 0.36, f"pgvector ef=512: planner switches\nto an exact scan — {cliff.qps.mean():.1f} QPS,\n"
                        f"p50 {cliff.p50_ms.mean()/1000:.2f} s (excluded)", transform=ax.transAxes, fontsize=10,
                        color=OI[6], va="bottom", fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=OI[6], lw=1.2))
            ax.text(0.995, 0.97, "≥0.99", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=9.5,
                    color=INK2) if ax.get_xlim()[1] > 0.99 else None
            ax.set_yscale("log")
            ax.set_title(f"{u}  ·  k={k}")
            ax.set_xlabel(f"recall@{k}")
            if j == 0:
                ax.set_ylabel("throughput (QPS, 1 client, log)")
    sys_legend(fig, [plt.Rectangle((0, 0), 1, 1, color=SHADE, label="high-recall band (≥ 0.99)")])
    savefig(fig, "F_s2_recall_qps_pareto", out)


def fig_s2_iso(agg, out):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.6), constrained_layout=True)
    w = 0.27
    for ax, u in zip(axes, UNITS):
        top = 0
        for i, db in enumerate(DBS):
            c = agg[(agg.unit == u) & (agg.db == db) & (agg.k == 10)].set_index("target").reindex(TARGETS[10])
            for xi, t in enumerate(TARGETS[10]):
                x = xi + (i - 1) * (w + 0.02)
                v, s, f = c.loc[t, "qps"], c.loc[t, "sd"], c.loc[t, "flag"]
                if f == "unreachable" or not np.isfinite(v):
                    ax.text(x, 12, "n/r", ha="center", va="bottom", fontsize=9, color=COL[db], rotation=90)
                    continue
                bound = f == "lower_bound"
                ax.bar(x, v, width=w, color="white" if bound else COL[db], edgecolor=COL[db], lw=1.6,
                       hatch="///" if bound else None, zorder=2)
                if not bound and np.isfinite(s):
                    ax.errorbar(x, v, yerr=s, color=INK, capsize=2.5, lw=1.1, zorder=3)
                ax.text(x, v + (s if np.isfinite(s) and not bound else 0) + 12, ("≥" if bound else "") + f"{v:.0f}",
                        ha="center", va="bottom", fontsize=9.5, rotation=90, color=INK)
                top = max(top, v + (s if np.isfinite(s) else 0))
        ax.set_ylim(0, top * 1.28)
        ax.set_xticks(range(len(TARGETS[10])))
        ax.set_xticklabels([f"{t:.2f}" for t in TARGETS[10]])
        ax.set_xlabel("target recall@10")
        ax.set_title(u)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("iso-recall throughput (QPS, mean ± SD)")
    h = [plt.Rectangle((0, 0), 1, 1, color=COL[d], label=NAME[d]) for d in DBS] + \
        [plt.Rectangle((0, 0), 1, 1, fc="white", ec=INK2, hatch="///", label="lower bound (target passed at smallest ef)"),
         Line2D([], [], color="none", label="n/r = target not reached (ef ≤ 512)")]
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=5)
    savefig(fig, "F_s2_iso_recall_qps", out)


def fig_s2_concurrency(cc, out):
    fig, axes = plt.subplots(2, 4, figsize=(19, 9), constrained_layout=True)
    for j, u in enumerate(UNITS):
        for i, t in enumerate((0.95, 0.99)):
            ax = axes[i, j]
            ax.axvspan(4, 9.5, color=SHADE, zorder=0)
            ymax, peaks, firsts = 0, [], []
            for db in DBS:
                c = cc[(cc.unit == u) & (cc.db == db) & (cc.recall_target == t)].sort_values("clients")
                ax.plot(c.clients, c.qps, color=COL[db], marker=MRK[db], mec="white", mew=0.8)
                peaks.append((db, c.loc[c.qps.idxmax()]))
                firsts.append((db, c.iloc[0].qps))
                ymax = max(ymax, c.qps.max())
            placed = []                       # peak labels: push one below its point if it would hit another
            for db, pk in sorted(peaks, key=lambda z: -z[1].qps):
                clash = any(x == pk.clients and abs(v - pk.qps) < 0.07 * ymax for x, v in placed)
                label(ax, pk.clients, pk.qps, fmt_k(pk.qps), COL[db], dy=-20 if clash else 7, size=10.5,
                      weight="bold")
                placed.append((pk.clients, pk.qps))
            ax.text(0.97, 0.035, "1 client (QPS)", transform=ax.transAxes, ha="right", fontsize=9.5, color=INK2)
            for r, (db, v) in enumerate(firsts):
                ax.text(0.97, 0.10 + 0.075 * (2 - r), f"{NAME[db]}: {v:.0f}", transform=ax.transAxes, ha="right",
                        fontsize=10, color=COL[db], fontweight="bold")
            ax.set_xscale("log", base=2)
            ax.set_xticks([1, 2, 4, 8])
            ax.set_xticklabels(["1", "2", "4", "8"])
            ax.set_xlim(0.7, 9.5)
            ax.set_ylim(0, ymax * 1.18)
            ax.set_title(f"{u}  ·  recall@10 = {t}")
            ax.set_xlabel("concurrent clients")
            if j == 0:
                ax.set_ylabel("throughput (QPS)")
            if u.startswith("HotpotQA") and t == 0.99:
                ax.text(0.03, 0.97, "0.99 not reached:\nrun at ef=512", transform=ax.transAxes, fontsize=10,
                        color=INK2, va="top")
    sys_legend(fig, [plt.Rectangle((0, 0), 1, 1, color=SHADE, label="more clients than vCPUs (4)")])
    savefig(fig, "F_s2_concurrency", out)


def fig_s2_resources(res, out):
    metrics = [("ingest_s", "Ingest + index build (s)", 1, "{:.0f}"),
               ("pss_peak_mb", "Peak memory, ingest (GB)", 1024, "{:.1f}"),
               ("pss_after_search_mb", "Memory after search (GB)", 1024, "{:.1f}"),
               ("disk_mb", "Disk footprint (GB)", 1024, "{:.2f}")]
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.6), constrained_layout=True)
    x = np.arange(len(UNITS))
    w = 0.27
    for ax, (mcol, lab, div, f) in zip(axes, metrics):
        top = 0
        for i, db in enumerate(DBS):
            v = np.array([res[(res.unit == u) & (res.db == NAME[db])][mcol].mean() for u in UNITS]) / div
            xs = x + (i - 1) * (w + 0.02)
            ax.bar(xs, v, width=w, color=COL[db], zorder=2)
            for xx, vv in zip(xs, v):
                ax.text(xx, vv * 1.02 + 0.01, f.format(vv), ha="center", va="bottom", fontsize=9.5, rotation=90)
            top = max(top, np.nanmax(v))
        ax.set_ylim(0, top * 1.25)
        ax.set_xticks(x)
        ax.set_xticklabels([u.replace("/", "\n") for u in UNITS], fontsize=10.5)
        ax.set_title(lab)
        ax.grid(axis="x", visible=False)
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=COL[d], label=NAME[d]) for d in DBS],
               loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    savefig(fig, "F_s2_resources", out)


def fig_s3_e3(e3, ex, out):
    units = sorted(e3.unit.unique(), key=lambda u: (u.split("/")[1] != "bge-base", u))
    cols, n = 7, len(units)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(24, 4.6 * rows), constrained_layout=True, sharey=True)
    for ax, u in zip(axes.flat, units):
        ax.axhspan(0.95, 1.0, color=SHADE, zorder=0)
        lines = []
        for db in DBS:
            c = e3[(e3.unit == u) & (e3.db == db)].sort_values("ef")
            ax.plot(c.ef, c.recall_gt, color=COL[db], marker=MRK[db], ms=6, mec="white", mew=0.7)
            lines.append((db, c.iloc[0].recall_gt))
        for r, (db, v) in enumerate(sorted(lines, key=lambda z: -z[1])):     # ef=10 values, best first
            ax.text(0.96, 0.05 + 0.1 * r, f"{v:.3f}", transform=ax.transAxes, ha="right", fontsize=10.5,
                    color=COL[db], fontweight="bold")
        ax.text(0.96, 0.36, "ef=10:", transform=ax.transAxes, ha="right", fontsize=9.5, color=INK2)
        ax.set_xscale("log", base=2)
        ax.set_xticks([16, 64, 256])
        ax.set_xticklabels(["16", "64", "256"])
        ax.set_title(u.replace("/", "\n"), fontsize=12.5)
        ax.set_ylim(0.66, 1.005)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    for r in range(rows):
        axes[r, 0].set_ylabel("recall@10 vs exact")
    for ax in axes[-1]:
        ax.set_xlabel("ef (search)")
    sys_legend(fig, [plt.Rectangle((0, 0), 1, 1, color=SHADE, label="recall ≥ 0.95")])
    savefig(fig, "F_s3_e3_recall_vs_ef", out)


def fig_s3_e4(e4, out):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    ef = 64
    var = [("qdrant", None, "-"), ("weaviate", None, "-"), ("pgvector", "off", "-"), ("pgvector", "relaxed_order", "--")]
    placed = {}
    for j, ds in enumerate(["hotpotqa_sub", "quora"]):
        for i, (metric, lab) in enumerate((("recall_gt", "filtered recall@10"), ("p50_ms", "p50 latency (ms, log)"))):
            ax = axes[i, j]
            if metric == "recall_gt":
                ax.axhspan(0, 0.9, color="#fbe9df", zorder=0)
                ax.text(0.02, 0.03, "recall < 0.90", transform=ax.transAxes, fontsize=10, color=OI[1])
            for db, it, ls in var:
                c = e4[(e4.dataset == ds) & (e4.db == db) & (e4.ef == ef)]
                if it:
                    c = c[c.pg_iterative_scan == it]
                c = c.sort_values("card")
                ax.plot(c.card, c[metric], color=COL[db], marker=MRK[db], ls=ls, mec="white", mew=0.7)
                if db == "pgvector":           # planner left HNSW for a B-tree scan: exact, not HNSW recall
                    bt = c[c.pg_plan == "btree"]
                    ax.scatter(bt.card, bt[metric], marker=MRK[db], s=90, facecolor="white", edgecolor=COL[db],
                               linewidth=1.8, zorder=4)
                for _, r in c.iterrows():   # label what deviates: lost recall, or slow queries
                    if metric == "recall_gt" and r.recall_gt < 0.93:
                        if any(x == r.card and abs(v - r.recall_gt) < 0.035 for x, v in placed.get(id(ax), [])):
                            continue        # same value already labelled at this point
                        label(ax, r.card, r.recall_gt, f"{r.recall_gt:.2f}", COL[db], dy=-18 if it == "off" else 8,
                              size=10.5, weight="bold", box=True)
                        placed.setdefault(id(ax), []).append((r.card, r.recall_gt))
                    if metric == "p50_ms" and r.p50_ms > 7:
                        label(ax, r.card, r.p50_ms, f"{r.p50_ms:.0f} ms", COL[db], dy=8, size=10.5, weight="bold",
                              box=True)
            ax.set_xscale("log")
            ax.set_xticks([2, 10, 100, 1000])
            ax.set_xticklabels(["1/2", "1/10", "1/100", "1/1000"])
            ax.set_xlabel("fraction of corpus passing the filter")
            ax.set_ylabel(lab)
            ax.set_title(f"{DS_LABEL[ds]}  ·  ef={ef}")
            if metric == "p50_ms":
                ax.set_yscale("log")
            else:
                ax.set_ylim(0, 1.06)
    h = [Line2D([], [], color=COL["qdrant"], marker="o", label="Qdrant"),
         Line2D([], [], color=COL["weaviate"], marker="s", label="Weaviate (ACORN)"),
         Line2D([], [], color=COL["pgvector"], marker="^", label="pgvector, iterative_scan=off"),
         Line2D([], [], color=COL["pgvector"], ls="--", marker="^", label="pgvector, relaxed_order"),
         Line2D([], [], color=COL["pgvector"], ls="", marker="^", mfc="white", ms=9, mew=1.8,
                label="pgvector planner chose B-tree (exact)")]
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    savefig(fig, "F_s3_e4_filtered_search", out)


def fig_s3_e5(e5, out):
    dsets = ["scifact", "nfcorpus", "fiqa", "trec-covid", "squad", "hotpotqa_sub"]
    fig, axes = plt.subplots(1, 2, figsize=(20, 6.2), constrained_layout=True)
    mk = {"dense": "o", "lexical": "D", "hybrid": "*"}
    sz = {"dense": 70, "lexical": 60, "hybrid": 190}
    x = np.arange(len(dsets))
    off = {"qdrant": -0.24, "weaviate": 0.0, "pgvector": 0.24}
    for k, ds in enumerate(dsets):
        if k % 2 == 0:
            for ax in axes:
                ax.axvspan(k - 0.5, k + 0.5, color="#f6f6f4", zorder=0)
    for db in DBS:
        for meth in ("dense", "lexical", "hybrid"):
            g = e5[(e5.db == db) & (e5.method == meth)].set_index("dataset").reindex(dsets)
            axes[0].scatter(x + off[db], g.ndcg10, color=COL[db], marker=mk[meth], s=sz[meth], edgecolor="white",
                            linewidth=0.8, zorder=3)
            axes[1].scatter(x + off[db], g.p50_ms, color=COL[db], marker=mk[meth], s=sz[meth], edgecolor="white",
                            linewidth=0.8, zorder=3)
        lx = e5[(e5.db == db) & (e5.method == "lexical")].set_index("dataset").reindex(dsets).p50_ms
        hy = e5[(e5.db == db) & (e5.method == "hybrid")].set_index("dataset").reindex(dsets).p50_ms
        for xi, a, b in zip(x, lx, hy):       # one combined label per system/dataset for slow lexical/hybrid
            if max(a, b) > 20:
                label(axes[1], xi + off[db], max(a, b), f"lex {a:.0f}\nhyb {b:.0f} ms", COL[db], dx=-12, dy=-2,
                      ha="right", va="center", size=10, weight="bold", box=True)
    for k, ds in enumerate(dsets):
        dv = e5[(e5.dataset == ds) & (e5.db == "exact") & (e5.method == "dense")].ndcg10.mean()
        hv = e5[(e5.dataset == ds) & (e5.db == "exact") & (e5.method == "hybrid")].ndcg10.mean()
        axes[0].hlines(dv, k - 0.42, k + 0.42, color=INK, lw=1.3, ls=":")
        axes[0].hlines(hv, k - 0.42, k + 0.42, color=INK, lw=1.3)
        d = hv - dv
        axes[0].text(k, 0.17, f"hybrid − dense\n{d:+.3f}", ha="center", fontsize=10.5, fontweight="bold",
                     color=OI[3] if d > 0 else INK2)       # vermillion is Weaviate's, so not used here
    axes[0].set_ylim(0.15, 0.93)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([DS_LABEL[d] for d in dsets])
        ax.grid(axis="x", visible=False)
        ax.set_xlim(-0.5, len(dsets) - 0.5)
    axes[0].set_ylabel("nDCG@10")
    axes[0].set_title("Retrieval quality  (exact reference: ··· dense,  — hybrid)")
    axes[1].set_ylabel("p50 latency (ms, log)")
    axes[1].set_yscale("log")
    axes[1].set_title("Query latency  (labels: lexical/hybrid > 20 ms)")
    h = [Line2D([], [], color=COL[d], marker="o", ls="", ms=9, label=NAME[d]) for d in DBS] + \
        [Line2D([], [], color=INK2, marker=mk[m], ls="", label=m, ms=13 if m == "hybrid" else 8) for m in mk]
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=6)
    savefig(fig, "F_s3_e5_hybrid", out)


def fig_s4_chain(df, out):
    fig, axes = plt.subplots(1, 4, figsize=(21, 6.2), constrained_layout=True)
    panels = [("squad", "bge-base"), ("squad", "bge-small"), ("hotpotqa_sub", "bge-base"), ("hotpotqa_sub", "bge-small")]
    for ax, (ds, m) in zip(axes, panels):
        g = df[(df.dataset == ds) & (df.model == m)]
        ob = df[(df.dataset == ds) & (df.config == "gold/oracle")].em.iat[0]
        cb = df[(df.dataset == ds) & (df.config == "none/closed_book")].em.iat[0]
        ax.axhspan(cb, ob, color=SHADE, zorder=0)
        ax.axhline(ob, color=INK, ls="--", lw=1.4)
        ax.axhline(cb, color=INK, ls=":", lw=1.6)
        dense = g[g.method == "dense"].copy()
        dense = dense[dense.db.isin(DBS)]
        dense["efn"] = dense.ef.astype(float)
        ax.text(0.97, 0.335, "EM at ef = 10", transform=ax.transAxes, ha="right", fontsize=10, color=INK2)
        for k, db in enumerate(DBS):
            c = dense[dense.db == db].sort_values("efn")
            ax.plot(c.hit5, c.em, color=COL[db], marker=MRK[db], lw=2, ms=8, mec="white", mew=0.8)
            ax.text(0.97, 0.29 - 0.045 * k, f"{NAME[db]}: {c.iloc[0].em:.3f}", transform=ax.transAxes, ha="right",
                    fontsize=10.5, color=COL[db], fontweight="bold")
        e = g[g.config == "exact/dense"].iloc[0]
        ax.scatter(e.hit5, e.em, marker="*", s=380, color=INK, zorder=5, edgecolor="white", linewidth=0.8)
        label(ax, e.hit5, e.em, f"exact {e.em:.3f}", INK, dx=-6, dy=12, ha="right", size=10.5, weight="bold", box=True)
        x0 = dense.hit5.min()
        ax.text(x0, ob, f" oracle {ob:.3f}", va="bottom", fontsize=10.5, color=INK, fontweight="bold")
        ax.text(x0, cb, f" closed book {cb:.3f}", va="bottom", fontsize=10.5, color=INK, fontweight="bold")
        r = np.corrcoef(dense.hit5, dense.em)[0, 1]
        sl = np.polyfit(dense.hit5, dense.em, 1)[0]
        ax.text(0.03, 0.40 if ds == "squad" else 0.255, f"r(hit@5, EM) = {r:.3f}\nslope = {sl:.2f}",
                transform=ax.transAxes, ha="left", fontsize=11, color=INK,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=GRID))
        lo10 = dense[dense.efn == 10]
        ax.annotate("ef = 10", (lo10.hit5.min(), lo10.em.min()), xytext=(2, -22), textcoords="offset points",
                    fontsize=10, color=INK2, ha="left")
        ax.set_title(f"{DS_LABEL[ds]}  ·  {m}")
        ax.set_xlabel("gold passage in top-5 (hit@5)")
        ax.set_ylim(0.0, 0.9)
    axes[0].set_ylabel("exact match (Qwen2.5-7B, top-5 context)")
    extra = [Line2D([], [], color=INK, marker="*", ls="", ms=15, label="exact search"),
             Line2D([], [], color=INK, ls="--", label="oracle passages"),
             Line2D([], [], color=INK, ls=":", label="closed book"),
             plt.Rectangle((0, 0), 1, 1, color=SHADE, label="headroom retrieval can buy")]
    sys_legend(fig, extra, ncol=7)
    savefig(fig, "F_s4_recall_to_answer_chain", out)


def fig_s4_effects(eff, out):
    eff = eff.copy()
    eff["lab"] = eff.effect.str.replace(r" \(largest pair.*\)", " (largest pair)", regex=True)
    labels = list(dict.fromkeys(eff.lab))
    fig, ax = plt.subplots(figsize=(13, 7), constrained_layout=True)
    ax.axvspan(-2, 2, color="#e3f1f8", zorder=0)
    ax.text(0, -0.62, "±2 pp equivalence margin", ha="center", va="center", fontsize=10.5, color=OI[0],
            fontweight="bold", bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=OI[4]))
    ax.set_ylim(-0.95, len(labels) - 0.45)
    y = np.arange(len(labels))[::-1]
    mk = {"SQuAD": ("o", OI[3]), "HotpotQA-sub": ("s", OI[4])}
    for k, (dsl, (mrk, col)) in enumerate(mk.items()):
        g = eff[eff.dataset == dsl].set_index("lab").reindex(labels)
        yy = y + (0.17 if k == 0 else -0.17)
        ax.errorbar(g.dEM * 100, yy, xerr=[(g.dEM - g.lo) * 100, (g.hi - g.dEM) * 100], fmt=mrk, color=col,
                    capsize=4, ms=10, lw=2, mec="white", label=dsl, zorder=3)
        for xv, hi, yv, p in zip(g.dEM * 100, g.hi * 100, yy, g.p_holm):
            star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            ax.text(hi + 1.0, yv, f"{xv:+.1f} pp  {star}", va="center", fontsize=11, color=INK, fontweight="bold")
    ax.axvline(0, color=INK2, lw=1.1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlabel("Δ exact match, percentage points (paired bootstrap 95% CI; Holm-adjusted McNemar)")
    ax.set_xlim(eff.lo.min() * 100 - 3, eff.hi.max() * 100 + 14)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", fontsize=12, title="dataset", title_fontsize=12)
    savefig(fig, "F_s4_effect_sizes", out)


# ============================================================================ report
def write_report(out):
    S = STATS
    L = ["# S5 analysis report (auto-generated)", ""]
    fr = S["s2_iso_recall"]["friedman"]
    L += ["## S2 — iso-recall throughput",
          f"- Friedman over {fr['n_blocks']} blocks (unit × k × target): χ² = {fr['chi2']}, p = {fr['p']:.2g}; "
          f"mean ranks {fr['mean_rank']} (Nemenyi CD = {fr['nemenyi_cd']}).",
          f"- Wins per block: {S['s2_iso_recall']['wins']}; by regime: {S['s2_wins_by_regime']}.",
          f"- Null-query p50 (ms): {S['s2_null_query_p50_ms']}.",
          f"- Planner cliff (quora/bge-base, pgvector k=10 ef=512): {S['s2_planner_cliff']}.", ""]
    L += ["## S3 — retrieval quality", "- Friedman (dataset×model blocks) per ef, recall_gt / nDCG@10:"]
    for k, v in S["s3_e3_friedman"].items():
        if v:
            L.append(f"  - {k}: p = {v['p']:.3g}, mean ranks {v['mean_rank']}")
    L += [f"- Per-query nDCG@10 Wilcoxon (Holm per ef): {S['s3_e3_pairwise_ndcg']}",
          f"- Mean nDCG@10 gap to exact by ef: {S['s3_e3_ndcg_gap_to_exact']}", ""]
    L += ["## S4 — end-to-end RAG",
          f"- vs exact dense: {S['s4_vs_exact']}", f"- system pairs: {S['s4_pairwise']}"]
    for k, v in S["s4_chain"].items():
        L.append(f"- {k}: r(overlap,hit) = {v['r_overlap_hit']:.3f}, r(hit,EM) = {v['r_hit_em']:.3f}, "
                 f"slope = {v['slope_em_per_hit']:.2f}; logit system p (given hit@5) = "
                 f"{ {s: round(p, 3) for s, p in v['logit']['hit5']['system_p'].items()} }, "
                 f"(given overlap@5) = { {s: round(p, 3) for s, p in v['logit']['overlap5']['system_p'].items()} }")
    L += ["", "## Effect sizes (EM, bge-base)"]
    for r in S["s4_effect_sizes"]:
        L.append(f"- {r['dataset']}: {r['effect']}: {r['dEM']*100:+.1f} pp [{r['lo']*100:+.1f}, {r['hi']*100:+.1f}], "
                 f"p_holm = {r['p_holm']:.2g}")
    (out / "REPORT.md").write_text("\n".join(L) + "\n")


SEARCH_ROOTS = [Path("/kaggle/input"), Path("/kaggle/working")]
WORK = Path("/kaggle/working") if KAGGLE else Path.cwd()


def find_input(local, marker, what):
    """Folder holding `marker`: <project>/results/<x>, else anywhere under /kaggle/input or /kaggle/working,
    else inside a .zip there (extracted to /kaggle/working/_inputs/<what>)."""
    if (local / marker).exists():
        return local
    roots = [r for r in SEARCH_ROOTS if r.exists()]
    for r in roots:
        hits = sorted(r.glob(f"**/{marker}"))
        if hits:
            return hits[0].parent
    import zipfile
    for r in roots:
        for z in sorted(r.glob("**/*.zip")):
            try:
                with zipfile.ZipFile(z) as zf:
                    names = [n for n in zf.namelist() if n.endswith(marker)]
                    if names:
                        dest = WORK / "_inputs" / what
                        zf.extractall(dest)
                        print(f"   extracted {z} -> {dest}", flush=True)
                        return (dest / names[0]).parent
            except zipfile.BadZipFile:
                continue
    seen = []
    for r in roots:
        seen += [str(p) for p in sorted(r.glob("*/*"))[:40]]
    raise FileNotFoundError(
        f"{what} results not found: no '{marker}' in {local}, under {', '.join(map(str, roots))}, or in a .zip there.\n"
        f"Add the {what} result archive as a Kaggle input (Add Input -> your dataset).\n"
        f"Currently visible:\n  " + "\n  ".join(seen or ["(nothing)"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s2", default=None)
    ap.add_argument("--s3", default=None)
    ap.add_argument("--s4", default=None)
    ap.add_argument("--out", default=str(Path("/kaggle/working/analysis") if KAGGLE else ROOT / "analysis"))
    a, _ = ap.parse_known_args()          # Jupyter passes its own -f <kernel.json>
    a.s2 = a.s2 or find_input(ROOT / "results" / "s2", "summary_sweep.csv", "S2")
    a.s3 = a.s3 or find_input(ROOT / "results" / "s3", "q_squad_bge-base_exact.npz", "S3")
    a.s4 = a.s4 or find_input(ROOT / "results" / "s4", "s4_squad_bge-base.npz", "S4")
    print(f">> inputs: S2={a.s2}\n           S3={a.s3}\n           S4={a.s4}", flush=True)
    out = Path(a.out)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    style()
    rng = np.random.default_rng(SEED)
    print(">> S2 ...", flush=True)
    analyse_s2(Path(a.s2), out, rng)
    print(">> S3 ...", flush=True)
    analyse_s3(Path(a.s3), out, rng)
    print(">> S4 ...", flush=True)
    analyse_s4(Path(a.s4), out, rng)
    (out / "stats.json").write_text(json.dumps(STATS, indent=2, default=float))
    write_report(out)
    for old in (out / "figures").glob("*.pdf"):          # PNG only (older runs wrote PDFs too)
        old.unlink()
    if KAGGLE:
        import shutil
        shutil.make_archive(str(out.parent / "analysis"), "zip", out)
        print(f">> archive: {out.parent / 'analysis.zip'}", flush=True)
    print(f">> done: {out}", flush=True)


if __name__ == "__main__":
    main()
