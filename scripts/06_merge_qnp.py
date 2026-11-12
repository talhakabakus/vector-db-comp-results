#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
06 — Merge the Qdrant rerun without payload indexes (S3/S4 preset "QNP") into the original S3/S4 results.

Runs locally in seconds (no database, no GPU). Nothing in the input folders is modified.

  python 06_merge_qnp.py --s3-old results/s3 --s3-new <s3_results QNP> <s3_results_qnpfix> \
                         --s4-old results/s4 --s4-new <s4_results QNP> --out results_qnp

Only rows with preset QNP/QNPFIX are taken from the new folders (a Kaggle working folder may still hold the older
Q1/Q2 files); QNPFIX rows are relabelled QNP. A Qdrant npz is taken from a new folder only if that folder has a
QNP/QNPFIX load row for the unit, so stale npz files of units that failed in QNP are ignored.

Output: <out>/s3 and <out>/s4, ready for 05_analysis_figures.py --s3 <out>/s3 --s4 <out>/s4.
  s3: all original files; rows.jsonl = original rows + QNP rows (S5 lets QNP replace Qdrant's E3/E5 rows);
      q_<ds>_<model>_qdrant.npz = QNP arrays (E3_*, E5_*) + the original E4_* arrays (E4 needs payload indexes
      and is not rerun); the QNP env/log/marker files are added with a "qnp_" prefix.
  s4: all original files; rows.jsonl = original rows + QNP rows (S5 lets QNP replace the Qdrant configs);
      s4_<ds>_<model>.npz = original arrays without qdrant__* + the QNP qdrant__* arrays;
      the QNP generation caches/logs are added with a "qnp_" prefix.
"""
import argparse, json, shutil
from pathlib import Path

import numpy as np


def copy_tree(src, dst):
    shutil.copytree(src, dst, dirs_exist_ok=True)


NEW = ("QNP", "QNPFIX")


def new_rows(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    rows = [r for r in rows if r.get("preset") in NEW]
    for r in rows:
        r["preset"] = "QNP"
    return rows


def append_rows(old, news, dst):
    lines = [l.rstrip("\n") for l in open(old) if l.strip()]
    for l in lines:
        json.loads(l)                                     # fail early on a corrupt line
    add = [json.dumps(r) for n in news for r in new_rows(n)]
    Path(dst).write_text("".join(l + "\n" for l in lines + add))
    return len(lines), len(add)


def merge_s3(old, news, out):
    copy_tree(old, out)
    n = append_rows(old / "rows.jsonl", [d / "rows.jsonl" for d in news], out / "rows.jsonl")
    merged, units = 0, []
    for new in news:
        ok = {(r["dataset"], r["model"]) for r in new_rows(new / "rows.jsonl") if r["exp"] == "load"}
        units += [new / f"q_{d}_{m}_qdrant.npz" for d, m in sorted(ok)]
    names = [f.name for f in units]
    assert len(names) == len(set(names)), f"unit in more than one new folder: {names}"
    for f in units:
        zn = dict(np.load(f))
        assert any(k.startswith("E3_") for k in zn), f"{f.name}: no E3 arrays"
        fo = old / f.name
        if fo.exists():
            zo = np.load(fo)
            zn.update({k: zo[k] for k in zo.files if k.startswith("E4_")})
        np.savez_compressed(out / f.name, **zn)
        merged += 1
    for new in news:
        tag = "qnpfix_" if (new / "PRESET_QNPFIX").exists() or list(new.glob("env_QNPFIX_*")) else "qnp_"
        for f in new.iterdir():
            if (f.name.startswith(("env_QNP", "PRESET_")) or f.suffix == ".log"
                    or f.name.startswith("summary_")):
                shutil.copy(f, out / f"{tag}{f.name}")
    return n, merged


def merge_s4(old, new, out):
    copy_tree(old, out)
    n = append_rows(old / "rows.jsonl", [new / "rows.jsonl"], out / "rows.jsonl")
    merged = 0
    for f in sorted(new.glob("s4_*.npz")):
        zn = np.load(f)
        assert all(k.startswith("qdrant__") for k in zn.files), f"{f.name}: non-Qdrant arrays in the QNP output"
        zo = np.load(old / f.name)
        z = {k: zo[k] for k in zo.files if not k.startswith("qdrant__")}
        z.update({k: zn[k] for k in zn.files})
        missing = {k for k in zo.files if k.startswith("qdrant__")} - set(zn.files)
        assert not missing, f"{f.name}: QNP lacks {sorted(missing)[:3]}"
        np.savez_compressed(out / f.name, **z)
        merged += 1
    for f in new.iterdir():
        if f.name.startswith(("env_QNP", "worker_", "summary_")) or (f.name.startswith("gen_cache_gpu")
                                                                   and "_prev" not in f.name):
            shutil.copy(f, out / f"qnp_{f.name}")
    return n, merged


def main():
    ap = argparse.ArgumentParser()
    for a in ("--s3-old", "--s4-old", "--s4-new", "--out"):
        ap.add_argument(a, required=True)
    ap.add_argument("--s3-new", required=True, nargs="+")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists; choose a new folder")
    n3, m3 = merge_s3(Path(a.s3_old), [Path(d) for d in a.s3_new], out / "s3")
    n4, m4 = merge_s4(Path(a.s4_old), Path(a.s4_new), out / "s4")
    print(f"S3: {n3[0]} old + {n3[1]} QNP rows, {m3} Qdrant npz | S4: {n4[0]} old + {n4[1]} QNP rows, "
          f"{m4} npz -> {out}")


if __name__ == "__main__":
    main()
