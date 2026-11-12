#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S1 — Datasets, embeddings and exact ground truth (Kaggle GPU notebook).

Kaggle settings: Accelerator = GPU T4 x2 (or P100), Internet = On.
Recommended: "Save Version" -> "Save & Run All (Commit)" so outputs persist (~1.5–3 h).
Afterwards: Output tab -> "New Dataset" -> name it e.g. `vdb-rag-data`; attach it to S2–S4.

Produces, for every dataset under /kaggle/working/vdb_data/<dataset>/ :
  corpus.parquet   doc_idx, doc_id, title, text, a2, a10, a100, a1000   (aX = int attribute with X values
                                                                           -> equality filter selectivity 1/X)
  queries.parquet  q_idx, query_id, text, [answers], q_a2, q_a10, q_a100, q_a1000
  qrels.parquet    query_id, doc_id, score
  <model>/docs.npy, queries.npy          float32, L2-normalised
  <model>/gt_top100.npy, gt_scores.npy   exact cosine top-100 (doc_idx, int32 / float32)
  <model>/gt_filt_a{X}.npy                exact filtered top-100 for each selectivity (-1 padded)
and vdb_data/meta.json (counts, model revisions, timings, hardware).

Datasets
  BEIR (test split): scifact, nfcorpus, fiqa, trec-covid, quora           -> retrieval quality + system perf
  hotpotqa_sub: 1000 BEIR-HotpotQA test queries with gold answers, corpus = their gold docs
                + 200K random distractors from the 5.2M BEIR corpus      -> end-to-end RAG
  squad: all unique SQuAD v1.1 paragraphs (train+dev), 1000 dev questions -> end-to-end RAG
Embedding models: BAAI/bge-small-en-v1.5 (384-d), BAAI/bge-base-en-v1.5 (768-d).
"""

import os, sys, re, io, json, time, gzip, zipfile, subprocess, random
from pathlib import Path

CFG = dict(
    BEIR=["scifact", "nfcorpus", "fiqa", "trec-covid", "quora"],
    MODELS={"bge-small": "BAAI/bge-small-en-v1.5", "bge-base": "BAAI/bge-base-en-v1.5"},
    QUERY_INSTRUCTION="Represent this sentence for searching relevant passages: ",
    SYMMETRIC=["quora"],              # no query instruction for symmetric similarity tasks
    MAX_SEQ_LEN=512, BATCH={"bge-small": 256, "bge-base": 128},
    HOTPOT_N_QUERIES=1000, HOTPOT_N_DISTRACTORS=200_000,
    SQUAD_N_QUERIES=1000,
    ATTR_CARD=[2, 10, 100, 1000], GT_K=100, SEED=42,
)
OUT = (Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()) / "vdb_data"
RAW = Path("/tmp/vdb_raw")
OUT.mkdir(parents=True, exist_ok=True)
RAW.mkdir(parents=True, exist_ok=True)


def sh(cmd, timeout=3600):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd}\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    return r.stdout


print(">> Installing packages ...", flush=True)
sh(f"{sys.executable} -m pip install -q 'sentence-transformers>=3.0' 'datasets>=2.19' pyarrow pandas requests huggingface_hub", 1800)

import numpy as np
import pandas as pd
import requests
import torch
from sentence_transformers import SentenceTransformer

META_PATH = OUT / "meta.json"
META = json.loads(META_PATH.read_text()) if META_PATH.exists() else {"datasets": {}, "models": {}}
META["config"] = CFG
META["hardware"] = dict(gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                        torch=torch.__version__, python=sys.version.split()[0])
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print(">> Hardware:", META["hardware"], flush=True)


def save_meta():
    META_PATH.write_text(json.dumps(META, indent=2, default=str))


def download(url, dest, retries=3):
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    for a in range(retries):
        try:
            with requests.get(url, stream=True, timeout=600) as r:
                r.raise_for_status()
                tmp = dest.with_suffix(dest.suffix + ".part")
                with open(tmp, "wb") as f:
                    for ch in r.iter_content(1 << 22):
                        f.write(ch)
                tmp.rename(dest)
                return dest
        except Exception as e:
            print(f"   download failed ({a+1}/{retries}): {e}", flush=True)
            time.sleep(10)
    raise RuntimeError(f"cannot download {url}")


# ----------------------------------------------------------------------------- BEIR loading
def beir_files(name):
    """Return (corpus_lines_iter_factory, queries_df, qrels_df) for a BEIR dataset."""
    zpath = RAW / f"{name}.zip"
    try:
        download(f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip", zpath)
        z = zipfile.ZipFile(zpath)

        def corpus_iter():
            with z.open(f"{name}/corpus.jsonl") as f:
                for line in io.TextIOWrapper(f, encoding="utf-8"):
                    yield line
        queries = pd.read_json(z.open(f"{name}/queries.jsonl"), lines=True, dtype={"_id": str})
        qrels = pd.read_csv(z.open(f"{name}/qrels/test.tsv"), sep="\t", dtype={"query-id": str, "corpus-id": str})
    except Exception as e:                       # fallback: Hugging Face mirrors
        print(f"   UKP mirror failed ({e}); trying Hugging Face BeIR/{name}", flush=True)
        from huggingface_hub import hf_hub_download
        cpath = hf_hub_download(f"BeIR/{name}", "corpus.jsonl.gz", repo_type="dataset")
        qpath = hf_hub_download(f"BeIR/{name}", "queries.jsonl.gz", repo_type="dataset")
        rpath = hf_hub_download(f"BeIR/{name}-qrels", "test.tsv", repo_type="dataset")

        def corpus_iter():
            with gzip.open(cpath, "rt", encoding="utf-8") as f:
                for line in f:
                    yield line
        queries = pd.read_json(qpath, lines=True, dtype={"_id": str})
        qrels = pd.read_csv(rpath, sep="\t", dtype={"query-id": str, "corpus-id": str})
    qrels = qrels.rename(columns={"query-id": "query_id", "corpus-id": "doc_id"})[["query_id", "doc_id", "score"]]
    queries = queries.rename(columns={"_id": "query_id"})[["query_id", "text"]]
    return corpus_iter, queries, qrels


def parse_corpus(lines, keep=None):
    rows = []
    for line in lines:
        d = json.loads(line)
        if keep is None or d["_id"] in keep:
            rows.append((str(d["_id"]), d.get("title") or "", d.get("text") or ""))
    return pd.DataFrame(rows, columns=["doc_id", "title", "text"])


def add_attrs(df, prefix, rng):
    for c in CFG["ATTR_CARD"]:
        df[f"{prefix}a{c}"] = rng.integers(0, c, len(df)).astype(np.int16)
    return df


def finalize(name, corpus, queries, qrels):
    rng = np.random.default_rng(CFG["SEED"] + sum(map(ord, name)))   # deterministic per dataset
    corpus = corpus.drop_duplicates("doc_id").reset_index(drop=True)
    corpus.insert(0, "doc_idx", np.arange(len(corpus), dtype=np.int32))
    corpus = add_attrs(corpus, "", rng)
    qrels = qrels[qrels["doc_id"].isin(set(corpus["doc_id"])) & (qrels["score"] > 0)]
    queries = queries[queries["query_id"].isin(set(qrels["query_id"]))].reset_index(drop=True)
    queries.insert(0, "q_idx", np.arange(len(queries), dtype=np.int32))
    queries = add_attrs(queries, "q_", rng)
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    corpus.to_parquet(d / "corpus.parquet", index=False)
    queries.to_parquet(d / "queries.parquet", index=False)
    qrels.reset_index(drop=True).to_parquet(d / "qrels.parquet", index=False)
    META["datasets"][name] = dict(n_docs=len(corpus), n_queries=len(queries), n_qrels=len(qrels))
    save_meta()
    print(f"   {name}: {len(corpus):,} docs, {len(queries):,} queries, {len(qrels):,} qrels", flush=True)


def prepare_beir(name):
    if (OUT / name / "queries.parquet").exists():
        return
    print(f">> Preparing BEIR/{name}", flush=True)
    it, q, r = beir_files(name)
    finalize(name, parse_corpus(it()), q, r)


def prepare_hotpot():
    name = "hotpotqa_sub"
    if (OUT / name / "queries.parquet").exists():
        return
    print(">> Preparing HotpotQA subset (downloads ~600 MB BEIR corpus)", flush=True)
    it, q, r = beir_files("hotpotqa")
    # gold answers from the original HotpotQA dev set (BEIR test == HotpotQA dev, same _id)
    try:
        from datasets import load_dataset
        hp = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation").to_pandas()[["id", "answer"]]
    except Exception as e:
        print(f"   HF hotpot_qa failed ({e}); using raw CMU json", flush=True)
        p = download("http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json", RAW / "hotpot_dev.json")
        hp = pd.DataFrame([{"id": x["_id"], "answer": x["answer"]} for x in json.load(open(p))])
    ans = dict(zip(hp["id"], hp["answer"]))
    q = q[q["query_id"].isin(ans.keys()) & q["query_id"].isin(set(r["query_id"]))]
    q = q.sample(n=CFG["HOTPOT_N_QUERIES"], random_state=CFG["SEED"]).reset_index(drop=True)
    q["answers"] = q["query_id"].map(lambda i: [ans[i]])
    r = r[r["query_id"].isin(set(q["query_id"]))]
    gold = set(r["doc_id"])
    # pass 1: count lines; pass 2: keep gold + uniformly sampled distractors
    n_lines = sum(1 for _ in it())
    rng = np.random.default_rng(CFG["SEED"])
    pick = set(rng.choice(n_lines, CFG["HOTPOT_N_DISTRACTORS"], replace=False).tolist())
    idre = re.compile(r'"_id":\s*"([^"]+)"')
    keep_lines = []
    for i, line in enumerate(it()):
        if i in pick:
            keep_lines.append(line)
        else:
            m = idre.search(line)
            if m and m.group(1) in gold:
                keep_lines.append(line)
    corpus = parse_corpus(keep_lines)
    META.setdefault("notes", {})["hotpotqa_sub"] = dict(full_corpus_lines=n_lines, gold_docs=len(gold),
                                                        gold_found=int(corpus["doc_id"].isin(gold).sum()))
    finalize(name, corpus, q, r)


def prepare_squad():
    name = "squad"
    if (OUT / name / "queries.parquet").exists():
        return
    print(">> Preparing SQuAD v1.1", flush=True)
    from datasets import load_dataset
    ds = load_dataset("rajpurkar/squad")
    ctx2id, rows = {}, []
    for split in ("train", "validation"):
        for t, c in zip(ds[split]["title"], ds[split]["context"]):
            if c not in ctx2id:
                ctx2id[c] = f"sq{len(ctx2id)}"
                rows.append((ctx2id[c], t.replace("_", " "), c))
    corpus = pd.DataFrame(rows, columns=["doc_id", "title", "text"])
    val = ds["validation"].to_pandas()
    val = val.sample(n=CFG["SQUAD_N_QUERIES"], random_state=CFG["SEED"]).reset_index(drop=True)
    q = pd.DataFrame({"query_id": val["id"].astype(str), "text": val["question"],
                      "answers": val["answers"].map(lambda a: list(dict.fromkeys(a["text"])))})
    r = pd.DataFrame({"query_id": val["id"].astype(str), "doc_id": val["context"].map(ctx2id), "score": 1})
    finalize(name, corpus, q, r)


# ----------------------------------------------------------------------------- embeddings + GT
def exact_topk(D, Q, k, d_attr=None, q_attr=None, chunk=256):
    """Exact cosine top-k on GPU (inputs L2-normalised float32). Optional equality filter."""
    Dt = torch.from_numpy(D).to(DEV)
    da = torch.from_numpy(d_attr.astype(np.int32)).to(DEV) if d_attr is not None else None
    k = min(k, D.shape[0])
    I = np.full((len(Q), k), -1, dtype=np.int32)
    S = np.full((len(Q), k), -np.inf, dtype=np.float32)
    for s in range(0, len(Q), chunk):
        qt = torch.from_numpy(Q[s:s + chunk]).to(DEV)
        sc = qt @ Dt.T
        if da is not None:
            qa = torch.from_numpy(q_attr[s:s + chunk].astype(np.int32)).to(DEV)
            sc = sc.masked_fill(da[None, :] != qa[:, None], float("-inf"))
        v, i = torch.topk(sc, k, dim=1)
        v, i = v.cpu().numpy(), i.cpu().numpy().astype(np.int32)
        i[~np.isfinite(v)] = -1
        I[s:s + chunk], S[s:s + chunk] = i, v
    del Dt
    torch.cuda.empty_cache()
    return I, S


_models = {}


def get_model(short):
    if short not in _models:
        name = CFG["MODELS"][short]
        m = SentenceTransformer(name, device=DEV)
        m.max_seq_length = CFG["MAX_SEQ_LEN"]
        if DEV == "cuda":
            m = m.half()
        try:
            from huggingface_hub import model_info
            META["models"][short] = dict(name=name, revision=model_info(name).sha,
                                         dim=m.get_sentence_embedding_dimension())
        except Exception:
            META["models"][short] = dict(name=name, dim=m.get_sentence_embedding_dimension())
        save_meta()
        _models[short] = m
    return _models[short]


def embed_dataset(name, short):
    d = OUT / name / short
    if (d / f"gt_filt_a{CFG['ATTR_CARD'][-1]}.npy").exists():
        print(f"   skip {name}/{short} (done)", flush=True)
        return
    d.mkdir(parents=True, exist_ok=True)
    corpus = pd.read_parquet(OUT / name / "corpus.parquet")
    queries = pd.read_parquet(OUT / name / "queries.parquet")
    m = get_model(short)
    t0 = time.time()
    docs_txt = (corpus["title"].fillna("") + " " + corpus["text"].fillna("")).str.strip().tolist()
    E = m.encode(docs_txt, batch_size=CFG["BATCH"][short], normalize_embeddings=True,
                 convert_to_numpy=True, show_progress_bar=True).astype(np.float32)
    t_docs = time.time() - t0
    prefix = "" if name in CFG["SYMMETRIC"] else CFG["QUERY_INSTRUCTION"]
    QE = m.encode([prefix + t for t in queries["text"].tolist()], batch_size=CFG["BATCH"][short],
                  normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    np.save(d / "docs.npy", E)
    np.save(d / "queries.npy", QE)
    t1 = time.time()
    I, S = exact_topk(E, QE, CFG["GT_K"])
    np.save(d / "gt_top100.npy", I)
    np.save(d / "gt_scores.npy", S)
    for c in CFG["ATTR_CARD"]:
        If, _ = exact_topk(E, QE, CFG["GT_K"], corpus[f"a{c}"].values, queries[f"q_a{c}"].values)
        np.save(d / f"gt_filt_a{c}.npy", If)
    META["datasets"][name].setdefault("embed", {})[short] = dict(
        docs_s=round(t_docs, 1), docs_per_s=round(len(E) / t_docs, 1), gt_s=round(time.time() - t1, 1),
        query_instruction=bool(prefix))
    save_meta()
    print(f"   {name}/{short}: {E.shape} in {t_docs:.0f}s ({len(E)/t_docs:.0f} docs/s)", flush=True)


# ----------------------------------------------------------------------------- main
t_all = time.time()
for b in CFG["BEIR"]:
    prepare_beir(b)
prepare_squad()
prepare_hotpot()
ALL = CFG["BEIR"] + ["squad", "hotpotqa_sub"]
for short in CFG["MODELS"]:
    print(f"\n>> Embedding with {short}", flush=True)
    for name in ALL:
        embed_dataset(name, short)
    _models.pop(short, None)
    torch.cuda.empty_cache()

# sanity check: exact dense retrieval nDCG@10 vs published BEIR numbers (validates the pipeline)
def ndcg10(name, short):
    q = pd.read_parquet(OUT / name / "queries.parquet")
    c = pd.read_parquet(OUT / name / "corpus.parquet", columns=["doc_id"])
    r = pd.read_parquet(OUT / name / "qrels.parquet")
    I = np.load(OUT / name / short / "gt_top100.npy")[:, :10]
    rel = {}
    for qi, di, s in r.itertuples(index=False):
        rel.setdefault(qi, {})[di] = s
    ids = c["doc_id"].values
    out = []
    for qid, row in zip(q["query_id"], I):
        g = rel.get(qid, {})
        # linear gain, as in pytrec_eval / official BEIR evaluation
        dcg = sum(g.get(ids[j], 0) / np.log2(r_ + 2) for r_, j in enumerate(row) if j >= 0)
        ideal = sorted(g.values(), reverse=True)[:10]
        idcg = sum(s / np.log2(r_ + 2) for r_, s in enumerate(ideal))
        out.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(out))

print("\n>> Sanity check — exact-search nDCG@10 (compare with MTEB/BEIR leaderboard values):")
san = {}
for name in ALL:
    san[name] = {s: round(ndcg10(name, s), 4) for s in CFG["MODELS"]}
    print(f"   {name:14s} {san[name]}")
META["sanity_ndcg10_exact"] = san
META["total_runtime_min"] = round((time.time() - t_all) / 60, 1)
save_meta()
print(sh(f"du -sh {OUT} {OUT}/*"))
print(f"Done in {META['total_runtime_min']} min. Now: Output -> New Dataset (e.g. 'vdb-rag-data').")
