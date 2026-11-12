#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S4 — End-to-end RAG: does the retrieval difference between Qdrant, Weaviate and pgvector (and between ef settings)
     reach the generated answer?

Kaggle: Accelerator = GPU T4 x2, Internet = On.
Inputs (Add Input):
  * the S1 dataset (vdb_data: corpus/queries/qrels + exact ground truth)
  * the FINAL S3 results as a dataset (unzip s3_results.zip into it; needs q_<dataset>_<model>_<db>.npz files)
Paste the whole file into ONE cell. Choose PRESET:
  "SMOKE"  squad/bge-base, 100 queries, ef in {10, 128} + hybrid + references          (~20 min incl. model download)
  "FULL"   squad + hotpotqa_sub  x  {bge-base, bge-small}, 1000 queries, every S3 config (~3-5 h on T4 x2)
  "QNP"    same jobs, Qdrant configs only, from the S3 "QNP" rerun (Qdrant without payload indexes).
           Inputs: the S1 dataset, the S3 QNP results (folder with the PRESET_QNP marker) and, optionally,
           the FULL S4 results: their gen_cache_gpu*.jsonl are reused, so prompts already answered in FULL
           (identical question + identical ordered passages) are not generated again.          (~1-2 h)
Resumable: generations are cached per prompt (gen_cache_gpu*.jsonl). After an interruption just run the cell again.

Generator: Qwen2.5-7B-Instruct, 4-bit NF4 (bitsandbytes), greedy decoding, max 24 new tokens, top-5 passages in
retrieved order. One model replica per GPU (data parallel); identical prompts are generated once, so configs that
retrieve the same top-5 for a query share the answer (greedy decoding is deterministic per prompt).

Configs per (dataset, model):
  none/closed_book        no context                                (once per dataset)
  gold/oracle             gold passage(s) from qrels                 (once per dataset)
  exact/dense             exact dense top-5 (S1 ground truth)
  exact/lexical|hybrid    exact BM25 / exact RRF(dense, BM25)        (bge-base only; lexical once per dataset)
  <db>/dense/ef<e>        E3 HNSW search, k=10, ef in {10..256} -> first 5
  <db>/hybrid, <db>/lexical   E5 (bge-base only)
Metrics per query: EM and token-F1 (SQuAD normalisation, max over gold answers), answer-in-context (normalised
answer string inside the 5 passages; NaN for yes/no answers), gold hit@5 / all-gold@5 (qrels), overlap@5 with
the exact dense top-5.
Outputs (s4_results/): rows.jsonl (per-config means), s4_<dataset>_<model>.npz (per-query arrays for S5),
summary_S4.csv, env_*.json, s4_results.zip.
"""

import os, sys, re, json, time, glob, shutil, string, hashlib, subprocess, platform, traceback
from pathlib import Path
from collections import Counter

PRESET = "QNP"              # "SMOKE" | "FULL" | "QNP"

LLM_ID = "Qwen/Qwen2.5-7B-Instruct"
GEN = dict(max_new_tokens=24, top_n=5, passage_words=220, max_batch=32, token_budget=16384)
PRESETS = {
    "SMOKE": dict(JOBS=[("squad", "bge-base")], NQ=100, EFS=[10, 128], E5=True),
    "FULL": dict(JOBS=[(d, m) for d in ("squad", "hotpotqa_sub") for m in ("bge-base", "bge-small")],
                 NQ=None, EFS=[10, 16, 32, 64, 128, 256], E5=True),
    "QNP": dict(JOBS=[(d, m) for d in ("squad", "hotpotqa_sub") for m in ("bge-base", "bge-small")],
                NQ=None, EFS=[10, 16, 32, 64, 128, 256], E5=True, DBS=["qdrant"], REFS=False),
}
P = PRESETS[PRESET]
DBS = P.get("DBS", ["qdrant", "weaviate", "pgvector"])
DATA_HINT = "/kaggle/input/datasets/talhakabakus/vdb-data/vdb_data"
OUT = (Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()) / "s4_results"
HF_HOME = "/tmp/hf"

SYSTEM_RAG = ("You are a question answering assistant. Answer the question using the provided passages. "
              "Reply with the shortest possible answer (usually a few words: a name, number, date or short phrase), "
              "with no explanation and no full sentence. If the passages do not contain the answer, give your best guess.")
SYSTEM_CB = ("You are a question answering assistant. Reply with the shortest possible answer (usually a few words: "
             "a name, number, date or short phrase), with no explanation and no full sentence.")


# ============================================================================ text metrics (SQuAD v1.1 official)
def normalize_answer(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1_score(pred, gold):
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


def em_score(pred, gold):
    return float(normalize_answer(pred) == normalize_answer(gold))


def best(fn, pred, golds):
    return max(fn(pred, g) for g in golds) if golds else 0.0


def clean_answer(text):
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    if lines and re.match(r"^question\s*[:：]", lines[0], flags=re.I):   # model echoed the question first
        ans = [l for l in lines[1:] if re.match(r"^(final answer|answer)\s*[:：]", l, flags=re.I)]
        lines = ans or lines[1:]
    t = lines[0] if lines else ""
    t = re.sub(r"^(final answer|answer)\s*[:：]\s*", "", t, flags=re.I)
    return t.strip().strip('"').strip()


def answer_in_context(golds, ctx_norm):
    g = [normalize_answer(a) for a in golds]
    if all(a in ("yes", "no") for a in g):
        return float("nan")
    return float(any(a and a in ctx_norm for a in g))


# ============================================================================ prompts
def passage(title, text, words):
    w = text.split()
    body = " ".join(w[:words]) + (" ..." if len(w) > words else "")
    return f"Title: {title}\n{body}" if title else body


def messages_for(question, passages):
    if not passages:
        return [{"role": "system", "content": SYSTEM_CB}, {"role": "user", "content": f"Question: {question}\nAnswer:"}]
    ctx = "\n\n".join(f"[{i+1}] {p}" for i, p in enumerate(passages))
    return [{"role": "system", "content": SYSTEM_RAG},
            {"role": "user", "content": f"Passages:\n{ctx}\n\nQuestion: {question}\nAnswer:"}]


def prompt_hash(msgs):
    return hashlib.sha1(json.dumps(msgs, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


# ============================================================================ configs from S3 outputs
def top_ids(arr, n):
    """per-query list of the first n valid (>= 0) ids, retrieved order kept."""
    return [[int(d) for d in row if d >= 0][:n] for row in arr]


def build_configs(dataset, model, s3dir, GT, gold, nq, first_model):
    """returns list of (name, db, method, ef, ids[list per query])."""
    n, cfgs = GEN["top_n"], []
    refs = P.get("REFS", True)                       # QNP: references (closed book, oracle, exact) exist from FULL
    if refs and first_model:
        cfgs.append(("none/closed_book", "none", "closed_book", None, [[] for _ in range(nq)]))
        cfgs.append(("gold/oracle", "gold", "oracle", None, [sorted(g)[:n] for g in gold]))
    if refs:
        cfgs.append(("exact/dense", "exact", "dense", "exact", top_ids(GT[:nq], n)))
    ex = s3dir / f"q_{dataset}_{model}_exact.npz"
    if refs and P["E5"] and ex.exists():
        z = np.load(ex)
        if "E5_hybrid_ids" in z.files:
            cfgs.append(("exact/hybrid", "exact", "hybrid", "exact", top_ids(z["E5_hybrid_ids"][:nq], n)))
            if first_model:
                cfgs.append(("exact/lexical", "exact", "lexical", "exact", top_ids(z["E5_lexical_ids"][:nq], n)))
    for db in DBS:
        f = s3dir / f"q_{dataset}_{model}_{db}.npz"
        if not f.exists():
            raise FileNotFoundError(f"S3 output missing: {f}")
        z = np.load(f)
        for ef in P["EFS"]:
            cfgs.append((f"{db}/dense/ef{ef}", db, "dense", ef, top_ids(z[f"E3_k10_ef{ef}_ids"][:nq], n)))
        if P["E5"] and "E5_hybrid_ids" in z.files:
            cfgs.append((f"{db}/hybrid", db, "hybrid", None, top_ids(z["E5_hybrid_ids"][:nq], n)))
            if first_model:
                cfgs.append((f"{db}/lexical", db, "lexical", None, top_ids(z["E5_lexical_ids"][:nq], n)))
    for c in cfgs:
        assert len(c[4]) == nq, (c[0], len(c[4]), nq)
    return cfgs


# ============================================================================ GPU worker (one process per GPU)
WORKER = r'''
import sys, json, time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
shard, cache, model_dir, log_path, max_new, max_bs, budget = sys.argv[1:8]
max_new, max_bs, budget = int(max_new), int(max_bs), int(budget)
log = open(log_path, "a", buffering=1)
def say(*a): print(time.strftime("%H:%M:%S"), *a, file=log, flush=True)
done = set()
try:
    for l in open(cache):
        done.add(json.loads(l)["h"])
except FileNotFoundError:
    pass
items = [json.loads(l) for l in open(shard)]
items = [it for it in items if it["h"] not in done]
say(f"{len(items)} prompts to generate ({len(done)} cached)")
if not items:
    sys.exit(0)
tok = AutoTokenizer.from_pretrained(model_dir)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
import transformers
kw = {"dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype": torch.float16}
model = AutoModelForCausalLM.from_pretrained(model_dir, quantization_config=q, device_map={"": 0},
                                             attn_implementation="sdpa", **kw).eval()
say("model loaded:", torch.cuda.get_device_name(0), f"{torch.cuda.memory_allocated()/2**30:.1f} GiB")
texts = [tok.apply_chat_template(it["messages"], tokenize=False, add_generation_prompt=True) for it in items]
lens = [len(tok(t, add_special_tokens=False).input_ids) for t in texts]
order = sorted(range(len(items)), key=lambda i: (-lens[i], items[i]["h"]))   # longest first: OOM shows up early
batches, cur = [], []
for i in order:
    if cur and (len(cur) + 1 > max_bs or (len(cur) + 1) * lens[cur[0]] > budget):
        batches.append(cur); cur = []
    cur.append(i)
if cur:
    batches.append(cur)
out = open(cache, "a", buffering=1)

@torch.inference_mode()
def run(idx):
    enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(0)
    try:
        g = model.generate(**enc, max_new_tokens=max_new, do_sample=False, temperature=None, top_p=None, top_k=None,
                           pad_token_id=tok.pad_token_id)
    except torch.cuda.OutOfMemoryError:
        del enc
        torch.cuda.empty_cache()
        if len(idx) == 1:
            raise
        h = len(idx) // 2
        run(idx[:h]); run(idx[h:])
        return
    dec = tok.batch_decode(g[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
    for i, d in zip(idx, dec):
        out.write(json.dumps({"h": items[i]["h"], "out": d, "in_tok": lens[i]}, ensure_ascii=False) + "\n")

t0, n = time.time(), 0
for b, idx in enumerate(batches):
    run(idx)
    n += len(idx)
    if b % 10 == 0 or b == len(batches) - 1:
        say(f"{n}/{len(items)} ({n/(time.time()-t0):.2f}/s, batch {len(idx)} x {lens[idx[0]]} tok)")
say("done")
'''


def gpu_count():
    try:
        return int(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.count("GPU "))
    except Exception:
        return 0


def read_cache():
    ans = {}
    for f in sorted(OUT.glob("gen_cache_gpu*.jsonl")):
        for l in open(f):
            try:
                r = json.loads(l)
            except json.JSONDecodeError:          # a line cut by an interruption
                continue
            ans[r["h"]] = r
    return ans


def generate_all(pending, model_dir):
    """pending: {hash: messages}. Splits over GPUs, runs workers, waits with progress logging."""
    if not pending:
        print("   all prompts cached", flush=True)
        return
    ng = gpu_count()
    if ng == 0:
        raise SystemExit("No GPU found: set Accelerator = GPU T4 x2")
    wpath = OUT / "s4_worker.py"
    wpath.write_text(WORKER)
    hs = sorted(pending)
    procs = []
    for g in range(ng):
        shard = OUT / f"shard_gpu{g}.jsonl"
        with open(shard, "w") as f:
            for h in hs[g::ng]:
                f.write(json.dumps({"h": h, "messages": pending[h]}, ensure_ascii=False) + "\n")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), HF_HOME=HF_HOME, TOKENIZERS_PARALLELISM="false")
        cmd = [sys.executable, str(wpath), str(shard), str(OUT / f"gen_cache_gpu{g}.jsonl"), model_dir,
               str(OUT / f"worker_gpu{g}.log"), str(GEN["max_new_tokens"]), str(GEN["max_batch"]), str(GEN["token_budget"])]
        procs.append(subprocess.Popen(cmd, env=env, stdin=subprocess.DEVNULL, stdout=open(OUT / f"worker_gpu{g}.out", "a"),
                                      stderr=subprocess.STDOUT))
    print(f"   generating {len(pending):,} prompts on {ng} GPU(s) ...", flush=True)
    t0 = time.time()
    while any(p.poll() is None for p in procs):
        time.sleep(60)
        got = sum(1 for h in hs if h in read_cache())
        rate = got / (time.time() - t0)
        eta = (len(hs) - got) / rate / 60 if rate > 0 else float("nan")
        print(f"   [{time.strftime('%H:%M:%S')}] {got:,}/{len(hs):,} prompts  {rate:.2f}/s  ETA {eta:.0f} min", flush=True)
    for g, p in enumerate(procs):
        if p.returncode != 0:
            tail = (OUT / f"worker_gpu{g}.out").read_text()[-3000:]
            raise RuntimeError(f"GPU worker {g} failed (rc={p.returncode}):\n{tail}")


# ============================================================================ main pipeline
def data_root():
    if Path(DATA_HINT, "meta.json").exists():
        return Path(DATA_HINT)
    hits = glob.glob("/kaggle/input/**/meta.json", recursive=True)
    if not hits:
        raise FileNotFoundError("S1 dataset (meta.json) not found under /kaggle/input")
    return Path(hits[0]).parent


def s3_root():
    if PRESET == "QNP":
        # The S3 QNP folder carries PRESET_QNP and env_QNP_*.json (Kaggle may drop the extensionless marker on upload)
        hits = (glob.glob("/kaggle/input/**/PRESET_QNP", recursive=True)
                + glob.glob("/kaggle/input/**/env_QNP_*.json", recursive=True))
        hits = [h for h in hits if (Path(h).parent / "q_squad_bge-base_qdrant.npz").exists()]
        if not hits:
            print("\n".join(sorted(glob.glob("/kaggle/input/**/*", recursive=True))[:200]), flush=True)
            raise FileNotFoundError("S3 QNP results (PRESET_QNP or env_QNP_*.json next to q_*_qdrant.npz) "
                                    "not found under /kaggle/input; the listing above shows what is attached")
        return Path(hits[0]).parent
    hits = glob.glob("/kaggle/input/**/q_squad_bge-base_exact.npz", recursive=True)
    if not hits:
        raise FileNotFoundError("S3 results (q_*.npz) not found under /kaggle/input: add them as a dataset")
    return Path(hits[0]).parent


def load_dataset_job(root, dataset, nq):
    d = root / dataset
    corpus = pd.read_parquet(d / "corpus.parquet", columns=["doc_id", "title", "text"])
    queries = pd.read_parquet(d / "queries.parquet")
    qrels = pd.read_parquet(d / "qrels.parquet")
    if nq:
        queries = queries.iloc[:nq].reset_index(drop=True)
    id2idx = {x: i for i, x in enumerate(corpus["doc_id"])}
    gold = {}
    for qid, did, s in qrels[["query_id", "doc_id", "score"]].itertuples(index=False):
        if s > 0 and did in id2idx:
            gold.setdefault(qid, set()).add(id2idx[did])
    gold = [gold.get(q, set()) for q in queries["query_id"]]
    answers = [list(a) for a in queries["answers"]]
    return corpus, queries, gold, answers


def emit(row):
    row.setdefault("preset", PRESET)
    row["ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(OUT / "rows.jsonl", "a") as f:
        f.write(json.dumps(row, default=float) + "\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    root, s3dir = data_root(), s3_root()
    print(f">> S1 data: {root}\n>> S3 results: {s3dir}", flush=True)
    if PRESET == "QNP":                              # reuse the FULL generations (identical prompts -> identical answers)
        prev = [f for f in glob.glob("/kaggle/input/**/gen_cache_gpu*.jsonl", recursive=True) if "_prev" not in f]
        for i, f in enumerate(sorted(prev)):
            dst = OUT / f"gen_cache_gpu_prev{i}.jsonl"
            if not dst.exists():
                shutil.copy(f, dst)
        print(f">> reused generation caches: {len(prev)} file(s)", flush=True)

    # ---- 1. prompts for every (job, config, query)
    jobs, pending, seen_ds = [], {}, set()
    cache = read_cache()
    for dataset, model in P["JOBS"]:
        nq = P["NQ"]
        corpus, queries, gold, answers = load_dataset_job(root, dataset, nq)
        nq = len(queries)
        GT = np.load(root / dataset / model / "gt_top100.npy")
        assert len(GT) >= nq
        first = dataset not in seen_ds
        seen_ds.add(dataset)
        cfgs = build_configs(dataset, model, s3dir, GT, gold, nq, first)
        titles, texts = corpus["title"].fillna("").tolist(), corpus["text"].fillna("").tolist()
        qtext = queries["text"].tolist()
        hashes = {}
        for name, db, method, ef, ids in cfgs:
            hs = []
            for qi in range(nq):
                msgs = messages_for(qtext[qi], [passage(titles[d], texts[d], GEN["passage_words"]) for d in ids[qi]])
                h = prompt_hash(msgs)
                hs.append(h)
                if h not in cache:
                    pending[h] = msgs
            hashes[name] = hs
        n_unique = len({h for hs in hashes.values() for h in hs})
        print(f">> {dataset}/{model}: {nq} queries x {len(cfgs)} configs = {nq*len(cfgs):,} prompts, "
              f"{n_unique:,} unique", flush=True)
        jobs.append(dict(dataset=dataset, model=model, nq=nq, cfgs=cfgs, hashes=hashes, gold=gold, answers=answers,
                         titles=titles, texts=texts, GT=GT[:nq]))
    print(f">> {len(pending):,} prompts not cached yet", flush=True)

    # ---- 2. generate
    if pending:
        from huggingface_hub import snapshot_download
        t = time.time()
        model_dir = snapshot_download(LLM_ID, cache_dir=f"{HF_HOME}/hub",
                                      allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "merges.txt"])
        print(f">> model files ready in {time.time()-t:.0f}s: {model_dir}", flush=True)
        generate_all(pending, model_dir)
    cache = read_cache()
    missing = [h for j in jobs for hs in j["hashes"].values() for h in hs if h not in cache]
    if missing:
        raise RuntimeError(f"{len(missing)} prompts have no generation; run the cell again")

    # ---- 3. score
    for j in jobs:
        store, nq = {}, j["nq"]
        exact5 = [set(r) for r in top_ids(j["GT"], GEN["top_n"])]
        for name, db, method, ef, ids in j["cfgs"]:
            em, f1, aic, hit, allg, ov, ntok = (np.zeros(nq, np.float32) for _ in range(7))
            preds = []
            for qi in range(nq):
                r = cache[j["hashes"][name][qi]]
                pred, golds = clean_answer(r["out"]), j["answers"][qi]
                preds.append(pred)
                em[qi], f1[qi] = best(em_score, pred, golds), best(f1_score, pred, golds)
                ctx = normalize_answer(" ".join(j["titles"][d] + " " + j["texts"][d] for d in ids[qi]))
                aic[qi] = answer_in_context(golds, ctx) if ids[qi] else float("nan")
                g, s = j["gold"][qi], set(ids[qi])
                hit[qi] = float(bool(g & s)) if g else float("nan")
                allg[qi] = float(g <= s) if g else float("nan")
                ov[qi] = len(s & exact5[qi]) / max(len(exact5[qi]), 1) if method != "closed_book" else float("nan")
                ntok[qi] = r.get("in_tok", 0)
            key = name.replace("/", "__")
            for k, v in dict(em=em, f1=f1, aic=aic, hit5=hit, allgold5=allg, overlap5=ov).items():
                store[f"{key}__{k}"] = v
            store[f"{key}__pred"] = np.array(preds, dtype=str)
            emit(dict(dataset=j["dataset"], model=j["model"], config=name, db=db, method=method, ef=ef, nq=nq,
                      em=float(em.mean()), f1=float(f1.mean()), ans_in_ctx=float(np.nanmean(aic)) if np.isfinite(aic).any() else None,
                      hit5=float(np.nanmean(hit)), allgold5=float(np.nanmean(allg)),
                      overlap5=float(np.nanmean(ov)) if method != "closed_book" else None,
                      mean_in_tok=float(ntok.mean()), llm=LLM_ID, **{k: v for k, v in GEN.items()}))
        np.savez_compressed(OUT / f"s4_{j['dataset']}_{j['model']}.npz", **store)
    summarize()
    print(f"Finished in {(time.time()-t_start)/60:.0f} min", flush=True)


def summarize():
    df = pd.DataFrame([json.loads(l) for l in open(OUT / "rows.jsonl")])
    df = df[df.preset == PRESET].drop_duplicates(["dataset", "model", "config"], keep="last")
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 500)
    cols = ["dataset", "model", "config", "em", "f1", "ans_in_ctx", "hit5", "allgold5", "overlap5", "mean_in_tok"]
    t = df[cols].sort_values(["dataset", "model", "config"]).round(4)
    t.to_csv(OUT / "summary_S4.csv", index=False)
    print("\n===== S4: end-to-end RAG (EM / F1 / answer-in-context / gold hit@5 / overlap@5 with exact) =====\n",
          t.to_string(index=False))
    shutil.make_archive(str(OUT.parent / "s4_results"), "zip", OUT)
    try:
        from IPython.display import FileLink, display
        display(FileLink("s4_results.zip"))
    except Exception:
        pass
    print(f"Results: {OUT} | archive: {OUT.parent / 's4_results.zip'}")


def env_info():
    import importlib.metadata as md
    v = {}
    for p in ("torch", "transformers", "bitsandbytes", "accelerate", "huggingface_hub", "numpy", "pandas"):
        try:
            v[p] = md.version(p)
        except Exception:
            v[p] = None
    gpus = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip().splitlines() if shutil.which("nvidia-smi") else []
    info = dict(preset=PRESET, cfg=P, gen=GEN, llm=LLM_ID, system_rag=SYSTEM_RAG, system_cb=SYSTEM_CB, versions=v,
                gpus=gpus, python=sys.version, platform=platform.platform())
    (OUT / f"env_{PRESET}_{int(time.time())}.json").write_text(json.dumps(info, indent=2, default=str))
    print(">> env:", json.dumps(dict(versions=v, gpus=gpus)), flush=True)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    print(">> Installing packages ...", flush=True)
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "bitsandbytes>=0.45", "accelerate>=1.0", "pyarrow"], capture_output=True, text=True)
    if r.returncode:
        print(r.stdout[-2000:], r.stderr[-2000:])
        raise SystemExit("pip install failed")

import numpy as np
import pandas as pd
try:
    import transformers as _tf                      # Kaggle's preinstalled build matches its torch; upgrade only if old
    if tuple(int(x) for x in _tf.__version__.split(".")[:2]) < (4, 46):
        raise ImportError(_tf.__version__)
except ImportError:
    if __name__ == "__main__":
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers>=4.46"], check=True)

if __name__ == "__main__":
    env_info()
    main()
