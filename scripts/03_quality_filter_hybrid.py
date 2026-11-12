#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S3 — Retrieval quality (E3), filtered search (E4) and hybrid search (E5)
     for Qdrant v1.20.0, Weaviate v1.40.4 and PostgreSQL+pgvector v0.8.7.

Kaggle: Accelerator = None (CPU), Internet = On, Input = the S1 dataset (vdb_data).
Paste the whole file into ONE cell. Choose PRESET:
  "SMOKE"  scifact/bge-base, all three experiments with short grids            (~15 min)
  "Q1"     E3: scifact, nfcorpus, fiqa, trec-covid, squad x {bge-small, bge-base}
           E5: same datasets with bge-base                                      (~3-5 h)
  "Q2"     E3: hotpotqa_sub, quora x {bge-small, bge-base}
           E4: hotpotqa_sub, quora (bge-base);  E5: hotpotqa_sub (bge-base)      (~4-6 h)
  "QNP"    Qdrant only, NO payload indexes (no filter attributes stored): E3 on all 14 units and E5 on the
           six bge-base hybrid units, so that Qdrant's HNSW graph has no filter-aware links, like the graphs
           of Weaviate and pgvector. Payload indexes make Qdrant's filterable HNSW add filter-aware links, which raised its
           unfiltered recall at low ef in Q1/Q2. E4 keeps the Q2 results (filtering needs the payload indexes).
           Output carries the marker file PRESET_QNP; feed it to S4 preset "QNP".            (~2-3 h)
  "QNPFIX" the two QNP units that failed the indexing check (FiQA and TREC-COVID, bge-base: E3 + E5). Writes to
           s3_results_qnpfix/ (archive s3_results_qnpfix.zip); merged by 06_merge_qnp.py.      (~30 min)
Can run in parallel with S2 in a separate Kaggle session.

Per (dataset, model, db): ONE build (common config: cosine, HNSW M=16, efC=128) holding dense vectors,
filter attributes a2/a10/a100/a1000 and — for hybrid jobs — the lexical representation:
  Weaviate : native BM25 on the text property (own tokenizer, k1=1.2, b=0.75), native hybrid (rankedFusion,
             alpha=0.5  == equal-weight RRF with k=60)
  Qdrant   : sparse vectors with client-side BM25 weights (identical to pgvector), server-side RRF (k=60 if supported)
  pgvector : sparsevec with the same client-side BM25 weights (exact scan: pgvector has no inverted index),
             RRF (k=60) in SQL
  Reference: exact dense top-100 (S1 GT) + exact BM25 (scipy) fused with RRF k=60 on the client.
Outputs (s3_results/): rows.jsonl, per-query arrays (npz: retrieved ids, per-query nDCG@10 etc.),
summary_*.csv and s3_results.zip. Retrieved ids for squad/hotpotqa_sub are what S4 (end-to-end RAG) consumes.
"""

import os, sys, re, json, time, glob, shutil, tarfile, subprocess, platform, traceback, math, threading
from pathlib import Path
from collections import Counter

PRESET = "QNPFIX"         # "SMOKE" | "Q1" | "Q2" | "QNP" | "QNPFIX"

VERSIONS = dict(QDRANT_TAG="v1.20.0", WEAVIATE_TAG="v1.40.4", PGVECTOR_TAG="v0.8.7",
                PIP=["qdrant-client==1.20.0", "weaviate-client==4.24.1", "psycopg[binary]==3.3.8",
                     "pgvector==0.5.1", "psutil", "requests", "pandas", "pyarrow", "scipy"])
COMMON = dict(M=16, EF_C=128, EF_INIT=64)
Q1_DS = ["scifact", "nfcorpus", "fiqa", "trec-covid", "squad"]
PRESETS = {
    "SMOKE": dict(E3=[("scifact", "bge-base")], E4=[("scifact", "bge-base")], E5=[("scifact", "bge-base")],
                  EF10=[10, 32, 128], EF100=[100, 256], FILT_EF=[64, 256], FILT_SEL=[10, 100], QUERY_CAP=300),
    "Q1": dict(E3=[(d, m) for d in Q1_DS for m in ("bge-small", "bge-base")], E4=[],
               E5=[(d, "bge-base") for d in Q1_DS],
               EF10=[10, 16, 32, 64, 128, 256], EF100=[100, 256], FILT_EF=[], FILT_SEL=[], QUERY_CAP=2000),
    "Q2": dict(E3=[(d, m) for d in ("hotpotqa_sub", "quora") for m in ("bge-small", "bge-base")],
               E4=[("hotpotqa_sub", "bge-base"), ("quora", "bge-base")], E5=[("hotpotqa_sub", "bge-base")],
               EF10=[10, 16, 32, 64, 128, 256], EF100=[100, 256], FILT_EF=[32, 64, 128, 256],
               FILT_SEL=[2, 10, 100, 1000], QUERY_CAP=2000),
    "QNP": dict(E3=[(d, m) for d in Q1_DS + ["hotpotqa_sub", "quora"] for m in ("bge-small", "bge-base")], E4=[],
                E5=[(d, "bge-base") for d in Q1_DS + ["hotpotqa_sub"]],
                EF10=[10, 16, 32, 64, 128, 256], EF100=[100, 256], FILT_EF=[], FILT_SEL=[], QUERY_CAP=2000,
                DBS=["qdrant"], NO_PAYLOAD=True, SKIP_REF=True),
}
PRESETS["QNPFIX"] = dict(PRESETS["QNP"], E3=[("fiqa", "bge-base"), ("trec-covid", "bge-base")],
                         E5=[("fiqa", "bge-base"), ("trec-covid", "bge-base")])
P = PRESETS[PRESET]
HYB_EF = 128                 # dense ef inside hybrid queries (~0.975 recall@10 in S2 SMOKE for all DBs)
RRF_K = 60
BM25_K1, BM25_B = 1.2, 0.75
DBS = P.get("DBS", ["qdrant", "weaviate", "pgvector"])
SEED = 42
DATA_HINT = "/kaggle/input/datasets/talhakabakus/vdb-data/vdb_data"
OUT = (Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()) / (
    "s3_results_qnpfix" if PRESET == "QNPFIX" else "s3_results")
ROOT = Path("/tmp/vdb")

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    ROOT.mkdir(parents=True, exist_ok=True)
    print(">> Installing pinned Python packages ...", flush=True)
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *VERSIONS["PIP"]], capture_output=True, text=True)
    if r.returncode:
        print(r.stdout[-2000:], r.stderr[-2000:])
        raise SystemExit("pip install failed")

import numpy as np
import pandas as pd
import psutil
import requests
import scipy.sparse as sp


# ============================================================================ helpers
def sh(cmd, check=True, timeout=3600):
    """Run a shell command in its own process group with stdin closed. On timeout the WHOLE group is killed:
    subprocess.run(timeout=...) with shell=True only kills the shell, and grandchildren (apt, dpkg, make) keep the
    output pipes open, so run() can block forever. Output goes to temp files, not pipes, for the same reason."""
    import tempfile, signal
    with tempfile.TemporaryFile("w+") as fo, tempfile.TemporaryFile("w+") as fe:
        p = subprocess.Popen(cmd, shell=True, stdin=subprocess.DEVNULL, stdout=fo, stderr=fe, text=True,
                             start_new_session=True)
        try:
            rc = p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            p.wait()
            rc = None
        fo.seek(0); fe.seek(0)
        out, err = fo.read(), fe.read()
    if rc is None:
        raise RuntimeError(f"Command timed out after {timeout}s (process group killed): {cmd}\n{out[-2000:]}\n{err[-2000:]}")
    if check and rc != 0:
        raise RuntimeError(f"Command failed: {cmd}\n{out[-2000:]}\n{err[-2000:]}")
    return out.strip()


def install_binary(repo, tag, pattern, name):
    bdir = ROOT / "bin" / name
    found = [p for p in bdir.rglob(name) if p.is_file()] if bdir.exists() else []
    if found:
        return found[0]
    rel = requests.get(f"https://api.github.com/repos/{repo}/releases/tags/{tag}", timeout=60)
    rel.raise_for_status()
    an, url = next((a["name"], a["browser_download_url"]) for a in rel.json()["assets"] if re.search(pattern, a["name"]))
    tp = ROOT / "dl" / an
    tp.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(tp, "wb") as f:
            for ch in r.iter_content(1 << 22):
                f.write(ch)
    with tarfile.open(tp) as t:
        t.extractall(bdir)
    tp.unlink()
    b = [p for p in bdir.rglob(name) if p.is_file()][0]
    b.chmod(0o755)
    return b


def wait_http(url, timeout=300):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if requests.get(url, timeout=3).status_code < 500:
                return
        except Exception:
            pass
        time.sleep(0.5)
    raise TimeoutError(url)


def lat_stats(lat_s):
    a = np.asarray(lat_s) * 1000
    return dict(p50_ms=float(np.percentile(a, 50)), p95_ms=float(np.percentile(a, 95)), mean_ms=float(a.mean()))


def pad(rows, k):
    return np.array([list(r[:k]) + [-1] * (k - len(r[:k])) for r in rows], dtype=np.int32)


ROWS, DONE = OUT / "rows.jsonl", OUT / "done.json"


def done_set():
    return set(json.loads(DONE.read_text())) if DONE.exists() else set()


def mark_done(key):
    s = done_set()
    s.add(key)
    DONE.write_text(json.dumps(sorted(s)))


def emit(row):
    row.setdefault("preset", PRESET)
    row["ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(ROWS, "a") as f:
        f.write(json.dumps(row, default=float) + "\n")


# ============================================================================ progress heartbeat
# Long silent phases (HNSW build of 523K x 768 in pgvector, Weaviate synchronous insert) print nothing for many
# minutes. A daemon thread prints the current stage every HB_PERIOD s; during start/insert/build it also shows
# PostgreSQL's own index-build progress. It reads no smaps and stays quiet in the stages being timed.
HB_PERIOD = 120
STAGE = {"name": "init", "t": time.time()}
QUIET_STAGES = ()   # S3 measures quality, not latency


def stage(name):
    STAGE.update(name=name, t=time.time())
    print(f"   .. {name}  [{time.strftime('%H:%M:%S')}]", flush=True)


def _pg_progress():
    try:
        import psycopg
        with psycopg.connect(PgVector.DSN, autocommit=True, connect_timeout=3) as c:
            c.execute("SET statement_timeout = 3000")
            pr = c.execute("SELECT phase, blocks_done, blocks_total, tuples_done, tuples_total "
                           "FROM pg_stat_progress_create_index").fetchall()
            act = c.execute("SELECT state, wait_event_type, wait_event, round(extract(epoch FROM now()-query_start)), "
                            "left(regexp_replace(query, '\\s+', ' ', 'g'), 60) FROM pg_stat_activity "
                            "WHERE state <> 'idle' AND pid <> pg_backend_pid() AND backend_type = 'client backend'").fetchall()
        return f"index build {pr} | active {act}"
    except Exception as e:
        return f"pg n/a ({type(e).__name__})"


def _heartbeat():
    while True:
        time.sleep(HB_PERIOD)
        try:
            name = STAGE["name"]
            if name == "done" or any(q in name for q in QUIET_STAGES):
                continue
            vm = psutil.virtual_memory()
            top = []
            for p in psutil.process_iter(["name", "memory_info"]):
                try:
                    top.append((p.info["memory_info"].rss, p.info["name"]))
                except Exception:
                    pass
            top = ", ".join(f"{n}:{r / 2**30:.1f}G" for r, n in sorted(top, reverse=True)[:4])
            msg = (f"   [hb {time.strftime('%H:%M:%S')}] stage={name} for {(time.time() - STAGE['t']) / 60:.1f} min | "
                   f"RAM {vm.used / 2**30:.1f}/{vm.total / 2**30:.1f} GB | CPU {psutil.cpu_percent(interval=1):.0f}% | "
                   f"disk free {shutil.disk_usage('/').free / 2**30:.0f} GB | top RSS {top}")
            if name.startswith("pgvector"):
                msg += "\n        " + _pg_progress()
            print(msg, flush=True)
        except Exception as e:
            print(f"   [hb] error {e}", flush=True)


import builtins  # noqa: E402
if not getattr(builtins, "_VDB_HB_STARTED", False):   # one heartbeat per kernel, even if the cell is re-run
    builtins._VDB_HB_STARTED = True
    threading.Thread(target=_heartbeat, daemon=True).start()



# ============================================================================ metrics
def build_qrels(qrels_df, queries_df, corpus_doc_ids):
    """returns list (per query, aligned to queries_df order) of {doc_idx: grade}"""
    id2idx = {d: i for i, d in enumerate(corpus_doc_ids)}
    rel = {}
    for qid, did, s in qrels_df.itertuples(index=False):
        if did in id2idx and s > 0:
            rel.setdefault(qid, {})[id2idx[did]] = int(s)
    return [rel.get(q, {}) for q in queries_df["query_id"]]


def ndcg_at(pred, rels, k=10):
    """per-query nDCG@k, linear gain (pytrec_eval / BEIR convention)."""
    out = np.zeros(len(pred), dtype=np.float32)
    for i, (p, g) in enumerate(zip(pred, rels)):
        dcg = sum(g.get(int(d), 0) / math.log2(r + 2) for r, d in enumerate(p[:k]) if d >= 0)
        ideal = sorted(g.values(), reverse=True)[:k]
        idcg = sum(s / math.log2(r + 2) for r, s in enumerate(ideal))
        out[i] = dcg / idcg if idcg > 0 else 0.0
    return out


def mrr_at(pred, rels, k=10):
    out = np.zeros(len(pred), dtype=np.float32)
    for i, (p, g) in enumerate(zip(pred, rels)):
        for r, d in enumerate(p[:k]):
            if g.get(int(d), 0) > 0:
                out[i] = 1.0 / (r + 1)
                break
    return out


def recall_rel(pred, rels, k):
    out = np.zeros(len(pred), dtype=np.float32)
    for i, (p, g) in enumerate(zip(pred, rels)):
        if g:
            out[i] = len({int(d) for d in p[:k]} & set(g)) / len(g)
    return out


def recall_gt(pred, gt, k):
    out = np.zeros(len(pred), dtype=np.float32)
    for i, (p, g) in enumerate(zip(pred, gt)):
        gg = {int(x) for x in g[:k] if x >= 0}
        out[i] = len({int(d) for d in p[:k]} & gg) / len(gg) if gg else 1.0
    return out


# ============================================================================ BM25 (client side, shared by Qdrant/pgvector/reference)
STOP = set("""a an and are as at be but by for if in into is it no not of on or such that the their then there these
they this to was will with from has have had he her his i its me my our she so than them we were what when where which
who whom why you your yours do does did can could should would may might been being am""".split())
TOK = re.compile(r"[a-z0-9]+")


def tokenize(s):
    return [t for t in TOK.findall(s.lower()) if t not in STOP and len(t) > 1]


class BM25:
    def __init__(self, docs):
        toks = [tokenize(d) for d in docs]
        self.vocab, df = {}, Counter()
        for t in toks:
            df.update(set(t))
        for w in df:
            self.vocab[w] = len(self.vocab)
        N = len(docs)
        self.idf = np.zeros(len(self.vocab), dtype=np.float32)
        for w, n in df.items():
            self.idf[self.vocab[w]] = math.log(1 + (N - n + 0.5) / (n + 0.5))
        avgdl = max(1e-9, np.mean([len(t) for t in toks]))
        rows, cols, vals = [], [], []
        for i, t in enumerate(toks):
            c = Counter(t)
            dl = len(t)
            for w, tf in c.items():
                j = self.vocab[w]
                rows.append(i)
                cols.append(j)
                vals.append(self.idf[j] * tf * (BM25_K1 + 1) / (tf + BM25_K1 * (1 - BM25_B + BM25_B * dl / avgdl)))
        self.M = sp.csr_matrix((np.array(vals, np.float32), (rows, cols)), shape=(N, len(self.vocab)))
        self.dim = len(self.vocab)

    def doc_sparse(self, i):
        s, e = self.M.indptr[i], self.M.indptr[i + 1]
        return self.M.indices[s:e].tolist(), self.M.data[s:e].astype(float).tolist()

    def query_terms(self, q):
        return sorted({self.vocab[w] for w in tokenize(q) if w in self.vocab})

    def search(self, q, k=100):
        idx = self.query_terms(q)
        if not idx:
            return []
        sc = np.asarray(self.M[:, idx].sum(axis=1)).ravel()
        nz = np.flatnonzero(sc > 0)
        if len(nz) == 0:
            return []
        top = nz[np.argsort(-sc[nz], kind="stable")[:k]]
        return top.tolist()


def rrf(lists, k_rrf=RRF_K, k=10):
    sc = {}
    for lst in lists:
        for r, d in enumerate(lst):
            if d >= 0:
                sc[d] = sc.get(d, 0.0) + 1.0 / (k_rrf + r + 1)
    return [d for d, _ in sorted(sc.items(), key=lambda x: -x[1])[:k]]


# ============================================================================ adapters
class Qdrant:
    name = "qdrant"
    coll = "bench"
    fst = None          # hnsw full_scan_threshold (KB); None = Qdrant default (10000). Set by run_job BEFORE start().

    def start(self):
        b = install_binary("qdrant/qdrant", VERSIONS["QDRANT_TAG"], r"x86_64-unknown-linux-gnu\.tar\.gz$", "qdrant")
        self.data = ROOT / "data" / "qdrant"
        shutil.rmtree(self.data, ignore_errors=True)
        self.data.mkdir(parents=True)
        env = dict(os.environ, QDRANT__STORAGE__STORAGE_PATH=str(self.data / "storage"),
                   QDRANT__STORAGE__SNAPSHOTS_PATH=str(self.data / "snapshots"), QDRANT__SERVICE__HOST="127.0.0.1",
                   QDRANT__SERVICE__HTTP_PORT="6333", QDRANT__SERVICE__GRPC_PORT="6334",
                   QDRANT__TELEMETRY_DISABLED="true", QDRANT__LOG_LEVEL="WARN")
        self.log = open(OUT / "qdrant.log", "a")
        self.proc = subprocess.Popen([str(b)], cwd=str(self.data), env=env, stdout=self.log, stderr=subprocess.STDOUT)
        wait_http("http://127.0.0.1:6333/readyz")
        from qdrant_client import QdrantClient, models
        self.m = models
        self.c = QdrantClient(host="127.0.0.1", port=6333, grpc_port=6334, prefer_grpc=True, timeout=3600)
        self.ef, self.rrf_mode = COMMON["EF_INIT"], None

    def load(self, X, attrs, bm25, texts):
        m = self.m
        if self.c.collection_exists(self.coll):
            self.c.delete_collection(self.coll)
        kw = dict(vectors_config={"dense": m.VectorParams(size=X.shape[1], distance=m.Distance.COSINE,
                                                          hnsw_config=m.HnswConfigDiff(m=COMMON["M"], ef_construct=COMMON["EF_C"],
                                                                                       full_scan_threshold=self.fst))},
                  optimizers_config=m.OptimizersConfigDiff(indexing_threshold=0))
        if bm25 is not None:
            kw["sparse_vectors_config"] = {"bm25": m.SparseVectorParams()}   # weights already include IDF
        self.c.create_collection(self.coll, **kw)
        for a in attrs:
            self.c.create_payload_index(self.coll, a, field_schema=m.PayloadSchemaType.INTEGER)

        def gen():
            for i in range(len(X)):
                vec = {"dense": X[i].tolist()}
                if bm25 is not None:
                    ind, val = bm25.doc_sparse(i)
                    vec["bm25"] = m.SparseVector(indices=ind, values=val)
                yield m.PointStruct(id=i, vector=vec, payload={a: int(v[i]) for a, v in attrs.items()})
        self.c.upload_points(self.coll, points=gen(), batch_size=256, parallel=1, wait=True)
        # indexing_threshold=1 KB (not the 10000 KB default): Qdrant otherwise leaves segments < 10 MB unindexed and
        # brute-forces them, so small corpora (scifact, nfcorpus) would get EXACT search and no ef effect. E3 measures
        # HNSW quality vs ef, so every segment must be indexed (Qdrant's small-collection default is noted in the paper).
        self.c.update_collection(self.coll, optimizers_config=m.OptimizersConfigDiff(indexing_threshold=1))
        time.sleep(1)
        # indexed_vectors_count counts dense + non-empty sparse vectors (a few FiQA/TREC-COVID documents have no
        # BM25 terms: 39 and 12, the same counts as in Q1)
        need = len(X) + (int((np.diff(bm25.M.indptr[:len(X) + 1]) > 0).sum()) if bm25 is not None else 0)
        green, t0 = 0, time.time()
        while green < 3:
            info = self.c.get_collection(self.coll)
            ok = info.status == m.CollectionStatus.GREEN and (info.indexed_vectors_count or 0) >= need
            green = green + 1 if ok else 0
            if time.time() - t0 > 3600:
                raise RuntimeError(f"Qdrant indexing incomplete: {info.indexed_vectors_count}/{need}")
            time.sleep(0.5)
        vp = info.config.params.vectors
        vp = vp.get("dense") if isinstance(vp, dict) else vp
        fst = (vp.hnsw_config.full_scan_threshold if vp.hnsw_config and vp.hnsw_config.full_scan_threshold is not None
               else info.config.hnsw_config.full_scan_threshold)
        if self.fst is not None and fst != self.fst:
            raise RuntimeError(f"Qdrant full_scan_threshold is {fst} KB, expected {self.fst} KB")
        self.indexed = dict(qdrant_indexed_vectors=info.indexed_vectors_count, qdrant_points=info.points_count,
                            qdrant_segments=info.segments_count, qdrant_fst_kb=fst)   # read back from the server
        print(f"   [qdrant] indexed {self.indexed}", flush=True)

    def set_ef(self, ef):
        self.ef = ef

    def _flt(self, flt):
        m = self.m
        return None if flt is None else m.Filter(must=[m.FieldCondition(key=flt[0], match=m.MatchValue(value=int(flt[1])))])

    def dense(self, q, k, flt=None):
        m = self.m
        r = self.c.query_points(self.coll, query=q.tolist(), using="dense", limit=k, query_filter=self._flt(flt),
                                search_params=m.SearchParams(hnsw_ef=self.ef), with_payload=False)
        return [int(p.id) for p in r.points]

    def lexical(self, qtext, qterms, k):
        m = self.m
        if not qterms:
            return []
        r = self.c.query_points(self.coll, query=m.SparseVector(indices=qterms, values=[1.0] * len(qterms)),
                                using="bm25", limit=k, with_payload=False)
        return [int(p.id) for p in r.points]

    def hybrid(self, q, qtext, qterms, k):
        m = self.m
        pre = [m.Prefetch(query=q.tolist(), using="dense", limit=100, params=m.SearchParams(hnsw_ef=HYB_EF))]
        if qterms:
            pre.append(m.Prefetch(query=m.SparseVector(indices=qterms, values=[1.0] * len(qterms)), using="bm25", limit=100))
        modes = [self.rrf_mode] if self.rrf_mode else ["rrf_k60", "rrf_default"]
        for mode in modes:
            try:
                if mode == "rrf_k60":
                    fq = m.RrfQuery(rrf=m.Rrf(k=RRF_K))
                else:
                    fq = m.FusionQuery(fusion=m.Fusion.RRF)
                r = self.c.query_points(self.coll, prefetch=pre, query=fq, limit=k, with_payload=False)
                self.rrf_mode = mode
                return [int(p.id) for p in r.points]
            except Exception:
                if mode == modes[-1]:
                    raise
        return []

    def info(self):
        return dict(rrf_mode=self.rrf_mode, **getattr(self, "indexed", {}))

    def stop(self):
        try:
            self.c.close()
        except Exception:
            pass
        self.proc.terminate()
        try:
            self.proc.wait(60)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.log.close()
        shutil.rmtree(self.data, ignore_errors=True)


class Weaviate:
    name = "weaviate"
    coll = "Bench"

    def start(self):
        b = install_binary("weaviate/weaviate", VERSIONS["WEAVIATE_TAG"], r"linux-amd64\.tar\.gz$", "weaviate")
        self.data = ROOT / "data" / "weaviate"
        shutil.rmtree(self.data, ignore_errors=True)
        self.data.mkdir(parents=True)
        env = dict(os.environ, PERSISTENCE_DATA_PATH=str(self.data), AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED="true",
                   ENABLE_MODULES="", CLUSTER_HOSTNAME="node1", GRPC_PORT="50051", RAFT_BOOTSTRAP_EXPECT="1",
                   DISABLE_TELEMETRY="true", QUERY_DEFAULTS_LIMIT="25", LOG_LEVEL="warning", ASYNC_INDEXING="false",
                   DISK_USE_WARNING_PERCENTAGE="99", DISK_USE_READONLY_PERCENTAGE="100")
        self.log = open(OUT / "weaviate.log", "a")
        self.proc = subprocess.Popen([str(b), "--host", "127.0.0.1", "--port", "8080", "--scheme", "http"],
                                     cwd=str(self.data), env=env, stdout=self.log, stderr=subprocess.STDOUT)
        wait_http("http://127.0.0.1:8080/v1/.well-known/ready")
        import weaviate
        from weaviate.classes.init import AdditionalConfig, Timeout
        self.c = weaviate.connect_to_local(host="127.0.0.1", port=8080, grpc_port=50051,
                                           additional_config=AdditionalConfig(timeout=Timeout(init=60, query=600, insert=3600)))

    def load(self, X, attrs, bm25, texts):
        import uuid
        from weaviate.classes.config import Configure, Property, DataType, VectorDistances, Tokenization
        if self.c.collections.exists(self.coll):
            self.c.collections.delete(self.coll)
        props = [Property(name=a, data_type=DataType.INT) for a in attrs]
        if texts is not None:
            props.append(Property(name="body", data_type=DataType.TEXT, tokenization=Tokenization.WORD))
        hnsw = Configure.VectorIndex.hnsw(distance_metric=VectorDistances.COSINE, ef_construction=COMMON["EF_C"],
                                          max_connections=COMMON["M"], ef=COMMON["EF_INIT"])
        self.c.collections.create(self.coll, properties=props,
                                  vector_config=Configure.Vectors.self_provided(vector_index_config=hnsw),
                                  inverted_index_config=Configure.inverted_index(bm25_k1=BM25_K1, bm25_b=BM25_B))
        self.col = self.c.collections.get(self.coll)
        with self.col.batch.fixed_size(batch_size=500, concurrent_requests=4) as b:
            for i in range(len(X)):
                pr = {a: int(v[i]) for a, v in attrs.items()}
                if texts is not None:
                    pr["body"] = texts[i]
                b.add_object(properties=pr, vector={"default": X[i].tolist()}, uuid=uuid.UUID(int=i + 1))
        if self.col.batch.failed_objects:
            raise RuntimeError(f"{len(self.col.batch.failed_objects)} failed: {self.col.batch.failed_objects[0]}")
        while self.col.aggregate.over_all(total_count=True).total_count < len(X):
            time.sleep(0.5)

    def set_ef(self, ef):
        from weaviate.classes.config import Reconfigure
        self.col.config.update(vector_config=Reconfigure.Vectors.update(
            name="default", vector_index_config=Reconfigure.VectorIndex.hnsw(ef=int(ef))))
        time.sleep(1.0)

    def dense(self, q, k, flt=None):
        from weaviate.classes.query import Filter
        f = None if flt is None else Filter.by_property(flt[0]).equal(int(flt[1]))
        r = self.col.query.near_vector(near_vector=q.tolist(), limit=k, filters=f, return_properties=[])
        return [o.uuid.int - 1 for o in r.objects]

    def lexical(self, qtext, qterms, k):
        r = self.col.query.bm25(query=qtext, limit=k, return_properties=[])
        return [o.uuid.int - 1 for o in r.objects]

    def hybrid(self, q, qtext, qterms, k):
        from weaviate.classes.query import HybridFusion
        r = self.col.query.hybrid(query=qtext, vector=q.tolist(), alpha=0.5, limit=k,
                                  fusion_type=HybridFusion.RANKED, return_properties=[])
        return [o.uuid.int - 1 for o in r.objects]

    def info(self):
        try:
            vi = self.col.config.get().vector_config["default"].vector_index_config
            return dict(filter_strategy=str(getattr(vi, "filter_strategy", None)),
                        flat_search_cutoff=getattr(vi, "flat_search_cutoff", None))
        except Exception:
            return {}

    def stop(self):
        try:
            self.c.close()
        except Exception:
            pass
        self.proc.terminate()
        try:
            self.proc.wait(120)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.log.close()
        shutil.rmtree(self.data, ignore_errors=True)


class PgVector:
    name = "pgvector"
    DSN = "host=127.0.0.1 port=5432 user=postgres dbname=postgres"
    _installed = False

    @classmethod
    def install(cls):
        if cls._installed:
            return
        print("   installing PostgreSQL + pgvector ...", flush=True)
        def pg_bins():
            return sorted((b for b in glob.glob("/usr/lib/postgresql/*/bin") if Path(b, "pg_config").exists()
                           and Path(b, "postgres").exists()), key=lambda p: int(Path(p).parent.name))
        if not pg_bins():   # idempotent: a re-run in the same Kaggle session skips apt entirely
            apt = ("DEBIAN_FRONTEND=noninteractive apt-get -y -qq -o DPkg::Lock::Timeout=600 -o Acquire::Retries=3 "
                   "-o Acquire::http::Timeout=60 -o Dpkg::Options::=--force-confdef -o Dpkg::Options::=--force-confold")
            sh("DEBIAN_FRONTEND=noninteractive dpkg --configure -a", check=False, timeout=600)  # heal interrupted runs
            sh(f"{apt} update || echo 'apt update had errors; continuing with cached lists'", timeout=900)
            sh(f"{apt} install postgresql postgresql-contrib postgresql-server-dev-all build-essential git "
               f"&& apt-get clean", timeout=1800)
        else:
            print("   PostgreSQL already installed; skipping apt", flush=True)
        bins = pg_bins()
        cls.pg_major = Path(bins[-1]).parent.name
        pgc = Path(bins[-1]) / "pg_config"
        ver = VERSIONS['PGVECTOR_TAG'].lstrip("v")
        ctl = Path(sh(f"{pgc} --sharedir")) / "extension" / f"vector--{ver}.sql"
        if not ctl.exists():
            src = ROOT / "pgvector-src"
            shutil.rmtree(src, ignore_errors=True)
            sh(f"git clone -q --depth 1 --branch {VERSIONS['PGVECTOR_TAG']} https://github.com/pgvector/pgvector.git {src}",
               timeout=600)
            sh(f"cd {src} && make -s PG_CONFIG={pgc} && make -s install PG_CONFIG={pgc}", timeout=1800)
        else:
            print(f"   pgvector {ver} already built; skipping", flush=True)
        etc = Path(f"/etc/postgresql/{cls.pg_major}/main")
        if not etc.exists():
            sh(f"pg_createcluster {cls.pg_major} main")
        hba = etc / "pg_hba.conf"
        hba.write_text(re.sub(r"\b(peer|md5|scram-sha-256)\b", "trust", hba.read_text()))
        mem = psutil.virtual_memory().total / 2**30
        cls.settings = {"listen_addresses": "'127.0.0.1'", "port": "5432",
                        "shared_buffers": f"'{max(1, int(mem * 0.25))}GB'",
                        "maintenance_work_mem": f"'{max(1, int(mem * 0.25))}GB'",
                        "max_parallel_maintenance_workers": str(max(1, (os.cpu_count() or 2) - 1)),
                        "max_parallel_workers": str(os.cpu_count() or 2),
                        "effective_cache_size": f"'{max(1, int(mem * 0.5))}GB'",
                        "work_mem": "'64MB'", "jit": "off"}
        (etc / "conf.d").mkdir(exist_ok=True)
        (etc / "conf.d" / "bench.conf").write_text("".join(f"{k} = {v}\n" for k, v in cls.settings.items()))
        conf = etc / "postgresql.conf"
        if "include_dir = 'conf.d'" not in conf.read_text():
            conf.write_text(conf.read_text() + "\ninclude_dir = 'conf.d'\n")
        cls._installed = True

    def start(self):
        self.install()
        sh(f"pg_ctlcluster {self.pg_major} main restart || pg_ctlcluster {self.pg_major} main start")
        import psycopg
        from pgvector.psycopg import register_vector
        for _ in range(120):
            try:
                # prepare_threshold=None: never server-side prepare, so every query gets a custom plan that sees
                # the real LIMIT and filter value (a generic plan for "LIMIT $n" falls back to a sequential scan,
                # and a generic plan cannot see filter selectivity).
                self.conn = psycopg.connect(self.DSN, autocommit=True, prepare_threshold=None)
                break
            except Exception:
                time.sleep(0.5)
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS pg_prewarm")
        register_vector(self.conn)
        self.conn.execute("DROP TABLE IF EXISTS items")
        self.iter_mode = "off"

    def load(self, X, attrs, bm25, texts):
        from pgvector import SparseVector
        c = self.conn
        acols = list(attrs)
        cols = ", ".join(f"{a} smallint" for a in acols)
        spcol = f", sp sparsevec({bm25.dim})" if bm25 is not None else ""
        c.execute(f"CREATE TABLE items (id int PRIMARY KEY, {cols}, emb vector({X.shape[1]}){spcol})")
        names = ["id"] + acols + ["emb"] + (["sp"] if bm25 is not None else [])
        types = ["int4"] + ["int2"] * len(acols) + ["vector"] + (["sparsevec"] if bm25 is not None else [])
        with c.cursor() as cur:
            with cur.copy(f"COPY items ({', '.join(names)}) FROM STDIN WITH (FORMAT BINARY)") as cp:
                cp.set_types(types)
                for i in range(len(X)):
                    row = [i] + [int(attrs[a][i]) for a in acols] + [np.asarray(X[i], dtype=np.float32)]
                    if bm25 is not None:
                        ind, val = bm25.doc_sparse(i)
                        row.append(SparseVector(dict(zip(ind, val)), bm25.dim))
                    cp.write_row(row)
        c.execute(f"CREATE INDEX items_hnsw ON items USING hnsw (emb vector_cosine_ops) "
                  f"WITH (m = {COMMON['M']}, ef_construction = {COMMON['EF_C']})")
        for a in acols:
            c.execute(f"CREATE INDEX ON items ({a})")
        c.execute("VACUUM (ANALYZE) items")
        c.execute("SELECT pg_prewarm('items'), pg_prewarm('items_hnsw')")
        self.has_sp = bm25 is not None
        self.dim_sp = bm25.dim if bm25 is not None else None

    def set_ef(self, ef):
        self.conn.execute(f"SET hnsw.ef_search = {int(ef)}")

    def set_iter(self, mode):
        self.conn.execute(f"SET hnsw.iterative_scan = {mode}")
        self.iter_mode = mode

    def set_force(self, on):
        """E3/E5 measure HNSW at a fixed ef, so a sequential (exact) scan is disabled; E4 leaves the planner
        free (choosing between HNSW, the attribute B-tree and a scan is part of filtered-search behaviour)."""
        self.conn.execute("SET enable_seqscan = off" if on else "RESET enable_seqscan")
        self.force = on

    def plan_of(self, q, k, flt=None):
        lit = "'[" + ",".join(f"{float(x):.6f}" for x in q) + "]'"
        where = f"WHERE {flt[0]} = {int(flt[1])} " if flt else ""
        p = "\n".join(r[0] for r in self.conn.execute(
            f"EXPLAIN SELECT id FROM items {where}ORDER BY emb <=> {lit} LIMIT {int(k)}").fetchall())
        if "items_hnsw" in p:
            return "hnsw"
        if "Index" in p or "Bitmap" in p:
            return "btree"
        return "seqscan"

    def dense(self, q, k, flt=None):
        if flt is None:
            rows = self.conn.execute(f"SELECT id FROM items ORDER BY emb <=> %s LIMIT {int(k)}", (q,)).fetchall()
        else:
            rows = self.conn.execute(f"SELECT id FROM items WHERE {flt[0]} = {int(flt[1])} "
                                     f"ORDER BY emb <=> %s LIMIT {int(k)}", (q,)).fetchall()
        return [r[0] for r in rows]

    def _sq(self, qterms):
        from pgvector import SparseVector
        return SparseVector({t: 1.0 for t in qterms}, self.dim_sp)

    def lexical(self, qtext, qterms, k):
        if not qterms:
            return []
        rows = self.conn.execute("SELECT id FROM (SELECT id, sp <#> %s AS sc FROM items) x WHERE sc < 0 "
                                 f"ORDER BY sc LIMIT {int(k)}", (self._sq(qterms),)).fetchall()
        return [r[0] for r in rows]

    def hybrid(self, q, qtext, qterms, k):
        self.conn.execute(f"SET hnsw.ef_search = {HYB_EF}")
        if not qterms:
            return self.dense(q, k)
        sql = f"""
        WITH d AS (SELECT id, ROW_NUMBER() OVER (ORDER BY dist) AS r
                   FROM (SELECT id, emb <=> %(q)s AS dist FROM items ORDER BY dist LIMIT 100) s),
             l AS (SELECT id, ROW_NUMBER() OVER (ORDER BY sc) AS r
                   FROM (SELECT id, sc FROM (SELECT id, sp <#> %(s)s AS sc FROM items) x
                         WHERE sc < 0 ORDER BY sc LIMIT 100) s2)
        SELECT COALESCE(d.id, l.id) AS id,
               COALESCE(1.0 / ({RRF_K} + d.r), 0) + COALESCE(1.0 / ({RRF_K} + l.r), 0) AS score
        FROM d FULL OUTER JOIN l ON d.id = l.id ORDER BY score DESC, id LIMIT {int(k)}"""
        return [r[0] for r in self.conn.execute(sql, {"q": q, "s": self._sq(qterms)}).fetchall()]

    def info(self):
        return dict(iterative_scan=self.iter_mode, force_index=getattr(self, "force", False),
                    max_scan_tuples=self.conn.execute("SHOW hnsw.max_scan_tuples").fetchone()[0])

    def stop(self):
        try:
            self.conn.execute("DROP TABLE IF EXISTS items")
            self.conn.close()
        except Exception:
            pass
        sh(f"pg_ctlcluster {self.pg_major} main stop", check=False)


ADAPTERS = {"qdrant": Qdrant, "weaviate": Weaviate, "pgvector": PgVector}


# ============================================================================ data
def data_root():
    if Path(DATA_HINT, "meta.json").exists():
        return Path(DATA_HINT)
    hits = glob.glob("/kaggle/input/**/meta.json", recursive=True)
    if not hits:
        raise FileNotFoundError("S1 dataset (meta.json) not found under /kaggle/input")
    return Path(hits[0]).parent


# Qdrant's hnsw full_scan_threshold (default 10000 KB) also governs UNFILTERED search: a segment whose vectors
# take less than that is brute-forced even when its HNSW graph exists (scifact/nfcorpus x bge-small in Q1 gave
# exact results at every ef). E3/E5 compare HNSW at a fixed ef, so unfiltered jobs lower it to the minimum (10 KB).
# E4 jobs keep the default: there it is Qdrant's cardinality switch between payload-index and graph search.
QDRANT_FST_DEFAULT_KB, QDRANT_FST_UNFILTERED = 10000, 10


def qdrant_fst(job):
    """True when lowering full_scan_threshold changes Qdrant's behaviour for this job (rerun needed)."""
    if job in P["E4"]:
        return False
    n, d = np.load(data_root() / job[0] / job[1] / "docs.npy", mmap_mode="r").shape
    return n * d * 4 / 1024 < QDRANT_FST_DEFAULT_KB


def load_job(dataset, model, need_text, need_filter):
    root = data_root() / dataset
    cols = ["doc_id", "a2", "a10", "a100", "a1000"] + (["title", "text"] if need_text else [])
    corpus = pd.read_parquet(root / "corpus.parquet", columns=cols)
    queries = pd.read_parquet(root / "queries.parquet")
    qrels = pd.read_parquet(root / "qrels.parquet")
    X = np.load(root / model / "docs.npy", mmap_mode="r")
    Q = np.load(root / model / "queries.npy")
    GT = np.load(root / model / "gt_top100.npy")
    FGT = {c: np.load(root / model / f"gt_filt_a{c}.npy") for c in P["FILT_SEL"]} if need_filter else {}
    if len(Q) > P["QUERY_CAP"]:
        sel = np.sort(np.random.default_rng(SEED).choice(len(Q), P["QUERY_CAP"], replace=False))
        Q, GT, queries = Q[sel], GT[sel], queries.iloc[sel].reset_index(drop=True)
        FGT = {c: v[sel] for c, v in FGT.items()}
    rels = build_qrels(qrels, queries, corpus["doc_id"].tolist())
    texts = ((corpus["title"].fillna("") + " " + corpus["text"].fillna("")).str.strip().tolist() if need_text else None)
    attrs = {f"a{c}": corpus[f"a{c}"].values for c in (2, 10, 100, 1000)}
    return dict(X=X, Q=np.ascontiguousarray(Q, dtype=np.float32), GT=GT, FGT=FGT, rels=rels, queries=queries,
                texts=texts, attrs=attrs)


# ============================================================================ experiments
def timed(fn, items):
    out, lat = [], []
    for it in items:
        t = time.perf_counter()
        out.append(fn(*it))
        lat.append(time.perf_counter() - t)
    return out, lat


def exp_quality(db, J, base, store):
    Q, GT, rels = J["Q"], J["GT"], J["rels"]
    if db.name == "pgvector":         # record where the planner alone would leave the HNSW index, then force it
        db.set_force(False)
        for k, efs in ((10, P["EF10"]), (100, P["EF100"])):
            for ef in efs:
                db.set_ef(ef)
                emit(dict(base, exp="pg_plan", k=k, ef=ef, plan=db.plan_of(Q[0], k)))
        db.set_force(True)
        db.set_ef(max(P["EF10"]))
        assert db.plan_of(Q[0], 10) == "hnsw", "pgvector does not use HNSW even with enable_seqscan=off"
    db.set_ef(max(P["EF10"] + P["EF100"]))
    for q in Q:                                      # warm-up pass
        db.dense(q, 10)
    for k, efs in ((10, P["EF10"]), (100, P["EF100"])):
        for ef in efs:
            db.set_ef(ef)
            pred, lat = timed(lambda q: db.dense(q, k), [(q,) for q in Q])
            nd, mr = ndcg_at(pred, rels, 10), mrr_at(pred, rels, 10)
            rr, rg = recall_rel(pred, rels, k), recall_gt(pred, GT, k)
            emit(dict(base, exp="E3", k=k, ef=ef, ndcg10=float(nd.mean()), mrr10=float(mr.mean()),
                      recall_rel=float(rr.mean()), recall_gt=float(rg.mean()), qps=len(Q) / sum(lat), **lat_stats(lat)))
            store[f"E3_k{k}_ef{ef}_ids"] = pad(pred, k)
            store[f"E3_k{k}_ef{ef}_ndcg10"] = nd
            store[f"E3_k{k}_ef{ef}_recallgt"] = rg


def exp_filter(db, J, base, store):
    Q, queries = J["Q"], J["queries"]
    modes = ["off", "relaxed_order"] if db.name == "pgvector" else [None]
    if db.name == "pgvector":
        db.set_force(False)
    for mode in modes:
        if mode:
            db.set_iter(mode)
        for c in P["FILT_SEL"]:
            qv, gt = queries[f"q_a{c}"].values, J["FGT"][c]
            for ef in P["FILT_EF"]:
                db.set_ef(ef)
                for i in range(min(50, len(Q))):     # warm-up for this filter path
                    db.dense(Q[i], 10, (f"a{c}", qv[i]))
                pred, lat = timed(lambda q, v: db.dense(q, 10, (f"a{c}", v)), list(zip(Q, qv)))
                ok = all(int(J["attrs"][f"a{c}"][d]) == int(v) for p, v in zip(pred, qv) for d in p)
                rg = recall_gt(pred, gt, 10)
                extra = {"pg_plan": db.plan_of(Q[0], 10, (f"a{c}", qv[0]))} if db.name == "pgvector" else {}
                emit(dict(base, exp="E4", **extra, selectivity=1.0 / c, card=c, ef=ef, pg_iterative_scan=mode,
                          recall_gt=float(rg.mean()), mean_returned=float(np.mean([len(p) for p in pred])),
                          filter_correct=bool(ok), qps=len(Q) / sum(lat), **lat_stats(lat), **db.info()))
                store[f"E4_a{c}_ef{ef}_{mode}_recall"] = rg
    if db.name == "pgvector":
        db.set_iter("off")


def exp_hybrid(db, J, base, store, qterms, qtexts):
    Q, rels = J["Q"], J["rels"]
    if db.name == "pgvector":         # dense CTE must use HNSW; the sparsevec scan has no index either way
        db.set_force(True)
    db.set_ef(HYB_EF)
    for i in range(min(50, len(Q))):
        db.hybrid(Q[i], qtexts[i], qterms[i], 10)
    for label, fn in (("lexical", lambda q, t, s: db.lexical(t, s, 100)),
                      ("hybrid", lambda q, t, s: db.hybrid(q, t, s, 10)),
                      ("dense", lambda q, t, s: db.dense(q, 10))):
        pred, lat = timed(fn, list(zip(Q, qtexts, qterms)))
        nd = ndcg_at(pred, rels, 10)
        emit(dict(base, exp="E5", method=label, ef=HYB_EF, ndcg10=float(nd.mean()),
                  mrr10=float(mrr_at(pred, rels, 10).mean()), recall_rel10=float(recall_rel(pred, rels, 10).mean()),
                  recall_rel100=float(recall_rel(pred, rels, 100).mean()) if label == "lexical" else None,
                  qps=len(Q) / sum(lat), **lat_stats(lat), **db.info()))
        store[f"E5_{label}_ids"] = pad(pred, 100 if label == "lexical" else 10)
        store[f"E5_{label}_ndcg10"] = nd


def reference_rows(job, J, bm25, qtexts, store):
    """Exact references independent of any database: exact dense, exact BM25, exact-hybrid RRF."""
    dataset, model = job
    base = dict(dataset=dataset, model=model, db="exact", N=len(J["X"]), D=J["X"].shape[1], nq=len(J["Q"]))
    dense100 = [list(r) for r in J["GT"]]
    nd = ndcg_at(dense100, J["rels"], 10)
    emit(dict(base, exp="E3", k=10, ef="exact", ndcg10=float(nd.mean()), mrr10=float(mrr_at(dense100, J["rels"]).mean()),
              recall_rel=float(recall_rel(dense100, J["rels"], 10).mean()), recall_gt=1.0))
    emit(dict(base, exp="E3", k=100, ef="exact", ndcg10=float(nd.mean()),
              recall_rel=float(recall_rel(dense100, J["rels"], 100).mean()), recall_gt=1.0))
    store["exact_ndcg10"] = nd
    if bm25 is not None:
        lex = [bm25.search(t, 100) for t in qtexts]
        hyb = [rrf([d, l], RRF_K, 10) for d, l in zip(dense100, lex)]
        for label, pred in (("lexical", lex), ("hybrid", hyb), ("dense", dense100)):
            ndl = ndcg_at(pred, J["rels"], 10)
            emit(dict(base, exp="E5", method=label, ef="exact", ndcg10=float(ndl.mean()),
                      mrr10=float(mrr_at(pred, J["rels"], 10).mean()),
                      recall_rel10=float(recall_rel(pred, J["rels"], 10).mean()),
                      recall_rel100=float(recall_rel(pred, J["rels"], 100).mean()) if label == "lexical" else None))
            store[f"E5_{label}_ndcg10"] = ndl
            store[f"E5_{label}_ids"] = pad(pred, 100 if label == "lexical" else 10)


def run_job(job):
    dataset, model = job
    do3, do4, do5 = job in P["E3"], job in P["E4"], job in P["E5"]
    print(f"\n>> Job {dataset}/{model}  E3={do3} E4={do4} E5={do5}", flush=True)
    def unit_key(dbn):
        k = f"{PRESET}|{dataset}|{model}|{dbn}"
        return k + "|fst2" if dbn == "qdrant" and qdrant_fst(job) else k
    rkey = f"{PRESET}|{dataset}|{model}|exact"
    if (P.get("SKIP_REF") or rkey in done_set()) and all(unit_key(d) in done_set() for d in DBS):
        print("   all units done; skipping", flush=True)
        return
    J = load_job(dataset, model, need_text=do5, need_filter=do4)
    bm25, qterms, qtexts = None, None, None
    if do5:
        t = time.time()
        bm25 = BM25(J["texts"])
        qtexts = J["queries"]["text"].tolist()
        qterms = [bm25.query_terms(s) for s in qtexts]
        print(f"   BM25: vocab={bm25.dim:,} nnz={bm25.M.nnz:,} ({time.time()-t:.0f}s)", flush=True)
    if rkey not in done_set() and not P.get("SKIP_REF"):    # QNP: exact references already exist from Q1/Q2
        store = {}
        reference_rows(job, J, bm25, qtexts, store)
        np.savez_compressed(OUT / f"q_{dataset}_{model}_exact.npz", **store)
        mark_done(rkey)
    for dbn in DBS:
        key = unit_key(dbn)
        if key in done_set():
            continue
        base = dict(dataset=dataset, model=model, db=dbn, N=len(J["X"]), D=J["X"].shape[1], nq=len(J["Q"]))
        db, store = ADAPTERS[dbn](), {}
        if dbn == "qdrant" and not do4:
            db.fst = QDRANT_FST_UNFILTERED
            base["qdrant_fst_kb"] = db.fst
        if dbn == "qdrant":
            base["qdrant_payload_index"] = not P.get("NO_PAYLOAD", False)
        attrs = {} if P.get("NO_PAYLOAD") else J["attrs"]          # QNP: no payload, no payload indexes
        try:
            stage(f"{dbn}: start")
            db.start()
            t = time.time()
            stage(f"{dbn}: load + index {len(J['X'])} vectors")
            db.load(J["X"], attrs, bm25, J["texts"] if do5 else None)
            emit(dict(base, exp="load", load_s=time.time() - t, with_sparse=do5, **getattr(db, "indexed", {})))
            print(f"   [{dbn}] loaded in {time.time()-t:.0f}s", flush=True)
            if do3:
                stage(f"{dbn}: E3 quality")
                exp_quality(db, J, base, store)
            if do4:
                stage(f"{dbn}: E4 filter")
                exp_filter(db, J, base, store)
            if do5:
                stage(f"{dbn}: E5 hybrid")
                exp_hybrid(db, J, base, store, qterms, qtexts)
            np.savez_compressed(OUT / f"q_{dataset}_{model}_{dbn}.npz", **store)
            mark_done(key)
        except Exception as e:
            tb = traceback.format_exc()
            print(tb, flush=True)
            emit(dict(base, exp="error", error=f"{type(e).__name__}: {e}", traceback=tb))
        finally:
            try:
                db.stop()
            except Exception as e:
                print(f"   stop() failed: {e}", flush=True)


def summarize():
    if not ROWS.exists():
        return
    df = pd.DataFrame([json.loads(l) for l in open(ROWS)])
    if "preset" in df:                       # summarise this preset only (rows.jsonl accumulates across presets)
        df = df[df.preset == PRESET]
    if "qdrant_fst_kb" in df:                # a unit rerun with the lowered threshold supersedes its earlier rows
        idx = df.set_index(["dataset", "model", "db"]).index      # (incl. the 1st rerun, where it was not applied)
        rerun = idx[df.qdrant_fst_kb.notna().values].unique()
        df = df[~(idx.isin(rerun) & df.qdrant_fst_kb.isna().values)]
    if not len(df):
        return
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 500)
    e3 = df[df.exp == "E3"]
    if len(e3):
        e3 = e3.assign(ef=e3.ef.map(lambda v: f"{int(v):04d}" if isinstance(v, (int, float)) else str(v)))
        t = e3[e3.k == 10].pivot_table(index=["dataset", "model", "ef"], columns="db",
                                       values=["ndcg10", "recall_gt"], aggfunc="mean").round(4)
        t.to_csv(OUT / "summary_E3.csv")
        print("\n===== E3: nDCG@10 and recall@10 (vs exact) by ef =====\n", t.to_string())
    e4 = df[df.exp == "E4"]
    if len(e4):
        e4 = e4.assign(variant=e4.db + e4.pg_iterative_scan.fillna("").map(lambda s: f"[{s}]" if s else ""))
        t = e4.pivot_table(index=["dataset", "card", "ef"], columns="variant", values=["recall_gt", "p50_ms"], aggfunc="mean").round(3)
        t.to_csv(OUT / "summary_E4.csv")
        print("\n===== E4: filtered recall@10 / p50 (card = 1/selectivity) =====\n", t.to_string())
    e5 = df[df.exp == "E5"]
    if len(e5):
        t = e5.pivot_table(index=["dataset", "method"], columns="db", values=["ndcg10", "p50_ms"], aggfunc="mean").round(4)
        t.to_csv(OUT / "summary_E5.csv")
        print("\n===== E5: lexical / hybrid / dense nDCG@10 =====\n", t.to_string())
    er = df[df.exp == "error"]
    if len(er):
        print("\n===== ERRORS =====\n", er[["dataset", "model", "db", "error"]].to_string())
    shutil.make_archive(str(OUT.parent / OUT.name), "zip", OUT)
    try:
        from IPython.display import FileLink, display
        display(FileLink(f"{OUT.name}.zip"))
    except Exception:
        pass
    print(f"Results: {OUT} | archive: {OUT.parent / (OUT.name + '.zip')}")


def main():
    from importlib.metadata import version
    info = dict(preset=PRESET, cfg=P, common=COMMON, versions=VERSIONS, hyb_ef=HYB_EF, rrf_k=RRF_K,
                bm25=dict(k1=BM25_K1, b=BM25_B), cpu_count=os.cpu_count(),
                ram_gb=round(psutil.virtual_memory().total / 2**30, 1), python=sys.version.split()[0],
                platform=platform.platform(), started=time.strftime("%Y-%m-%d %H:%M:%S"),
                py_pkgs={p: version(p) for p in ("qdrant-client", "weaviate-client", "psycopg", "pgvector", "scipy")})
    (OUT / f"env_{PRESET}_{int(time.time())}.json").write_text(json.dumps(info, indent=2, default=str))
    if PRESET in ("QNP", "QNPFIX"):
        (OUT / f"PRESET_{PRESET}").write_text("Qdrant rerun without payload indexes (S3 preset QNP)\n")
    jobs = list(dict.fromkeys(P["E3"] + P["E4"] + P["E5"]))
    t0 = time.time()
    for i, job in enumerate(jobs):
        print(f"\n######## job {i+1}/{len(jobs)} (elapsed {(time.time()-t0)/60:.0f} min)", flush=True)
        run_job(job)
    summarize()
    STAGE.update(name="done", t=time.time())
    print(f"\n>> Finished in {(time.time() - t0) / 60:.0f} min")


if __name__ == "__main__":
    main()
