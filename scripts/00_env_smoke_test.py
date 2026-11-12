#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S0 — Environment smoke test for Qdrant, Weaviate and PostgreSQL+pgvector on Kaggle.

Kaggle settings: Accelerator = None (CPU), Internet = On.
Run: paste the whole file into a single notebook cell (or `!python 00_env_smoke_test.py`).

What it verifies (for each DB, isolated with try/except so one failure doesn't stop the others):
  1. Installation without Docker (Qdrant & Weaviate: official Linux release binaries;
     pgvector: apt PostgreSQL + pgvector compiled from source).
  2. Collection/table creation with identical HNSW parameters (M, ef_construction, cosine).
  3. Bulk ingest + index build timing, peak RSS of the server process(es), on-disk size.
  4. Run-time ef_search change works (recall must increase with ef).
  5. Dense k-NN recall@10 vs FAISS exact ground truth, latency p50/p95/p99, QPS.
  6. Filtered k-NN (integer attribute equality) recall vs filtered ground truth.
  7. Hybrid search path (Weaviate BM25+vector, Qdrant sparse+dense RRF, pgvector tsvector+RRF).
Outputs: /kaggle/working/s0_results.json and s0_summary.csv, plus server logs.
"""

import os, sys, re, json, time, glob, shutil, tarfile, subprocess, threading, platform, traceback, random
from pathlib import Path

# ----------------------------------------------------------------------------- config
CFG = dict(
    QDRANT_TAG=None,      # e.g. "v1.20.0"; None -> latest GitHub release
    WEAVIATE_TAG=None,    # e.g. "v1.34.0"; None -> latest GitHub release
    PGVECTOR_TAG=None,    # e.g. "v0.8.1";  None -> highest vX.Y.Z tag
    N=20_000, NQ=300, DIM=384, K=10,
    M=16, EF_C=128, EF_S=64, EF_SWEEP=[16, 64, 256],
    N_CAT=10, VOCAB=2_000, DOC_LEN=30, Q_TOKENS=5, N_CLUSTERS=200,
    WARMUP=50, SEED=42,
)
OUT = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd() / "s0_out"
ROOT = Path("/tmp/vdb")
for p in (OUT, ROOT):
    p.mkdir(parents=True, exist_ok=True)


def sh(cmd, check=True, timeout=3600):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    if check and r.returncode != 0:
        raise RuntimeError(f"Command failed ({r.returncode}): {cmd}\n--- stdout ---\n{r.stdout[-3000:]}\n--- stderr ---\n{r.stderr[-3000:]}")
    return r.stdout.strip()


# ----------------------------------------------------------------------------- python deps
print(">> Installing Python packages ...", flush=True)
sh(f"{sys.executable} -m pip install -q --upgrade "
   "'qdrant-client>=1.12' 'weaviate-client>=4.10' 'psycopg[binary]>=3.2' 'pgvector>=0.3' "
   "faiss-cpu psutil requests pandas numpy", timeout=1800)

import numpy as np
import pandas as pd
import psutil
import requests
import faiss

RESULTS = {"config": CFG, "env": {}, "versions": {}, "dbs": {}}


# ----------------------------------------------------------------------------- helpers
def gh_release(repo, tag=None):
    url = f"https://api.github.com/repos/{repo}/releases/" + (f"tags/{tag}" if tag else "latest")
    r = requests.get(url, headers={"Accept": "application/vnd.github+json"}, timeout=60)
    r.raise_for_status()
    return r.json()


def download(url, dest):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    return dest


def extract_binary(tar_path, out_dir, name):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path) as t:
        t.extractall(out_dir)
    cands = [p for p in out_dir.rglob(name) if p.is_file()]
    if not cands:
        raise FileNotFoundError(f"binary '{name}' not found in {tar_path}")
    b = cands[0]
    b.chmod(0o755)
    return b


def wait_http(url, timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if requests.get(url, timeout=3).status_code < 500:
                return True
        except Exception:
            pass
        time.sleep(1)
    raise TimeoutError(f"{url} not ready after {timeout}s")


def dir_size(path):
    total = 0
    for p in Path(path).rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            pass
    return total


class RSSSampler:
    """Samples summed RSS of processes returned by `pids_fn` every `dt` seconds, keeps the peak."""
    def __init__(self, pids_fn, dt=0.2):
        self.pids_fn, self.dt, self.peak, self._stop = pids_fn, dt, 0, threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def current(self):
        tot = 0
        for pid in self.pids_fn():
            try:
                tot += psutil.Process(pid).memory_info().rss
            except psutil.Error:
                pass
        return tot

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self.current())
            time.sleep(self.dt)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._t.join()


def pct(a, q):
    return float(np.percentile(a, q))


def recall_at_k(pred, gt, k):
    return float(np.mean([len(set(p[:k]) & set(g[:k])) / k for p, g in zip(pred, gt)]))


# ----------------------------------------------------------------------------- environment info
def cpu_model():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


RESULTS["env"] = dict(
    python=sys.version.split()[0], platform=platform.platform(), cpu=cpu_model(),
    logical_cpus=os.cpu_count(), ram_gb=round(psutil.virtual_memory().total / 2**30, 1),
    kaggle=bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")), timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
)
print(">> Environment:", json.dumps(RESULTS["env"], indent=1), flush=True)

# ----------------------------------------------------------------------------- synthetic data
print(">> Generating synthetic clustered data ...", flush=True)
rng = np.random.default_rng(CFG["SEED"])
N, NQ, D, K = CFG["N"], CFG["NQ"], CFG["DIM"], CFG["K"]
centers = rng.standard_normal((CFG["N_CLUSTERS"], D)).astype(np.float32)
assign = rng.integers(0, CFG["N_CLUSTERS"], N)
X = centers[assign] + 0.35 * rng.standard_normal((N, D)).astype(np.float32)
X /= np.linalg.norm(X, axis=1, keepdims=True)
src = rng.choice(N, NQ, replace=False)                    # each query is a perturbation of a base doc
Q = X[src] + 0.05 * rng.standard_normal((NQ, D)).astype(np.float32)
Q /= np.linalg.norm(Q, axis=1, keepdims=True)
cats = rng.integers(0, CFG["N_CAT"], N).astype(np.int64)
qcats = rng.integers(0, CFG["N_CAT"], NQ).astype(np.int64)
zipf = 1.0 / np.arange(1, CFG["VOCAB"] + 1)
zipf /= zipf.sum()
toks = [rng.choice(CFG["VOCAB"], CFG["DOC_LEN"], p=zipf) for _ in range(N)]
texts = [" ".join(f"w{t}" for t in tk) for tk in toks]
qtoks = [rng.choice(toks[s], CFG["Q_TOKENS"], replace=False) for s in src]
qtexts = [" ".join(f"w{t}" for t in tk) for tk in qtoks]

# exact ground truth (cosine == inner product on unit vectors)
idx = faiss.IndexFlatIP(D)
idx.add(X)
_, GT = idx.search(Q, K)
GT_F = []
for q, c in zip(Q, qcats):
    ids = np.where(cats == c)[0]
    s = X[ids] @ q
    GT_F.append(ids[np.argsort(-s)[:K]].tolist())
GT = GT.tolist()
print(f"   N={N} D={D} NQ={NQ}; filter selectivity ≈ {1/CFG['N_CAT']:.0%}", flush=True)


def sparse_of(tk):
    u, c = np.unique(tk, return_counts=True)
    return u.astype(int).tolist(), c.astype(float).tolist()


# ============================================================================ adapters
class QdrantDB:
    name = "qdrant"

    def start(self):
        rel = gh_release("qdrant/qdrant", CFG["QDRANT_TAG"])
        assets = {a["name"]: a["browser_download_url"] for a in rel["assets"]}
        pick = next((n for pat in (r"x86_64-unknown-linux-gnu\.tar\.gz$", r"x86_64-unknown-linux-musl\.tar\.gz$")
                     for n in assets if re.search(pat, n)), None)
        if not pick:
            raise RuntimeError(f"No Linux x86_64 asset in qdrant {rel['tag_name']}: {list(assets)}")
        self.version = rel["tag_name"]
        base = ROOT / "qdrant"
        shutil.rmtree(base, ignore_errors=True)
        b = extract_binary(download(assets[pick], base / pick), base / "bin", "qdrant")
        self.data = base / "data"
        self.data.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ,
                   QDRANT__STORAGE__STORAGE_PATH=str(self.data / "storage"),
                   QDRANT__STORAGE__SNAPSHOTS_PATH=str(self.data / "snapshots"),
                   QDRANT__SERVICE__HOST="127.0.0.1",
                   QDRANT__SERVICE__HTTP_PORT="6333", QDRANT__SERVICE__GRPC_PORT="6334",
                   QDRANT__TELEMETRY_DISABLED="true", QDRANT__LOG_LEVEL="WARN")
        self.log = open(OUT / "qdrant.log", "w")
        self.proc = subprocess.Popen([str(b)], cwd=str(base), env=env, stdout=self.log, stderr=subprocess.STDOUT)
        wait_http("http://127.0.0.1:6333/readyz")
        from qdrant_client import QdrantClient
        self.c = QdrantClient(host="127.0.0.1", port=6333, grpc_port=6334, prefer_grpc=True, timeout=600)
        self.server_version = requests.get("http://127.0.0.1:6333/").json().get("version")

    def pids(self):
        return [self.proc.pid] if self.proc.poll() is None else []

    def create(self):
        from qdrant_client import models as m
        self.m = m
        self.coll = "bench"
        if self.c.collection_exists(self.coll):
            self.c.delete_collection(self.coll)
        self.c.create_collection(
            self.coll,
            vectors_config={"dense": m.VectorParams(size=D, distance=m.Distance.COSINE,
                                                    hnsw_config=m.HnswConfigDiff(m=CFG["M"], ef_construct=CFG["EF_C"],
                                                                                 full_scan_threshold=10))},
            sparse_vectors_config={"sparse": m.SparseVectorParams(modifier=m.Modifier.IDF)},
            optimizers_config=m.OptimizersConfigDiff(indexing_threshold=0),  # 0 = no indexing during upload
        )
        self.c.create_payload_index(self.coll, "cat", field_schema=m.PayloadSchemaType.INTEGER)

    def insert(self):
        m, B = self.m, 512
        for s in range(0, N, B):
            pts = []
            for i in range(s, min(s + B, N)):
                ind, val = sparse_of(toks[i])
                pts.append(m.PointStruct(id=i, payload={"cat": int(cats[i])},
                                         vector={"dense": X[i].tolist(), "sparse": m.SparseVector(indices=ind, values=val)}))
            self.c.upsert(self.coll, points=pts, wait=True)

    def build(self):
        m = self.m
        self.c.update_collection(self.coll, optimizers_config=m.OptimizersConfigDiff(indexing_threshold=10))
        time.sleep(0.5)
        t0 = time.time()
        while True:
            info = self.c.get_collection(self.coll)
            if info.status == m.CollectionStatus.GREEN and (info.indexed_vectors_count or 0) >= N:
                break
            if time.time() - t0 > 3600:
                raise TimeoutError(f"qdrant indexing not finished: {info}")
            time.sleep(0.25)
        self.indexed = info.indexed_vectors_count

    def set_ef(self, ef):
        self.ef = ef

    def search(self, q, k, cat=None):
        m = self.m
        flt = None if cat is None else m.Filter(must=[m.FieldCondition(key="cat", match=m.MatchValue(value=int(cat)))])
        r = self.c.query_points(self.coll, query=q.tolist(), using="dense", limit=k, query_filter=flt,
                                search_params=m.SearchParams(hnsw_ef=self.ef, exact=False), with_payload=False)
        return [int(p.id) for p in r.points]

    def hybrid(self, q, qtext, qtk, k):
        m = self.m
        ind, val = sparse_of(qtk)
        r = self.c.query_points(self.coll, prefetch=[
            m.Prefetch(query=q.tolist(), using="dense", limit=50, params=m.SearchParams(hnsw_ef=self.ef)),
            m.Prefetch(query=m.SparseVector(indices=ind, values=val), using="sparse", limit=50)],
            query=m.FusionQuery(fusion=m.Fusion.RRF), limit=k, with_payload=False)
        return [int(p.id) for p in r.points]

    def disk(self):
        return dir_size(self.data)

    def stop(self):
        try:
            self.c.close()
        except Exception:
            pass
        self.proc.terminate()
        self.proc.wait(30)
        self.log.close()


class WeaviateDB:
    name = "weaviate"

    def start(self):
        rel = gh_release("weaviate/weaviate", CFG["WEAVIATE_TAG"])
        assets = {a["name"]: a["browser_download_url"] for a in rel["assets"]}
        pick = next((n for n in assets if re.search(r"linux-amd64\.tar\.gz$", n)), None)
        if not pick:
            raise RuntimeError(f"No linux-amd64 asset in weaviate {rel['tag_name']}: {list(assets)}")
        self.version = rel["tag_name"]
        base = ROOT / "weaviate"
        shutil.rmtree(base, ignore_errors=True)
        b = extract_binary(download(assets[pick], base / pick), base / "bin", "weaviate")
        self.data = base / "data"
        self.data.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ,
                   PERSISTENCE_DATA_PATH=str(self.data), AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED="true",
                   DEFAULT_VECTORIZER_MODULE="none", ENABLE_MODULES="", CLUSTER_HOSTNAME="node1",
                   GRPC_PORT="50051", DISABLE_TELEMETRY="true", QUERY_DEFAULTS_LIMIT="25", LOG_LEVEL="warning")
        self.log = open(OUT / "weaviate.log", "w")
        self.proc = subprocess.Popen([str(b), "--host", "127.0.0.1", "--port", "8080", "--scheme", "http"],
                                     cwd=str(base), env=env, stdout=self.log, stderr=subprocess.STDOUT)
        wait_http("http://127.0.0.1:8080/v1/.well-known/ready")
        import weaviate
        from weaviate.classes.init import AdditionalConfig, Timeout
        self.c = weaviate.connect_to_local(host="127.0.0.1", port=8080, grpc_port=50051,
                                           additional_config=AdditionalConfig(timeout=Timeout(init=60, query=300, insert=600)))
        self.server_version = requests.get("http://127.0.0.1:8080/v1/meta").json().get("version")
        self.client_version = weaviate.__version__

    def pids(self):
        return [self.proc.pid] if self.proc.poll() is None else []

    def _hnsw(self, Configure):
        from weaviate.classes.config import VectorDistances
        kw = dict(distance_metric=VectorDistances.COSINE, ef_construction=CFG["EF_C"],
                  max_connections=CFG["M"], ef=CFG["EF_S"])
        try:
            return Configure.VectorIndex.hnsw(flat_search_cutoff=0, **kw)   # force graph path for filters
        except TypeError:
            return Configure.VectorIndex.hnsw(**kw)

    def create(self):
        from weaviate.classes.config import Configure, Property, DataType
        self.coll = "Bench"
        if self.c.collections.exists(self.coll):
            self.c.collections.delete(self.coll)
        props = [Property(name="cat", data_type=DataType.INT), Property(name="text", data_type=DataType.TEXT)]
        hnsw = self._hnsw(Configure)
        try:                                   # new API (named vectors), weaviate-client >= 4.16
            self.c.collections.create(self.coll, properties=props,
                                      vector_config=Configure.Vectors.self_provided(vector_index_config=hnsw))
            self.named = True
        except (AttributeError, TypeError):    # legacy API
            self.c.collections.create(self.coll, properties=props,
                                      vectorizer_config=Configure.Vectorizer.none(), vector_index_config=hnsw)
            self.named = False
        self.col = self.c.collections.get(self.coll)

    def insert(self):
        import uuid
        with self.col.batch.fixed_size(batch_size=1000, concurrent_requests=2) as b:
            for i in range(N):
                v = X[i].tolist()
                b.add_object(properties={"cat": int(cats[i]), "text": texts[i]},
                             vector={"default": v} if self.named else v, uuid=uuid.UUID(int=i + 1))
        failed = self.col.batch.failed_objects
        if failed:
            raise RuntimeError(f"{len(failed)} objects failed, first: {failed[0]}")

    def build(self):
        # Weaviate builds HNSW synchronously during import (ASYNC_INDEXING=false): wait until counts match.
        t0 = time.time()
        while self.col.aggregate.over_all(total_count=True).total_count < N:
            if time.time() - t0 > 3600:
                raise TimeoutError("weaviate object count never reached N")
            time.sleep(0.25)
        self.indexed = N

    def set_ef(self, ef):
        from weaviate.classes.config import Reconfigure
        if self.named:
            self.col.config.update(vector_config=Reconfigure.Vectors.update(
                name="default", vector_index_config=Reconfigure.VectorIndex.hnsw(ef=ef)))
        else:
            self.col.config.update(vector_index_config=Reconfigure.VectorIndex.hnsw(ef=ef))
        time.sleep(1.0)
        self.ef = ef

    def search(self, q, k, cat=None):
        from weaviate.classes.query import Filter
        flt = None if cat is None else Filter.by_property("cat").equal(int(cat))
        r = self.col.query.near_vector(near_vector=q.tolist(), limit=k, filters=flt, return_properties=[])
        return [o.uuid.int - 1 for o in r.objects]

    def hybrid(self, q, qtext, qtk, k):
        from weaviate.classes.query import HybridFusion
        r = self.col.query.hybrid(query=qtext, vector=q.tolist(), alpha=0.5, limit=k,
                                  fusion_type=HybridFusion.RANKED, return_properties=[])
        return [o.uuid.int - 1 for o in r.objects]

    def disk(self):
        return dir_size(self.data)

    def stop(self):
        try:
            self.c.close()
        except Exception:
            pass
        self.proc.terminate()
        self.proc.wait(60)
        self.log.close()


class PgVectorDB:
    name = "pgvector"

    def start(self):
        print("   apt-get install postgresql (this takes a few minutes) ...", flush=True)
        sh("apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "
           "postgresql postgresql-contrib postgresql-server-dev-all build-essential git", timeout=1800)
        bins = sorted(glob.glob("/usr/lib/postgresql/*/bin"), key=lambda p: int(Path(p).parent.name))
        pgbin = Path(bins[-1])
        self.pg_major = pgbin.parent.name
        pg_config = pgbin / "pg_config"
        # pgvector from source
        tag = CFG["PGVECTOR_TAG"]
        if not tag:
            tags = re.findall(r"refs/tags/(v\d+\.\d+\.\d+)$",
                              sh("git ls-remote --tags https://github.com/pgvector/pgvector.git"), re.M)
            tag = max(tags, key=lambda t: tuple(map(int, t[1:].split("."))))
        self.version = tag
        srcdir = ROOT / "pgvector-src"
        shutil.rmtree(srcdir, ignore_errors=True)
        sh(f"git clone -q --depth 1 --branch {tag} https://github.com/pgvector/pgvector.git {srcdir}")
        sh(f"cd {srcdir} && make -s PG_CONFIG={pg_config} && make -s install PG_CONFIG={pg_config}", timeout=1800)
        # cluster
        etc = Path(f"/etc/postgresql/{self.pg_major}/main")
        if not etc.exists():
            sh(f"pg_createcluster {self.pg_major} main")
        hba = etc / "pg_hba.conf"
        hba.write_text(re.sub(r"\b(peer|md5|scram-sha-256)\b", "trust", hba.read_text()))
        mem_gb = psutil.virtual_memory().total / 2**30
        (etc / "conf.d").mkdir(exist_ok=True)
        self.pg_settings = {
            "listen_addresses": "'127.0.0.1'", "port": "5432",
            "shared_buffers": f"'{max(1, int(mem_gb * 0.25))}GB'",
            "maintenance_work_mem": f"'{max(1, int(mem_gb * 0.25))}GB'",
            "max_parallel_maintenance_workers": str(max(1, (os.cpu_count() or 2) - 1)),
            "max_parallel_workers": str(os.cpu_count() or 2),
            "effective_cache_size": f"'{max(1, int(mem_gb * 0.5))}GB'",
            "work_mem": "'64MB'", "jit": "off",
        }
        (etc / "conf.d" / "bench.conf").write_text("\n".join(f"{k} = {v}" for k, v in self.pg_settings.items()) + "\n")
        conf = etc / "postgresql.conf"
        if "include_dir = 'conf.d'" not in conf.read_text():
            with open(conf, "a") as f:
                f.write("\ninclude_dir = 'conf.d'\n")
        sh(f"pg_ctlcluster {self.pg_major} main restart || pg_ctlcluster {self.pg_major} main start")
        import psycopg
        from pgvector.psycopg import register_vector
        for _ in range(60):
            try:
                self.conn = psycopg.connect("host=127.0.0.1 port=5432 user=postgres dbname=postgres", autocommit=True)
                break
            except Exception:
                time.sleep(1)
        else:
            raise RuntimeError("cannot connect to PostgreSQL")
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self.conn)
        self.server_version = self.conn.execute("SHOW server_version").fetchone()[0]
        self.ext_version = self.conn.execute("SELECT extversion FROM pg_extension WHERE extname='vector'").fetchone()[0]
        self.data = Path(self.conn.execute("SHOW data_directory").fetchone()[0])

    def pids(self):
        out = []
        for p in psutil.process_iter(["name", "username"]):
            if p.info["name"] in ("postgres", "postmaster"):
                out.append(p.pid)
        return out

    def create(self):
        c = self.conn
        c.execute("DROP TABLE IF EXISTS items")
        c.execute(f"""CREATE TABLE items (id int PRIMARY KEY, cat int NOT NULL, txt text,
                      tsv tsvector GENERATED ALWAYS AS (to_tsvector('simple', txt)) STORED,
                      emb vector({D}))""")

    def insert(self):
        with self.conn.cursor() as cur:
            with cur.copy("COPY items (id, cat, txt, emb) FROM STDIN WITH (FORMAT BINARY)") as cp:
                cp.set_types(["int4", "int4", "text", "vector"])
                for i in range(N):
                    cp.write_row((i, int(cats[i]), texts[i], X[i]))

    def build(self):
        c = self.conn
        c.execute(f"CREATE INDEX items_hnsw ON items USING hnsw (emb vector_cosine_ops) "
                  f"WITH (m = {CFG['M']}, ef_construction = {CFG['EF_C']})")
        self.indexed = N
        # auxiliary indexes (timed separately, not part of HNSW build time)
        t = time.perf_counter()
        c.execute("CREATE INDEX items_cat ON items (cat)")
        c.execute("CREATE INDEX items_tsv ON items USING gin (tsv)")
        c.execute("VACUUM ANALYZE items")
        self.aux_index_s = time.perf_counter() - t
        try:
            c.execute("SET hnsw.iterative_scan = relaxed_order")
            self.iterative_scan = "relaxed_order"
        except Exception:
            self.iterative_scan = "unsupported"

    def set_ef(self, ef):
        self.conn.execute(f"SET hnsw.ef_search = {int(ef)}")
        self.ef = ef

    def search(self, q, k, cat=None):
        if cat is None:
            rows = self.conn.execute("SELECT id FROM items ORDER BY emb <=> %s LIMIT %s", (q, k)).fetchall()
        else:
            rows = self.conn.execute("SELECT id FROM items WHERE cat = %s ORDER BY emb <=> %s LIMIT %s",
                                     (int(cat), q, k)).fetchall()
        return [r[0] for r in rows]

    def hybrid(self, q, qtext, qtk, k):
        tsq = " | ".join(qtext.split())
        sql = """
        WITH d AS (SELECT id, ROW_NUMBER() OVER (ORDER BY dist) AS r
                   FROM (SELECT id, emb <=> %(q)s AS dist FROM items ORDER BY dist LIMIT 50) s),
             t AS (SELECT id, ROW_NUMBER() OVER (ORDER BY rk DESC) AS r
                   FROM (SELECT id, ts_rank_cd(tsv, query) AS rk
                         FROM items, to_tsquery('simple', %(t)s) query
                         WHERE tsv @@ query ORDER BY rk DESC LIMIT 50) s2)
        SELECT COALESCE(d.id, t.id) AS id,
               COALESCE(1.0 / (60 + d.r), 0) + COALESCE(1.0 / (60 + t.r), 0) AS score
        FROM d FULL OUTER JOIN t ON d.id = t.id
        ORDER BY score DESC LIMIT %(k)s"""
        return [r[0] for r in self.conn.execute(sql, {"q": q, "t": tsq, "k": k}).fetchall()]

    def disk(self):
        return int(self.conn.execute("SELECT pg_total_relation_size('items')").fetchone()[0])

    def stop(self):
        try:
            self.conn.close()
        except Exception:
            pass
        sh(f"pg_ctlcluster {self.pg_major} main stop", check=False)


# ============================================================================ benchmark routine
def timed_queries(fn, queries):
    lat, out = [], []
    for args in queries:
        t = time.perf_counter()
        out.append(fn(*args))
        lat.append((time.perf_counter() - t) * 1000)
    return out, np.array(lat)


def run_db(db):
    r = {"status": "FAIL"}
    try:
        t = time.perf_counter()
        db.start()
        r["startup_s"] = round(time.perf_counter() - t, 2)
        r["version"] = getattr(db, "version", None)
        r["server_version"] = getattr(db, "server_version", None)
        for k in ("client_version", "ext_version", "pg_major", "pg_settings"):
            if hasattr(db, k):
                r[k] = getattr(db, k)
        print(f"   started {db.name} {r['version']} (server {r['server_version']}) in {r['startup_s']}s", flush=True)
        rss = RSSSampler(db.pids)
        r["rss_idle_mb"] = round(rss.current() / 2**20, 1)
        db.create()
        with rss:
            t = time.perf_counter(); db.insert(); r["insert_s"] = round(time.perf_counter() - t, 2)
            t = time.perf_counter(); db.build();  r["index_build_s"] = round(time.perf_counter() - t, 2)
        r["ingest_total_s"] = round(r["insert_s"] + r["index_build_s"], 2)
        r["insert_vps"] = round(N / r["insert_s"], 1)
        r["rss_peak_ingest_mb"] = round(rss.peak / 2**20, 1)
        r["rss_after_ingest_mb"] = round(rss.current() / 2**20, 1)
        r["disk_mb"] = round(db.disk() / 2**20, 1)
        r["indexed_vectors"] = getattr(db, "indexed", None)
        for k in ("aux_index_s", "iterative_scan", "named"):
            if hasattr(db, k):
                r[k] = getattr(db, k)

        # ef sweep: recall must be (weakly) increasing in ef -> proves run-time ef works
        sweep = {}
        for ef in CFG["EF_SWEEP"]:
            db.set_ef(ef)
            for i in range(CFG["WARMUP"]):
                db.search(Q[i % NQ], K)
            pred, lat = timed_queries(lambda q: db.search(q, K), [(q,) for q in Q])
            sweep[ef] = dict(recall10=round(recall_at_k(pred, GT, K), 4), p50_ms=round(pct(lat, 50), 3),
                             p95_ms=round(pct(lat, 95), 3), p99_ms=round(pct(lat, 99), 3),
                             qps=round(NQ / (lat.sum() / 1000), 1))
        r["ef_sweep"] = sweep
        recs = [sweep[e]["recall10"] for e in CFG["EF_SWEEP"]]
        r["ef_effect_ok"] = bool(recs[-1] >= recs[0] and (recs[-1] > recs[0] or recs[0] >= 0.999))

        db.set_ef(CFG["EF_S"])
        pred, lat = timed_queries(lambda q, c: db.search(q, K, c), list(zip(Q, qcats)))
        r["filtered"] = dict(recall10=round(recall_at_k(pred, GT_F, K), 4), p50_ms=round(pct(lat, 50), 3),
                             p95_ms=round(pct(lat, 95), 3), qps=round(NQ / (lat.sum() / 1000), 1))
        r["filter_correct"] = bool(all(int(cats[i]) == int(c) for p, c in zip(pred, qcats) for i in p))

        dense_pred = [db.search(q, K) for q in Q]
        hyb_pred, lat = timed_queries(lambda q, t, tk: db.hybrid(q, t, tk, K), list(zip(Q, qtexts, qtoks)))
        r["hybrid"] = dict(src_hit10_dense=round(float(np.mean([s in p for s, p in zip(src, dense_pred)])), 4),
                           src_hit10_hybrid=round(float(np.mean([s in p for s, p in zip(src, hyb_pred)])), 4),
                           p50_ms=round(pct(lat, 50), 3), qps=round(NQ / (lat.sum() / 1000), 1))
        r["status"] = "PASS" if (r["ef_effect_ok"] and r["filter_correct"] and sweep[CFG["EF_SWEEP"][-1]]["recall10"] > 0.8) else "WARN"
    except Exception as e:
        r["error"] = f"{type(e).__name__}: {e}"
        r["traceback"] = traceback.format_exc()
        print(r["traceback"], flush=True)
    finally:
        try:
            db.stop()
        except Exception:
            pass
    return r


# ============================================================================ main
for DB in (QdrantDB, WeaviateDB, PgVectorDB):
    db = DB()
    print(f"\n>> {db.name}", flush=True)
    RESULTS["dbs"][db.name] = run_db(db)
    print(json.dumps({k: v for k, v in RESULTS["dbs"][db.name].items() if k != "traceback"}, indent=1, default=str), flush=True)

for pkg in ("qdrant-client", "weaviate-client", "psycopg", "pgvector", "faiss-cpu", "numpy"):
    try:
        from importlib.metadata import version
        RESULTS["versions"][pkg] = version(pkg)
    except Exception:
        RESULTS["versions"][pkg] = None

(OUT / "s0_results.json").write_text(json.dumps(RESULTS, indent=2, default=str))
rows = []
for name, r in RESULTS["dbs"].items():
    sw = r.get("ef_sweep", {})
    rows.append(dict(db=name, status=r.get("status"), version=r.get("version"), server=r.get("server_version"),
                     insert_s=r.get("insert_s"), build_s=r.get("index_build_s"),
                     rss_peak_mb=r.get("rss_peak_ingest_mb"), disk_mb=r.get("disk_mb"),
                     **{f"R@10 ef{e}": sw.get(e, {}).get("recall10") for e in CFG["EF_SWEEP"]},
                     **{f"qps ef{e}": sw.get(e, {}).get("qps") for e in CFG["EF_SWEEP"]},
                     filt_R10=r.get("filtered", {}).get("recall10"),
                     hyb_hit=r.get("hybrid", {}).get("src_hit10_hybrid"),
                     dense_hit=r.get("hybrid", {}).get("src_hit10_dense"),
                     error=(r.get("error") or "")[:120]))
df = pd.DataFrame(rows)
df.to_csv(OUT / "s0_summary.csv", index=False)
pd.set_option("display.width", 250, "display.max_columns", 50)
print("\n================ S0 SUMMARY ================")
print(df.to_string(index=False))
print(f"\nSaved: {OUT/'s0_results.json'}, {OUT/'s0_summary.csv'}, server logs in {OUT}")
