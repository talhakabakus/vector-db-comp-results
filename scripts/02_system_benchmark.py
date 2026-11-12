#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S2 — System benchmark: ingest/index (E1), search performance sweep + concurrency (E2),
     system-default vs common configuration (E2b).  Qdrant v1.20.0, Weaviate v1.40.4, PostgreSQL+pgvector v0.8.7.

Kaggle: Accelerator = None (CPU), Internet = On, Input = the S1 dataset (vdb_data).
Paste the whole file into ONE cell. Choose PRESET below:
  "A"  fiqa, hotpotqa_sub, quora (bge-small & bge-base) + default-config ablation on quora/bge-base  (~6-8 h)
  "B"  scale: dbpedia-entities-openai-1M (1536-d, downloaded from Hugging Face)                      (~4-6 h)
  "SMOKE"  fiqa only, 1 rep, short sweeps (~25 min) — run this first to validate.
Resumable: finished (job, db, rep) units are recorded in s2_results/done.json and skipped on re-run
(as long as /kaggle/working survives). Download s2_results.zip regularly (FileLink printed at the end).

Methodology (see plan §3.1, §4.3):
 * Each (job, db, rep) unit = fresh server process + empty storage -> ingest -> index -> footprint -> ef sweep.
 * Common config: cosine, HNSW M=16, ef_construction=128; documented bulk-load best practice per system
   (Qdrant: indexing disabled during upload; pgvector: COPY then CREATE INDEX; Weaviate: synchronous indexing).
 * Ingest metric = insert + index build until searchable & fully indexed.
 * Memory = PSS (proportional set size; avoids double counting PostgreSQL shared memory), peak & steady.
 * Sweep: k=10 and k=100 over ef grids, several passes per build; recall vs exact GT from S1.
 * Concurrency: spawn-based client processes at the smallest ef reaching recall@10 >= 0.95 / 0.99.
 * Protocol overhead: k=1 search on a 1-vector collection ("null query").
"""

import os, sys, re, json, time, glob, shutil, tarfile, subprocess, threading, platform, traceback, random
from pathlib import Path

PRESET = "A2"             # "SMOKE" | "A" | "A1" + "A2" | "B"

VERSIONS = dict(QDRANT_TAG="v1.20.0", WEAVIATE_TAG="v1.40.4", PGVECTOR_TAG="v0.8.7",
                PIP=["qdrant-client==1.20.0", "weaviate-client==4.24.1", "psycopg[binary]==3.3.8",
                     "pgvector==0.5.1", "faiss-cpu==1.15.1", "psutil", "requests", "pandas", "pyarrow",
                     "huggingface_hub"])
COMMON = dict(M=16, EF_C=128, EF_INIT=64)
PRESETS = {
    "SMOKE": dict(JOBS=[("fiqa", "bge-base")], REPS=1, PASSES=1, QUERY_CAP=500,
                  EF10=[10, 32, 128], EF100=[100, 256], CONC=[1, 4], CONC_DUR=5, CONC_REPS=1,
                  RECALL_TARGETS=[0.95], DEFAULT_JOBS=[("fiqa", "bge-base")]),
    "A": dict(JOBS=[("fiqa", "bge-base"), ("hotpotqa_sub", "bge-base"), ("quora", "bge-small"), ("quora", "bge-base")],
              REPS=3, PASSES=2, QUERY_CAP=2000,
              EF10=[10, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512], EF100=[100, 128, 192, 256, 384, 512],
              CONC=[1, 2, 4, 8], CONC_DUR=15, CONC_REPS=3, RECALL_TARGETS=[0.95, 0.99],
              DEFAULT_JOBS=[("quora", "bge-base")]),
    # A split over two sessions (same settings as "A"), if a single ~8 h session is too risky:
    "A1": dict(JOBS=[("fiqa", "bge-base"), ("hotpotqa_sub", "bge-base"), ("quora", "bge-small")], DEFAULT_JOBS=[]),
    "A2": dict(JOBS=[("quora", "bge-base")], DEFAULT_JOBS=[("quora", "bge-base")]),
    "B": dict(JOBS=[("dbpedia1m", "openai-ada-002")], REPS=1, PASSES=3, QUERY_CAP=2000,
              EF10=[10, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512], EF100=[100, 128, 192, 256, 384, 512],
              CONC=[1, 2, 4, 8], CONC_DUR=15, CONC_REPS=3, RECALL_TARGETS=[0.95, 0.99], DEFAULT_JOBS=[],
              DBPEDIA_N=1_000_000),
}
for _k in ("A1", "A2"):
    PRESETS[_k] = {**PRESETS["A"], **PRESETS[_k]}
P = PRESETS[PRESET]
DBS = ["qdrant", "weaviate", "pgvector"]
SEED = 42
DATA_HINT = "/kaggle/input/datasets/talhakabakus/vdb-data/vdb_data"
OUT = (Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()) / "s2_results"
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

# ============================================================================ worker module (spawn-safe)
WORKER_SRC = r'''
import time, numpy as np

def make_searcher(db, conf):
    """Returns (search_fn(q, k) -> list[int], close_fn). One connection per process."""
    if db == "qdrant":
        from qdrant_client import QdrantClient, models as m
        c = QdrantClient(host="127.0.0.1", port=6333, grpc_port=6334, prefer_grpc=True, timeout=600)
        sp = m.SearchParams(hnsw_ef=int(conf["ef"])) if conf.get("ef") else None
        coll = conf["coll"]
        def f(q, k):
            return [int(p.id) for p in c.query_points(coll, query=q.tolist(), limit=k, search_params=sp,
                                                      with_payload=False).points]
        return f, c.close
    if db == "weaviate":
        import weaviate
        from weaviate.classes.init import AdditionalConfig, Timeout
        c = weaviate.connect_to_local(host="127.0.0.1", port=8080, grpc_port=50051,
                                      additional_config=AdditionalConfig(timeout=Timeout(init=60, query=600, insert=600)))
        col = c.collections.get(conf["coll"])
        def f(q, k):
            return [o.uuid.int - 1 for o in col.query.near_vector(near_vector=q.tolist(), limit=k, return_properties=[]).objects]
        return f, c.close
    if db == "pgvector":
        import psycopg
        from pgvector.psycopg import register_vector
        cn = psycopg.connect(conf["dsn"], autocommit=True)
        register_vector(cn)
        if conf.get("ef"):
            cn.execute(f"SET hnsw.ef_search = {int(conf['ef'])}")
        if conf.get("force_index"):
            # Common config measures HNSW at a fixed ef. For large high-dimensional tables PostgreSQL's cost model
            # abandons the HNSW index at high ef_search with small k (exact parallel seq scan instead); that
            # planner behaviour is recorded separately (check_plan) and reported, not mixed into the ef sweep.
            cn.execute("SET enable_seqscan = off")
        tbl = conf["coll"]
        # LIMIT is a literal, NOT a bind parameter: psycopg auto-prepares a statement after 5 executions, and
        # PostgreSQL's generic plan for "LIMIT $2" assumes 10% of the table is returned and switches to a
        # parallel sequential scan (exact search, ~100x slower). With a literal LIMIT the generic plan keeps
        # using the HNSW index.
        sqls = {}
        def f(q, k):
            if k not in sqls:
                sqls[k] = f"SELECT id FROM {tbl} ORDER BY emb <=> %s LIMIT {int(k)}"
            return [r[0] for r in cn.execute(sqls[k], (q,)).fetchall()]
        return f, cn.close
    raise ValueError(db)

def run_worker(args):
    db, conf, qpath, start, duration, k, t_start = args
    Q = np.load(qpath)
    f, close = make_searcher(db, conf)
    for i in range(20):
        f(Q[(start + i) % len(Q)], k)
    while time.time() < t_start:
        time.sleep(0.0005)
    lat, i, t_end = [], start, t_start + duration
    while time.time() < t_end:
        q = Q[i % len(Q)]
        t = time.perf_counter()
        f(q, k)
        lat.append(time.perf_counter() - t)
        i += 1
    close()
    return lat
'''
ROOT.mkdir(parents=True, exist_ok=True)
(ROOT / "vdb_worker.py").write_text(WORKER_SRC)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# A notebook kernel caches imported modules: re-running this cell would silently keep the worker code of the
# FIRST run (this happened in A2). Drop the cached module so the file written above is what gets imported.
sys.modules.pop("vdb_worker", None)
import vdb_worker  # noqa: E402


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


def gh_asset(repo, tag, pattern):
    r = requests.get(f"https://api.github.com/repos/{repo}/releases/tags/{tag}", timeout=60)
    r.raise_for_status()
    for a in r.json()["assets"]:
        if re.search(pattern, a["name"]):
            return a["name"], a["browser_download_url"]
    raise RuntimeError(f"no asset {pattern} in {repo} {tag}")


def download(url, dest):
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for ch in r.iter_content(1 << 22):
                f.write(ch)
    return dest


def install_binary(repo, tag, pattern, name):
    bdir = ROOT / "bin" / name
    found = [p for p in bdir.rglob(name) if p.is_file()] if bdir.exists() else []
    if found:
        return found[0]
    an, url = gh_asset(repo, tag, pattern)
    tp = download(url, ROOT / "dl" / an)
    with tarfile.open(tp) as t:
        t.extractall(bdir)
    tp.unlink()                                   # save disk space
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


def dir_size(path, exclude=None):
    tot = 0
    for p in Path(path).rglob("*"):
        try:
            if p.is_file() and not (exclude and re.search(exclude, str(p))):
                tot += p.stat().st_size
        except OSError:
            pass
    return tot


def pss_of(pids):
    tot = 0
    for pid in pids:
        try:
            tot += psutil.Process(pid).memory_full_info().pss
        except (psutil.Error, AttributeError):
            try:
                tot += psutil.Process(pid).memory_info().rss
            except psutil.Error:
                pass
    return tot


class MemSampler:
    def __init__(self, pids_fn, dt=0.5):
        self.pids_fn, self.dt, self.peak = pids_fn, dt, 0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, pss_of(self.pids_fn()))
            time.sleep(self.dt)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._t.join()


def recall_rows(pred, gt, k):
    """per-query recall@k (gt rows: doc_idx, -1 padded)."""
    out = np.empty(len(pred), dtype=np.float32)
    for i, (p, g) in enumerate(zip(pred, gt)):
        g = [x for x in g[:k] if x >= 0]
        out[i] = len(set(p[:k]) & set(g)) / max(1, len(g))
    return out


def lat_stats(lat_s):
    a = np.asarray(lat_s) * 1000
    return dict(p50_ms=float(np.percentile(a, 50)), p95_ms=float(np.percentile(a, 95)),
                p99_ms=float(np.percentile(a, 99)), mean_ms=float(a.mean()))


# ============================================================================ results store
ROWS = OUT / "rows.jsonl"
DONE = OUT / "done.json"


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
QUIET_STAGES = ("sweep", "concurrency", "default-sweep", "warm-up")


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


# ============================================================================ server adapters
class Qdrant:
    name = "qdrant"
    coll = "bench"

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

    def pids(self):
        return [self.proc.pid] if self.proc.poll() is None else []

    def create(self, D, default=False, name=None):
        m, name = self.m, name or self.coll
        if self.c.collection_exists(name):
            self.c.delete_collection(name)
        if default:
            self.c.create_collection(name, vectors_config=m.VectorParams(size=D, distance=m.Distance.COSINE))
        else:
            self.c.create_collection(name, vectors_config=m.VectorParams(
                size=D, distance=m.Distance.COSINE, hnsw_config=m.HnswConfigDiff(m=COMMON["M"], ef_construct=COMMON["EF_C"])),
                optimizers_config=m.OptimizersConfigDiff(indexing_threshold=0))   # bulk-load best practice
        self.default = default

    def insert(self, X, name=None):
        self.c.upload_collection(name or self.coll, vectors=X, ids=list(range(len(X))), batch_size=1024, parallel=1, wait=True)

    def build(self, N):
        m = self.m
        if not self.default:
            self.c.update_collection(self.coll, optimizers_config=m.OptimizersConfigDiff(indexing_threshold=10000))
        time.sleep(1.0)
        green, t0 = 0, time.time()
        while green < 3:
            info = self.c.get_collection(self.coll)
            ok = info.status == m.CollectionStatus.GREEN
            green = green + 1 if ok else 0
            if time.time() - t0 > 4 * 3600:
                raise TimeoutError(str(info))
            time.sleep(0.5)
        return dict(indexed_vectors=info.indexed_vectors_count, points=info.points_count,
                    segments=info.segments_count, indexed_frac=(info.indexed_vectors_count or 0) / max(1, N))

    def footprint(self):
        return dict(disk_mb=dir_size(self.data) / 2**20, disk_excl_wal_mb=dir_size(self.data, r"/wal/") / 2**20)

    def warm(self):
        pass

    def search_conf(self, ef):
        return dict(coll=self.coll, ef=ef)

    def set_ef(self, ef):
        pass

    def drop(self, name):
        self.c.delete_collection(name)

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
        # DISK_USE_*: Kaggle's root FS is ~88% used -> Weaviate would warn at 80% and go READ-ONLY at 90% by default.
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

    def pids(self):
        return [self.proc.pid] if self.proc.poll() is None else []

    def create(self, D, default=False, name=None):
        from weaviate.classes.config import Configure, VectorDistances
        name = name or self.coll
        if self.c.collections.exists(name):
            self.c.collections.delete(name)
        if default:
            hnsw = Configure.VectorIndex.hnsw(distance_metric=VectorDistances.COSINE)
        else:
            hnsw = Configure.VectorIndex.hnsw(distance_metric=VectorDistances.COSINE, ef_construction=COMMON["EF_C"],
                                              max_connections=COMMON["M"], ef=COMMON["EF_INIT"])
        self.c.collections.create(name, vector_config=Configure.Vectors.self_provided(vector_index_config=hnsw))
        self.default = default

    def insert(self, X, name=None):
        import uuid
        col = self.c.collections.get(name or self.coll)
        with col.batch.fixed_size(batch_size=1000, concurrent_requests=4) as b:
            for i in range(len(X)):
                b.add_object(properties={}, vector={"default": X[i].tolist()}, uuid=uuid.UUID(int=i + 1))
        if col.batch.failed_objects:
            raise RuntimeError(f"{len(col.batch.failed_objects)} failed: {col.batch.failed_objects[0]}")

    def build(self, N):
        col = self.c.collections.get(self.coll)
        t0 = time.time()
        while col.aggregate.over_all(total_count=True).total_count < N:
            if time.time() - t0 > 3600:
                raise TimeoutError("weaviate count")
            time.sleep(0.5)
        cfg = col.config.get()
        try:
            vi = cfg.vector_config["default"].vector_index_config
            extra = dict(max_connections=vi.max_connections, ef_construction=vi.ef_construction, ef=vi.ef)
        except Exception:
            extra = {}
        return dict(indexed_vectors=N, **extra)

    def footprint(self):
        return dict(disk_mb=dir_size(self.data) / 2**20)

    def warm(self):
        pass

    def search_conf(self, ef):
        return dict(coll=self.coll)

    def set_ef(self, ef):
        from weaviate.classes.config import Reconfigure
        col = self.c.collections.get(self.coll)
        col.config.update(vector_config=Reconfigure.Vectors.update(
            name="default", vector_index_config=Reconfigure.VectorIndex.hnsw(ef=int(ef))))
        time.sleep(1.0)

    def drop(self, name):
        self.c.collections.delete(name)

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
    coll = "items"
    DSN = "host=127.0.0.1 port=5432 user=postgres dbname=postgres"
    _installed = False
    default = False

    @classmethod
    def install(cls):
        if cls._installed:
            return
        print("   installing PostgreSQL + pgvector (apt + source build) ...", flush=True)
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
                        "work_mem": "'64MB'", "jit": "off", "max_connections": "100"}
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
                self.conn = psycopg.connect(self.DSN, autocommit=True)
                break
            except Exception:
                time.sleep(0.5)
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self.conn)
        for t in ("items", "tiny"):
            self.conn.execute(f"DROP TABLE IF EXISTS {t}")
        self.conn.execute("CHECKPOINT")

    def pids(self):
        return [p.pid for p in psutil.process_iter(["name"]) if p.info["name"] in ("postgres", "postmaster")]

    def create(self, D, default=False, name=None):
        name = name or self.coll
        self.conn.execute(f"DROP TABLE IF EXISTS {name}")
        self.conn.execute(f"CREATE TABLE {name} (id int PRIMARY KEY, emb vector({D}))")
        self.default = default

    def insert(self, X, name=None):
        name = name or self.coll
        with self.conn.cursor() as cur:
            with cur.copy(f"COPY {name} (id, emb) FROM STDIN WITH (FORMAT BINARY)") as cp:
                cp.set_types(["int4", "vector"])
                for i in range(len(X)):
                    cp.write_row((i, np.asarray(X[i], dtype=np.float32)))
        if name != self.coll:     # tiny table: index too
            self.conn.execute(f"CREATE INDEX ON {name} USING hnsw (emb vector_cosine_ops)")

    def build(self, N):
        w = "" if self.default else f" WITH (m = {COMMON['M']}, ef_construction = {COMMON['EF_C']})"
        t = time.perf_counter()
        self.conn.execute(f"CREATE INDEX items_hnsw ON items USING hnsw (emb vector_cosine_ops){w}")
        hnsw_s = time.perf_counter() - t
        # Post-bulk-load maintenance (PostgreSQL best practice): sets hint bits + visibility map so index scans
        # don't pay first-touch heap costs, and prevents autovacuum from kicking in during measurements.
        t = time.perf_counter()
        self.conn.execute("VACUUM (ANALYZE) items")
        return dict(indexed_vectors=N, hnsw_build_s=hnsw_s, vacuum_s=time.perf_counter() - t)

    def warm(self):
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS pg_prewarm")
        self.conn.execute("SELECT pg_prewarm('items'), pg_prewarm('items_hnsw')")

    def _plans(self, lit, k):
        """(custom, generic) plan of the benchmark query for this session's settings: 'hnsw' or 'NO-INDEX'."""
        self.conn.execute(f"PREPARE chk(vector) AS SELECT id FROM items ORDER BY emb <=> $1 LIMIT {int(k)}")
        out = []
        try:
            for mode in ("force_custom_plan", "force_generic_plan"):
                self.conn.execute(f"SET plan_cache_mode = {mode}")
                plan = "\n".join(r[0] for r in self.conn.execute(f"EXPLAIN EXECUTE chk({lit})").fetchall())
                out.append("hnsw" if "items_hnsw" in plan else "NO-INDEX")
        finally:
            self.conn.execute("RESET plan_cache_mode")
            self.conn.execute("DEALLOCATE chk")
        return out

    def check_plan(self, q, grid):
        """grid: [(k, ef)], ef=None means the system default. Records where PostgreSQL's planner, left alone,
        would NOT use the HNSW index (a reportable finding). For the common config (force_index) it then
        verifies that every (k, ef) uses the index with enable_seqscan=off, and raises otherwise, so a silent
        exact scan can never end up in the ef sweep."""
        lit = "'[" + ",".join(f"{float(x):.6f}" for x in q) + "]'"
        force = not self.default
        noidx = []
        try:
            for k, ef in grid:
                self.conn.execute(f"SET hnsw.ef_search = {int(ef)}" if ef else "RESET hnsw.ef_search")
                c, g = self._plans(lit, k)
                if c != "hnsw" or g != "hnsw":
                    noidx.append(f"k{k}_ef{ef or 'default'}_{'custom' if c != 'hnsw' else 'generic'}")
            if force:
                self.conn.execute("SET enable_seqscan = off")
                bad = []
                for k, ef in grid:
                    self.conn.execute(f"SET hnsw.ef_search = {int(ef)}")
                    if self._plans(lit, k) != ["hnsw", "hnsw"]:
                        bad.append((k, ef))
                if bad:
                    raise RuntimeError(f"pgvector plan does not use HNSW even with enable_seqscan=off: {bad}")
        finally:
            self.conn.execute("RESET enable_seqscan")
            self.conn.execute("RESET hnsw.ef_search")
        return dict(pg_planner_noindex=noidx, pg_force_index=force)

    def footprint(self):
        tot = self.conn.execute("SELECT pg_total_relation_size('items')").fetchone()[0]
        idx = self.conn.execute("SELECT pg_relation_size('items_hnsw')").fetchone()[0]
        return dict(disk_mb=tot / 2**20, index_mb=idx / 2**20)

    def search_conf(self, ef):
        return dict(coll=self.coll, ef=ef, dsn=self.DSN, force_index=not self.default)

    def set_ef(self, ef):
        pass

    def drop(self, name):
        self.conn.execute(f"DROP TABLE IF EXISTS {name}")

    def stop(self):
        try:
            self.conn.execute("DROP TABLE IF EXISTS items")
            self.conn.execute("DROP TABLE IF EXISTS tiny")
            self.conn.close()
        except Exception:
            pass
        sh(f"pg_ctlcluster {self.pg_major} main stop", check=False)


ADAPTERS = {"qdrant": Qdrant, "weaviate": Weaviate, "pgvector": PgVector}


# ============================================================================ data loading
def data_root():
    if Path(DATA_HINT, "meta.json").exists():
        return Path(DATA_HINT)
    hits = glob.glob("/kaggle/input/**/meta.json", recursive=True)
    if not hits:
        raise FileNotFoundError("S1 dataset (meta.json) not found under /kaggle/input")
    return Path(hits[0]).parent


def load_dbpedia(n_docs, n_q):
    import pyarrow.parquet as pq
    import faiss
    cache = ROOT / "dbpedia"
    xp, qp, gp = cache / "X.npy", cache / "Q.npy", OUT / "dbpedia_gt100.npy"
    if not xp.exists():
        from huggingface_hub import snapshot_download
        print("   downloading dbpedia-entities-openai-1M from Hugging Face ...", flush=True)
        p = snapshot_download("KShivendu/dbpedia-entities-openai-1M", repo_type="dataset",
                              allow_patterns=["*.parquet"], local_dir=str(ROOT / "dl" / "dbpedia"))
        files = sorted(glob.glob(f"{p}/**/*.parquet", recursive=True))
        total = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
        A, pos = None, 0                                 # preallocate: keeps peak RAM ~1x the matrix
        for f in files:
            col = pq.read_table(f, columns=["openai"]).column("openai").combine_chunks()
            blk = col.flatten().to_numpy(zero_copy_only=False).astype(np.float32).reshape(len(col), -1)
            if A is None:
                A = np.empty((total, blk.shape[1]), dtype=np.float32)
            A[pos:pos + len(blk)] = blk
            pos += len(blk)
            del col, blk
        for s in range(0, total, 100_000):
            A[s:s + 100_000] /= np.linalg.norm(A[s:s + 100_000], axis=1, keepdims=True)
        perm = np.random.default_rng(SEED).permutation(total)
        cache.mkdir(parents=True, exist_ok=True)
        np.save(qp, A[perm[:n_q]])                       # held-out queries (same distribution)
        keep = np.sort(perm[n_q:n_q + n_docs])
        out = np.lib.format.open_memmap(xp, mode="w+", dtype=np.float32, shape=(len(keep), A.shape[1]))
        for s in range(0, len(keep), 100_000):
            out[s:s + 100_000] = A[keep[s:s + 100_000]]
        out.flush()
        del A, out
        shutil.rmtree(ROOT / "dl" / "dbpedia", ignore_errors=True)   # free ~6 GB of parquet
    X, Q = np.load(xp, mmap_mode="r"), np.load(qp)
    if not gp.exists():
        print("   exact GT (FAISS flat IP) ...", flush=True)
        idx = faiss.IndexFlatIP(X.shape[1])
        for s in range(0, len(X), 100_000):
            idx.add(np.ascontiguousarray(X[s:s + 100_000]))
        _, I = idx.search(Q, 100)
        np.save(gp, I.astype(np.int32))
        del idx
    return X, Q, np.load(gp)


def load_job(dataset, model):
    if dataset == "dbpedia1m":
        X, Q, GT = load_dbpedia(P["DBPEDIA_N"], P["QUERY_CAP"])
    else:
        d = data_root() / dataset / model
        X = np.load(d / "docs.npy", mmap_mode="r")
        Q, GT = np.load(d / "queries.npy"), np.load(d / "gt_top100.npy")
    if len(Q) > P["QUERY_CAP"]:
        sel = np.random.default_rng(SEED).choice(len(Q), P["QUERY_CAP"], replace=False)
        sel.sort()
        Q, GT = Q[sel], GT[sel]
    return X, np.ascontiguousarray(Q, dtype=np.float32), GT


# ============================================================================ measurement routines
def sweep(db, X, Q, GT, k, efs, passes, sampler_pids):
    rows, ids_keep = [], {}
    for ef in efs:
        db.set_ef(ef)
        f, close = vdb_worker.make_searcher(db.name, db.search_conf(ef))
        for i in range(min(100, len(Q))):           # warm-up
            f(Q[i], k)
        for p in range(passes):
            lat, pred = [], []          # no memory sampling here: reading smaps would perturb latency
            for q in Q:
                t = time.perf_counter()
                pred.append(f(q, k))
                lat.append(time.perf_counter() - t)
            rec = recall_rows(pred, GT, k)
            n_ret = np.array([len(x) for x in pred])
            rows.append(dict(k=k, ef=ef, pass_=p, recall=float(rec.mean()), recall_std=float(rec.std()),
                             qps=len(Q) / sum(lat), mean_returned=float(n_ret.mean()), **lat_stats(lat)))
            if p == 0 and k == 100:
                ids_keep[ef] = np.array([x + [-1] * (k - len(x)) for x in pred], dtype=np.int32)
        close()
    return rows, ids_keep


def concurrency(db, Q, ef, levels, dur, reps):
    import multiprocessing as mp
    qpath = ROOT / "conc_queries.npy"
    np.save(qpath, Q)
    db.set_ef(ef)
    ctx = mp.get_context("spawn")
    rows = []
    for n in levels:
        for r in range(reps):
            t_start = time.time() + 15 + 2 * n     # leave time for spawn + imports + connect + warm-up
            args = [(db.name, db.search_conf(ef), str(qpath), j * 997, dur, 10, t_start) for j in range(n)]
            with ctx.Pool(n) as pool:
                lats = pool.map(vdb_worker.run_worker, args)
            allv = [x for l in lats for x in l]
            rows.append(dict(clients=n, rep=r, ef=ef, total_queries=len(allv), qps=len(allv) / dur, **lat_stats(allv)))
    return rows


def null_query(db, D):
    db.create(D, name="tiny" if db.name != "weaviate" else "Tiny")
    v = np.random.default_rng(1).standard_normal((1, D)).astype(np.float32)
    v /= np.linalg.norm(v)
    db.insert(v, name="tiny" if db.name != "weaviate" else "Tiny")
    time.sleep(1)
    conf = dict(db.search_conf(None))
    conf["coll"] = "tiny" if db.name != "weaviate" else "Tiny"
    f, close = vdb_worker.make_searcher(db.name, conf)
    for _ in range(50):
        f(v[0], 1)
    lat = []
    for _ in range(500):
        t = time.perf_counter()
        f(v[0], 1)
        lat.append(time.perf_counter() - t)
    close()
    db.drop(conf["coll"])
    return lat_stats(lat)


def pick_ef(sweep_rows, target):
    by = {}
    for r in sweep_rows:
        if r["k"] == 10:
            by.setdefault(r["ef"], []).append(r["recall"])
    for ef in sorted(by):
        if np.mean(by[ef]) >= target:
            return ef, True
    return max(by), False


def run_unit(job, db_name, rep, X, Q, GT, default=False):
    dataset, model = job
    N, D = X.shape
    base = dict(dataset=dataset, model=model, db=db_name, rep=rep, config="default" if default else "common",
                N=N, D=D, nq=len(Q))
    if db_name == "pgvector":
        base["pg_force_index"] = not default      # rows without this field come from the pre-fix code
    db = ADAPTERS[db_name]()
    try:
        t = time.perf_counter()
        stage(f"{db_name}: start")
        db.start()
        base["startup_s"] = time.perf_counter() - t
        if rep == 0 and not default:
            emit(dict(base, phase="null_query", **null_query(db, D)))
        idle = pss_of(db.pids())
        db.create(D, default=default)
        with MemSampler(db.pids) as ms:
            t0 = time.perf_counter()
            stage(f"{db_name}: insert {N} vectors")
            db.insert(X)
            t1 = time.perf_counter()
            stage(f"{db_name}: build index (insert took {t1 - t0:.0f}s)")
            info = db.build(N)
            t2 = time.perf_counter()
        time.sleep(2)
        ing = dict(base, phase="ingest", insert_s=t1 - t0, build_s=t2 - t1, ingest_total_s=t2 - t0,
                   ingest_vps=N / (t2 - t0), pss_idle_mb=idle / 2**20, pss_peak_ingest_mb=ms.peak / 2**20,
                   pss_steady_mb=pss_of(db.pids()) / 2**20, **info, **db.footprint())
        if db_name == "pgvector":
            ing["pg_settings"] = PgVector.settings
        ing["disk_free_gb"] = shutil.disk_usage(str(ROOT)).free / 2**30
        # Warm-up, identical for all systems: explicit cache prewarm (pgvector) + one full pass over all
        # queries at the largest ef, so every system is measured in the same steady (hot) state.
        t = time.perf_counter()
        stage(f"{db_name}: warm-up")
        db.warm()
        if db_name == "pgvector":
            grid = [(10, None), (100, None)] if default else \
                [(10, e) for e in P["EF10"]] + [(100, e) for e in P["EF100"]]
            ing.update(db.check_plan(Q[0], grid))
            print(f"   [pgvector] plan check OK (force_index={ing['pg_force_index']}); planner alone would skip "
                  f"HNSW at: {ing['pg_planner_noindex'] or 'none'}", flush=True)
        warm_ef = None if default else max(P["EF10"] + P["EF100"])
        if warm_ef:
            db.set_ef(warm_ef)
        f, close = vdb_worker.make_searcher(db_name, dict(db.search_conf(warm_ef)))
        for q in Q:
            f(q, 10)
        close()
        ing["warmup_s"] = time.perf_counter() - t
        emit(ing)
        print(f"   [{db_name} rep{rep} {base['config']}] ingest {ing['ingest_total_s']:.1f}s "
              f"(insert {ing['insert_s']:.1f} + build {ing['build_s']:.1f}), PSS peak {ing['pss_peak_ingest_mb']:.0f} MB, "
              f"disk {ing['disk_mb']:.0f} MB", flush=True)

        if default:      # system-default search parameters: no ef override
            stage(f"{db_name}: default-sweep")
            for k in (10, 100):
                f, close = vdb_worker.make_searcher(db_name, dict(db.search_conf(None)))
                for i in range(min(100, len(Q))):
                    f(Q[i], k)
                for p in range(P["PASSES"]):
                    lat, pred = [], []
                    for q in Q:
                        t = time.perf_counter()
                        pred.append(f(q, k))
                        lat.append(time.perf_counter() - t)
                    emit(dict(base, phase="sweep", k=k, ef="default", pass_=p,
                              recall=float(recall_rows(pred, GT, k).mean()), qps=len(Q) / sum(lat),
                              mean_returned=float(np.mean([len(x) for x in pred])), **lat_stats(lat)))
                close()
            return

        all_sweep = []
        stage(f"{db_name}: sweep")
        for k, efs in ((10, P["EF10"]), (100, P["EF100"])):
            rows, ids = sweep(db, X, Q, GT, k, efs, P["PASSES"], db.pids)
            for r in rows:
                emit(dict(base, phase="sweep", **r))
            all_sweep += rows
            if ids and rep == 0:
                np.savez_compressed(OUT / f"ids_{dataset}_{model}_{db_name}_k{k}.npz", **{f"ef{e}": v for e, v in ids.items()})
        best = {e: max(r["recall"] for r in all_sweep if r["k"] == 10 and r["ef"] == e) for e in P["EF10"]}
        print(f"   [{db_name} rep{rep}] recall@10 by ef: " + ", ".join(f"{e}:{v:.3f}" for e, v in best.items()), flush=True)

        if rep == P["REPS"] - 1:
            stage(f"{db_name}: concurrency")
            for target in P["RECALL_TARGETS"]:
                ef, reached = pick_ef(all_sweep, target)
                for r in concurrency(db, Q, ef, P["CONC"], P["CONC_DUR"], P["CONC_REPS"]):
                    emit(dict(base, phase="concurrency", recall_target=target, target_reached=reached,
                                **{("conc_rep" if kk == "rep" else kk): vv for kk, vv in r.items()}))
                print(f"   [{db_name}] concurrency done at ef={ef} (target {target}, reached={reached})", flush=True)
        emit(dict(base, phase="post_search_memory", pss_mb=pss_of(db.pids()) / 2**20))
    finally:
        stage(f"{db_name}: stop")
        try:
            db.stop()
        except Exception as e:
            print(f"   stop() failed for {db_name}: {e}", flush=True)


# ============================================================================ main
def env_info():
    cpu = ""
    try:
        cpu = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
    except Exception:
        pass
    from importlib.metadata import version
    vers = {}
    for p in ("qdrant-client", "weaviate-client", "psycopg", "pgvector", "faiss-cpu", "numpy"):
        try:
            vers[p] = version(p)
        except Exception:
            vers[p] = None
    disks = {}
    for p in ("/tmp", "/kaggle/working"):
        if Path(p).exists():
            u = shutil.disk_usage(p)
            disks[p] = dict(total_gb=round(u.total / 2**30, 1), free_gb=round(u.free / 2**30, 1))
    return dict(preset=PRESET, cfg=P, common=COMMON, versions=VERSIONS, py_pkgs=vers, cpu=cpu, disks=disks,
                logical_cpus=os.cpu_count(), ram_gb=round(psutil.virtual_memory().total / 2**30, 1),
                python=sys.version.split()[0], platform=platform.platform(), started=time.strftime("%Y-%m-%d %H:%M:%S"))


def main():
    info = env_info()
    (OUT / f"env_{PRESET}_{int(time.time())}.json").write_text(json.dumps(info, indent=2, default=str))
    print(">> Env:", info["cpu"], info["logical_cpus"], "vCPU", info["ram_gb"], "GB", "| disks:", info["disks"], flush=True)
    need = {"A": 15, "A1": 15, "A2": 15, "B": 30}.get(PRESET, 5)
    if info["disks"].get("/tmp", {}).get("free_gb", 1e9) < need:
        print(f"!! WARNING: only {info['disks']['/tmp']['free_gb']} GB free on /tmp; preset {PRESET} needs ~{need} GB", flush=True)
    rng = random.Random(SEED)
    t_all = time.time()
    units = []
    for job in P["JOBS"]:
        for rep in range(P["REPS"]):
            order = DBS[:]
            rng.shuffle(order)                               # randomised DB order per rep
            units += [(job, db, rep, False) for db in order]
        if job in P["DEFAULT_JOBS"]:
            units += [(job, db, 0, True) for db in DBS]
    loaded = {}
    for i, (job, db, rep, default) in enumerate(units):
        key = f"{PRESET}|{job[0]}|{job[1]}|{db}|{rep}|{'default' if default else 'common'}"
        if db == "pgvector" and PRESET not in ("SMOKE", "A1"):
            key += "|pgfix"      # pgvector units of A2/B are (re)run with the plan fix; A1 was verified clean
        if key in done_set():
            continue
        if job not in loaded:
            loaded.clear()
            print(f"\n>> Loading {job} ...", flush=True)
            loaded[job] = load_job(*job)
        X, Q, GT = loaded[job]
        el = (time.time() - t_all) / 60
        print(f"\n>> unit {i+1}/{len(units)}  {key}   (elapsed {el:.0f} min)", flush=True)
        try:
            run_unit(job, db, rep, X, Q, GT, default)
            mark_done(key)
        except Exception as e:
            tb = traceback.format_exc()
            print(tb, flush=True)
            emit(dict(dataset=job[0], model=job[1], db=db, rep=rep, phase="error",
                      config="default" if default else "common", error=f"{type(e).__name__}: {e}", traceback=tb))
    summarize()
    STAGE.update(name="done", t=time.time())
    print(f"\n>> Finished in {(time.time() - t_all) / 60:.0f} min")


def summarize():
    if not ROWS.exists():
        return
    df = pd.DataFrame([json.loads(l) for l in open(ROWS)])
    pd.set_option("display.width", 250, "display.max_columns", 40)
    if "phase" in df:
        ing = df[df.phase == "ingest"]
        if len(ing):
            cols = [c for c in ["ingest_total_s", "insert_s", "build_s", "pss_peak_ingest_mb", "pss_steady_mb",
                                "disk_mb", "disk_excl_wal_mb", "index_mb"] if c in ing]
            t = ing.groupby(["dataset", "model", "config", "db"])[cols].agg(["mean", "std"]).round(1)
            t.to_csv(OUT / "summary_ingest.csv")
            print("\n===== INGEST =====\n", t.to_string())
        sw = df[(df.phase == "sweep")]
        if len(sw):
            t = sw.groupby(["dataset", "model", "config", "db", "k", "ef"])[["recall", "qps", "p50_ms", "p99_ms"]].mean().round(4)
            t.to_csv(OUT / "summary_sweep.csv")
            print("\n===== SWEEP (k=10, means) =====\n", t.xs(10, level="k").to_string())
        nq = df[df.phase == "null_query"]
        if len(nq):
            print("\n===== NULL QUERY (protocol overhead) =====\n",
                  nq.groupby(["db"])[["p50_ms", "p95_ms"]].mean().round(3).to_string())
        cc = df[df.phase == "concurrency"]
        if len(cc):
            t = cc.groupby(["dataset", "model", "db", "recall_target", "clients"])[["ef", "qps", "p50_ms", "p99_ms"]].mean().round(2)
            t.to_csv(OUT / "summary_concurrency.csv")
            print("\n===== CONCURRENCY =====\n", t.to_string())
        er = df[df.phase == "error"]
        if len(er):
            print("\n===== ERRORS =====\n", er[["dataset", "model", "db", "rep", "error"]].to_string())
    shutil.make_archive(str(OUT.parent / "s2_results"), "zip", OUT)
    try:
        from IPython.display import FileLink, display
        display(FileLink("s2_results.zip"))
    except Exception:
        pass
    print(f"Results: {OUT}  |  archive: {OUT.parent / 's2_results.zip'}")


if __name__ == "__main__":
    main()
