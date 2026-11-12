# Raw results: Does the Vector Database Matter for Retrieval-Augmented Generation?

<!-- After the first GitHub release is archived by Zenodo, replace this line with the Zenodo DOI badge. -->

This repository holds the raw measurements, per-query results, generated answers and analysis outputs behind the paper:

> *Does the Vector Database Matter for Retrieval-Augmented Generation? An Iso-Recall, End-to-End and Statistically Grounded Comparison of Qdrant, Weaviate and pgvector.*

The systems under test are Qdrant 1.20.0, Weaviate 1.40.4, and PostgreSQL 16.15 with pgvector 0.8.7. All three use a common HNSW configuration (M = 16, efConstruction = 128, cosine distance). The embeddings are BGE models: bge-small-en-v1.5 (384-d) and bge-base-en-v1.5 (768-d). The generator is Qwen2.5-7B-Instruct, 4-bit NF4, with greedy decoding.

The scripts are in `scripts/`; they also run as a notebook at https://kaggle.com/code/talhakabakus/vector-db-comp.

## Layout

```
scripts/            S0–S5 scripts exactly as run (Kaggle notebooks, pinned versions inside each script),
                    06_merge_qnp.py (merges the Qdrant rerun into results/s3 and results/s4)
results/s2/         S2 – system benchmark (E1 ingest/resources, E2 sweep/iso-recall/concurrency/null query, E2b defaults)
results/s3/         S3 – retrieval quality (E3), filtered search (E4), hybrid search (E5)
results/s4/         S4 – end-to-end RAG (E6)
analysis/           S5 – statistics (stats.json), report (REPORT.md), tables (CSV + Markdown), figures (PNG, 600 dpi)
```

## File formats

### `results/s2/`
- `rows.jsonl`: one JSON object per measurement. The `phase` field takes one of these values:
  - `ingest`: insert, build and total time; PSS idle/peak/steady in MiB; disk in MiB; segment and indexed-vector counts.
  - `sweep`: one (k, ef, pass) point with recall, QPS, and p50/p95/p99/mean latency in ms. The `config` field is `common` (M = 16, efC = 128) or `default` (E2b).
  - `concurrency`: QPS and latency for 1, 2, 4 or 8 client processes at a recall target.
  - `null_query`: protocol and client overhead.
  - `post_search_memory`: PSS after the sweep.
  - `error`: one failed attempt that was retried, kept for completeness.
- `summary_*.csv`: per-run summaries written by S2.
- `ids_<dataset>_<model>_<db>_k100.npz`: retrieved document indices for k = 100, one array per ef (`ef100` … `ef512`), with shape (queries × 100).
- `env_*.json`: the environment and configuration of each run.
- `*.log`: server logs.

### `results/s3/`
- `rows.jsonl`: one JSON object per measurement. The `exp` field takes one of these values:
  - `E3`: nDCG@10, MRR@10, recall against relevance judgments, and recall against exact search, per (k, ef).
  - `E4`: filtered search per (selectivity, ef). For pgvector the row also records the iterative-scan mode and the plan chosen by PostgreSQL.
  - `E5`: dense, lexical and hybrid retrieval at ef = 128.
  - `pg_plan`: the plan the unforced PostgreSQL planner would choose.
  - `load`: ingest information for each build (for Qdrant, segment count and the full-scan threshold read back from the server).
- `q_<dataset>_<model>_<db>.npz`: per-query arrays. `E3_k{k}_ef{ef}_ids` holds the retrieved indices, and `_ndcg10` and `_recallgt` hold the per-query metrics. The `E4_*` and `E5_*` arrays follow the same pattern.
- `q_<dataset>_<model>_qdrant.npz` holds the Qdrant build without payload indexes for `E3_*` and `E5_*`, and the build with payload indexes for `E4_*` (see below).
- `q_<dataset>_<model>_exact.npz`: exact-search references, with exact nDCG@10 and the exact dense, lexical (BM25) and hybrid (RRF) result lists.
- `summary_E3/E4/E5.csv`, `env_*.json`, `*.log`: as in S2.

### `results/s4/`
- `rows.jsonl`: one row per (dataset, model, configuration). Each row has EM, F1, the answer-in-context rate, hit@5, all-gold@5, overlap with the exact top-5, and the mean input tokens.
- `s4_<dataset>_<model>.npz`: per-question arrays for every configuration, keyed `<db>__<method>__ef<ef>__<metric>` (for example `qdrant__dense__ef64__em`), or `<db>__<method>__<metric>` for configurations without ef. The metrics are `em`, `f1`, `aic` (answer in context), `hit5`, `allgold5` and `overlap5`. The `pred` arrays hold the generated answers.
- `gen_cache_gpu*.jsonl`: every unique prompt (SHA-1 hash `h`), its generated answer (`out`) and its input length in tokens.
- `s4_worker.py`, `worker_gpu*.log/.out`: the two-GPU generation worker and its logs.

## Rows to exclude (as done by `scripts/05_analysis_figures.py`)

- **Smoke tests:**
  - S3 and S4 rows with `preset == "SMOKE"`. In S4 these are also the rows with `nq == 100`.
  - S2 rows on FiQA with `nq == 500`.
- **Superseded pgvector run (S2):** rows for Quora/bge-base and pgvector without a `preset` field come from a first run. The A2 rows replace them.
- **Planner cliff (S2):** the pgvector point on Quora/bge-base at k = 10 and ef = 512 is an exact sequential scan chosen by the planner. It is reported separately and excluded from the HNSW curves.
- **Qdrant without payload indexes (S3, S4):** rows with `preset == "QNP"` replace Qdrant's E3, E5 and load rows in S3 and Qdrant's configurations in S4. E4 keeps the Q2 build with payload indexes, which filtering needs. The Qdrant arrays in `q_*_qdrant.npz` (E3, E5) and the `qdrant__*` arrays in `s4_*.npz` already hold the QNP results. The earlier Qdrant rows (with payload indexes) remain in `rows.jsonl` for comparison. Files prefixed `qnp_` or `qnpfix_` are the environment files and logs of this rerun (QNPFIX repeated two units that failed an over-strict indexing check in QNP).
- **Qdrant full-scan rerun (S3, superseded by QNP):** for SciFact and NFCorpus with bge-small, the Q1 rows carrying `qdrant_fst_kb` replace the earlier Q1 rows of the same unit. With the default full-scan threshold, these two units returned exact results at every ef; all QNP builds use the lowered threshold.

After these filters, S4 has 96 configurations: 48 per dataset, 1,000 questions each.

## Reproducing the analysis

```bash
python scripts/05_analysis_figures.py --s2 results/s2 --s3 results/s3 --s4 results/s4 --out analysis
```

This regenerates `analysis/stats.json`, `analysis/REPORT.md`, the tables and the figures. The figure files in `analysis/figures` are named by their figure number in the paper. `Fig01_architecture_overview.png` is a diagram drawn by `scripts/architecture_overview.py`; S5 does not produce it.

## Notes

- The corpora, queries and relevance judgments come from BEIR, SQuAD v1.1 and HotpotQA under their own licenses, and are not redistributed here. Document and query indices refer to the order produced by `scripts/01_build_embeddings.py`, which uses a fixed seed of 42.
- Memory values are proportional set size (PSS) in MiB. The paper reports them in GiB.
- In S3, the pgvector and Weaviate builds include the four filter attributes. In Qdrant, payload indexes on these attributes add filter-aware HNSW links and raise unfiltered recall at low ef, so Qdrant's E3, E5 and E6 results come from a build without payload indexes and only E4 from a build with them (see Section 4.2 of the paper).

## License

- Code in `scripts/`: MIT License (`LICENSE`).
- Results, analysis outputs and figures in `results/` and `analysis/`: CC BY 4.0 (`LICENSE-DATA.md`).

## Citation

Please cite the paper and this repository; `CITATION.cff` holds the citation metadata (GitHub shows it under "Cite this repository"). The archived version of each release, with its DOI, is on Zenodo.
