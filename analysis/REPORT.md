# S5 analysis report (auto-generated)

## S2 — iso-recall throughput
- Friedman over 19 blocks (unit × k × target): χ² = 21.895, p = 1.8e-05; mean ranks {'qdrant': 1.368, 'weaviate': 2.842, 'pgvector': 1.789} (Nemenyi CD = 0.76).
- Wins per block: {'Qdrant': 12, 'Weaviate': 0, 'pgvector': 7}; by regime: {'k=10, recall<0.99': {'Qdrant': 5, 'Weaviate': 0, 'pgvector': 7}, 'recall=0.99 or k=100': {'Qdrant': 7, 'Weaviate': 0, 'pgvector': 0}}.
- Null-query p50 (ms): {'pgvector': 0.249, 'Qdrant': 1.348, 'Weaviate': 1.529}.
- Planner cliff (quora/bge-base, pgvector k=10 ef=512): {'exact_scan_point': {'rep': 1.0, 'recall': 0.9975, 'qps': 0.8091, 'p50_ms': 1222.9588}, 'hnsw_ef256': {'recall': 0.994, 'qps': 114.917, 'p50_ms': 8.6491}}.

## S3 — retrieval quality
- Friedman (dataset×model blocks) per ef, recall_gt / nDCG@10:
  - ef10_recall_gt: p = 5.22e-06, mean ranks {'qdrant': 2.179, 'weaviate': 2.821, 'pgvector': 1.0}
  - ef10_ndcg10: p = 0.000235, mean ranks {'qdrant': 2.214, 'weaviate': 2.643, 'pgvector': 1.143}
  - ef16_recall_gt: p = 8.32e-07, mean ranks {'qdrant': 2.0, 'weaviate': 3.0, 'pgvector': 1.0}
  - ef16_ndcg10: p = 0.00036, mean ranks {'qdrant': 1.929, 'weaviate': 2.786, 'pgvector': 1.286}
  - ef32_recall_gt: p = 8.78e-06, mean ranks {'qdrant': 2.214, 'weaviate': 2.786, 'pgvector': 1.0}
  - ef32_ndcg10: p = 0.109, mean ranks {'qdrant': 2.071, 'weaviate': 2.357, 'pgvector': 1.571}
  - ef64_recall_gt: p = 3.41e-05, mean ranks {'qdrant': 2.0, 'weaviate': 2.857, 'pgvector': 1.143}
  - ef64_ndcg10: p = 0.0691, mean ranks {'qdrant': 1.929, 'weaviate': 2.464, 'pgvector': 1.607}
  - ef128_recall_gt: p = 0.000538, mean ranks {'qdrant': 2.179, 'weaviate': 2.607, 'pgvector': 1.214}
  - ef128_ndcg10: p = 0.327, mean ranks {'qdrant': 2.036, 'weaviate': 2.25, 'pgvector': 1.714}
  - ef256_recall_gt: p = 0.00537, mean ranks {'qdrant': 2.25, 'weaviate': 2.429, 'pgvector': 1.321}
  - ef256_ndcg10: p = 0.059, mean ranks {'qdrant': 2.143, 'weaviate': 2.321, 'pgvector': 1.536}
- Per-query nDCG@10 Wilcoxon (Holm per ef): {'ef16': {'n_tests': 42, 'n_sig_holm': 0, 'n_equiv_0p01': 5, 'max_abs_diff': 0.01918753795325756}, 'ef64': {'n_tests': 42, 'n_sig_holm': 0, 'n_equiv_0p01': 32, 'max_abs_diff': 0.010440049692988396}, 'ef128': {'n_tests': 42, 'n_sig_holm': 0, 'n_equiv_0p01': 40, 'max_abs_diff': 0.006631179712712765}}
- Mean nDCG@10 gap to exact by ef: {'pgvector': {10: -0.0441, 16: -0.0301, 32: -0.0167, 64: -0.0084, 128: -0.0048, 256: -0.0021}, 'Qdrant': {10: -0.0577, 16: -0.0361, 32: -0.0192, 64: -0.01, 128: -0.0053, 256: -0.0031}, 'Weaviate': {10: -0.0624, 16: -0.0403, 32: -0.0225, 64: -0.0112, 128: -0.0061, 256: -0.0032}}

## S4 — end-to-end RAG
- vs exact dense: {'ef10': {'n': 12, 'n_sig_holm': 12, 'n_equiv': 0, 'dEM_range': [-0.077, -0.042]}, 'ef16': {'n': 12, 'n_sig_holm': 12, 'n_equiv': 0, 'dEM_range': [-0.054, -0.024]}, 'ef32': {'n': 12, 'n_sig_holm': 6, 'n_equiv': 0, 'dEM_range': [-0.031, -0.008]}, 'ef64': {'n': 12, 'n_sig_holm': 1, 'n_equiv': 11, 'dEM_range': [-0.015, 0.001]}, 'ef128': {'n': 12, 'n_sig_holm': 2, 'n_equiv': 12, 'dEM_range': [-0.012, 0.003]}, 'ef256': {'n': 12, 'n_sig_holm': 0, 'n_equiv': 12, 'dEM_range': [-0.006, 0.004]}}
- system pairs: {'ef10': {'n': 12, 'n_sig_holm': 0, 'n_equiv': 0, 'max_abs_dEM': 0.02}, 'ef16': {'n': 12, 'n_sig_holm': 0, 'n_equiv': 3, 'max_abs_dEM': 0.019}, 'ef32': {'n': 12, 'n_sig_holm': 0, 'n_equiv': 11, 'max_abs_dEM': 0.011}, 'ef64': {'n': 12, 'n_sig_holm': 0, 'n_equiv': 12, 'max_abs_dEM': 0.004}, 'ef128': {'n': 12, 'n_sig_holm': 0, 'n_equiv': 12, 'max_abs_dEM': 0.005}, 'ef256': {'n': 12, 'n_sig_holm': 0, 'n_equiv': 12, 'max_abs_dEM': 0.004}}
- squad/bge-base: r(overlap,hit) = 0.996, r(hit,EM) = 0.993, slope = 0.71; logit system p (given hit@5) = {'pgvector': 0.174, 'weaviate': 0.222}, (given overlap@5) = {'pgvector': 0.305, 'weaviate': 0.13}
- squad/bge-small: r(overlap,hit) = 0.991, r(hit,EM) = 0.995, slope = 0.69; logit system p (given hit@5) = {'pgvector': 0.11, 'weaviate': 0.147}, (given overlap@5) = {'pgvector': 0.613, 'weaviate': 0.003}
- hotpotqa_sub/bge-base: r(overlap,hit) = 0.995, r(hit,EM) = 0.987, slope = 0.38; logit system p (given hit@5) = {'pgvector': 0.707, 'weaviate': 0.911}, (given overlap@5) = {'pgvector': 0.403, 'weaviate': 0.617}
- hotpotqa_sub/bge-small: r(overlap,hit) = 0.995, r(hit,EM) = 0.978, slope = 0.32; logit system p (given hit@5) = {'pgvector': 0.861, 'weaviate': 0.941}, (given overlap@5) = {'pgvector': 0.061, 'weaviate': 0.339}

## Effect sizes (EM, bge-base)
- SQuAD: Retrieval ceiling (oracle − exact dense): +15.3 pp [+12.6, +18.0], p_holm = 0
- SQuAD: Context vs none (exact dense − closed book): +51.8 pp [+48.6, +55.0], p_holm = 0
- SQuAD: Embedding model (bge-base − bge-small): +4.6 pp [+2.3, +6.8], p_holm = 0.0004
- SQuAD: Hybrid vs dense (exact): +2.0 pp [-0.2, +4.2], p_holm = 0.48
- SQuAD: ANN at ef=10 (Weaviate − exact): -7.2 pp [-9.1, -5.4], p_holm = 0
- SQuAD: DB choice at ef=128 (largest pair: Weaviate − pgvector): +0.5 pp [-0.1, +1.1], p_holm = 0.72
- HotpotQA-sub: Retrieval ceiling (oracle − exact dense): +16.5 pp [+13.9, +19.1], p_holm = 0
- HotpotQA-sub: Context vs none (exact dense − closed book): +30.1 pp [+26.9, +33.3], p_holm = 0
- HotpotQA-sub: Embedding model (bge-base − bge-small): -0.1 pp [-2.2, +2.0], p_holm = 1
- HotpotQA-sub: Hybrid vs dense (exact): +0.5 pp [-1.6, +2.6], p_holm = 1
- HotpotQA-sub: ANN at ef=10 (Weaviate − exact): -5.9 pp [-7.8, -4.0], p_holm = 0
- HotpotQA-sub: DB choice at ef=128 (largest pair: Qdrant − pgvector): +0.3 pp [+0.0, +0.7], p_holm = 0.75
