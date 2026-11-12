| effect                                                  | dataset      |    dEM |     lo |     hi |      p |   p_holm |
|:--------------------------------------------------------|:-------------|-------:|-------:|-------:|-------:|---------:|
| Retrieval ceiling (oracle − exact dense)                | SQuAD        |  0.153 |  0.126 |  0.18  | 0      |   0      |
| Context vs none (exact dense − closed book)             | SQuAD        |  0.518 |  0.486 |  0.55  | 0      |   0      |
| Embedding model (bge-base − bge-small)                  | SQuAD        |  0.046 |  0.023 |  0.068 | 0.0001 |   0.0004 |
| Hybrid vs dense (exact)                                 | SQuAD        |  0.02  | -0.002 |  0.042 | 0.0953 |   0.4763 |
| ANN at ef=10 (Weaviate − exact)                         | SQuAD        | -0.072 | -0.091 | -0.054 | 0      |   0      |
| DB choice at ef=128 (largest pair: Weaviate − pgvector) | SQuAD        |  0.005 | -0.001 |  0.011 | 0.1797 |   0.7188 |
| Retrieval ceiling (oracle − exact dense)                | HotpotQA-sub |  0.165 |  0.139 |  0.191 | 0      |   0      |
| Context vs none (exact dense − closed book)             | HotpotQA-sub |  0.301 |  0.269 |  0.333 | 0      |   0      |
| Embedding model (bge-base − bge-small)                  | HotpotQA-sub | -0.001 | -0.022 |  0.02  | 1      |   1      |
| Hybrid vs dense (exact)                                 | HotpotQA-sub |  0.005 | -0.016 |  0.026 | 0.7117 |   1      |
| ANN at ef=10 (Weaviate − exact)                         | HotpotQA-sub | -0.059 | -0.078 | -0.04  | 0      |   0      |
| DB choice at ef=128 (largest pair: Qdrant − pgvector)   | HotpotQA-sub |  0.003 |  0     |  0.007 | 0.25   |   0.75   |
