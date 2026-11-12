| unit                  | db       |   ingest_s |   ingest_sd |   build_s |   pss_peak_mb |   disk_mb |   pss_after_search_mb |
|:----------------------|:---------|-----------:|------------:|----------:|--------------:|----------:|----------------------:|
| FiQA/bge-base         | pgvector |        109 |           1 |       106 |          1006 |       457 |                  1029 |
| FiQA/bge-base         | Qdrant   |         42 |           1 |        36 |           481 |       422 |                   239 |
| FiQA/bge-base         | Weaviate |         45 |           1 |         0 |           682 |       215 |                   727 |
| HotpotQA-sub/bge-base | pgvector |        205 |           2 |       195 |          5439 |      1601 |                  3656 |
| HotpotQA-sub/bge-base | Qdrant   |        200 |           6 |       178 |          1362 |      1008 |                   681 |
| HotpotQA-sub/bge-base | Weaviate |        215 |           5 |         0 |          1734 |       717 |                  2160 |
| Quora/bge-base        | pgvector |        519 |           7 |       473 |         14061 |      4144 |                  8796 |
| Quora/bge-base        | Qdrant   |        446 |           9 |       382 |          3365 |      1764 |                  1662 |
| Quora/bge-base        | Weaviate |        490 |           7 |         0 |          3899 |      1800 |                  5273 |
| Quora/bge-small       | pgvector |        364 |           4 |       354 |          5820 |      1849 |                  4163 |
| Quora/bge-small       | Qdrant   |        355 |           2 |       311 |          1900 |      1534 |                   913 |
| Quora/bge-small       | Weaviate |        413 |           2 |         0 |          2516 |      1058 |                  3078 |
