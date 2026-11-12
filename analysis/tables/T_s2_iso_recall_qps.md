| unit                  |   k |   target | Qdrant      | Weaviate    | pgvector    | best                         |
|:----------------------|----:|---------:|:------------|:------------|:------------|:-----------------------------|
| FiQA/bge-base         |  10 |     0.9  | 442 ± 37    | 372 ± 14    | 585 ± 28    | pgvector                     |
| FiQA/bge-base         |  10 |     0.95 | 377 ± 30    | 302 ± 15    | 413 ± 32    | pgvector                     |
| FiQA/bge-base         |  10 |     0.97 | 321 ± 21    | 252 ± 6     | 302 ± 20    | Qdrant                       |
| FiQA/bge-base         |  10 |     0.98 | 261 ± 7     | 204 ± 5     | 236 ± 11    | Qdrant                       |
| FiQA/bge-base         |  10 |     0.99 | 186 ± 4     | 144 ± 3     | 131 ± 10    | Qdrant                       |
| FiQA/bge-base         | 100 |     0.95 | 197 ± 8     | 156 ± 3     | 183 ± 7     | Qdrant                       |
| FiQA/bge-base         | 100 |     0.99 | unreachable | unreachable | 85 ± 0      | pgvector (only one reaching) |
| HotpotQA-sub/bge-base |  10 |     0.9  | 343 ± 4     | 275 ± 6     | 410 ± 27    | pgvector                     |
| HotpotQA-sub/bge-base |  10 |     0.95 | 268 ± 8     | 199 ± 18    | 245 ± 11    | Qdrant                       |
| HotpotQA-sub/bge-base |  10 |     0.97 | 206 ± 6     | 154 ± 2     | 157 ± 8     | Qdrant                       |
| HotpotQA-sub/bge-base |  10 |     0.98 | 160 ± 5     | 121 ± 1     | 107 ± 4     | Qdrant                       |
| HotpotQA-sub/bge-base |  10 |     0.99 | unreachable | unreachable | unreachable | none                         |
| HotpotQA-sub/bge-base | 100 |     0.95 | 136 ± 0     | 104 ± 2     | 96 ± 6      | Qdrant                       |
| HotpotQA-sub/bge-base | 100 |     0.99 | unreachable | unreachable | unreachable | none                         |
| Quora/bge-base        |  10 |     0.9  | ≥ 502       | ≥ 466       | ≥ 633       | n/a (bounds only)            |
| Quora/bge-base        |  10 |     0.95 | 493 ± 65    | 456 ± 69    | ≥ 633       | pgvector (bound)             |
| Quora/bge-base        |  10 |     0.97 | 460 ± 63    | 411 ± 70    | 612 ± 67    | pgvector                     |
| Quora/bge-base        |  10 |     0.98 | 439 ± 53    | 351 ± 49    | 537 ± 72    | pgvector                     |
| Quora/bge-base        |  10 |     0.99 | 334 ± 31    | 246 ± 28    | 325 ± 32    | Qdrant                       |
| Quora/bge-base        | 100 |     0.95 | ≥ 223       | ≥ 176       | ≥ 158       | n/a (bounds only)            |
| Quora/bge-base        | 100 |     0.99 | 198 ± 7     | 124 ± 9     | 125 ± 9     | Qdrant                       |
| Quora/bge-small       |  10 |     0.9  | ≥ 533       | ≥ 430       | ≥ 873       | n/a (bounds only)            |
| Quora/bge-small       |  10 |     0.95 | 528 ± 27    | 404 ± 64    | ≥ 873       | pgvector (bound)             |
| Quora/bge-small       |  10 |     0.97 | 516 ± 18    | 372 ± 67    | 796 ± 47    | pgvector                     |
| Quora/bge-small       |  10 |     0.98 | 498 ± 15    | 309 ± 55    | 689 ± 19    | pgvector                     |
| Quora/bge-small       |  10 |     0.99 | 402 ± 26    | 249 ± 29    | 382 ± 16    | Qdrant                       |
| Quora/bge-small       | 100 |     0.95 | ≥ 227       | ≥ 190       | ≥ 266       | n/a (bounds only)            |
| Quora/bge-small       | 100 |     0.99 | 183 ± 11    | 124 ± 1     | 148 ± 4     | Qdrant                       |
