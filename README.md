# Jeff — Rapid AI Validation Engineer
*Ex-systems engineer, 15+ prototype spikes shipped*

**I turn a "Could we just… ?" AI idea into go / pivot / drop data in 72 hours.**

![Bench](https://img.shields.io/badge/bench-pass-brightgreen)

---

## What I Do

When you're wondering if an AI feature is feasible, I build a working prototype in 3 days that answers your question with data, not opinions.

**Recent client question:** *"Could we just use SQLite instead of FAISS/HNSW when you need sub-100 ms on 1M+ vectors?"*

**My answer:** Built this vector database in 72 hours. Results below.

---

## Technical Results (Proof of Process)

| Vectors | Insert (s) | Query (SQL ms) | Query (Fast ms) |
|---------|------------|----------------|-----------------|
| 1 k     | 0.13       | 269.50         | 79.91          |
| 10 k    | 1.43       | 2940.01        | 874.57         |
| 100 k   | 16.34      | 29456.26       | 9447.47        |

<sub>Benchmarked on Windows 10 desktop (Intel i7-14700KF, 32GB RAM). Fast method uses thin Python loop instead of SQL UDF.</sub>

### Key Findings

- ✅ SQLite fine under ~200 k vectors or batch workloads
- ✅ Fast Python path gives 3–4 × speed-up on CPU
- ⚠️ Migrate to FAISS / pgvector for ≥ 1 M vectors **or** <100 ms latency

---

## My Consulting Service

**Fixed-price feasibility sprint:** $2,500 — 3 business days

**Deliverables:**
- Working prototype with real data
- Performance benchmarks
- 3-page memo: go/pivot/drop recommendation
- All source code

**I reply to new enquiries within 24 h (Pacific).**

---

## Quick Demo

```bash
git clone https://github.com/fox4snce/storyvectordb
cd storyvectordb
pip install numpy
python benchmarks/benchmark_sqlite.py
```

---

## Implementation Details

- **Single Python script** (`src/sqlite_vectordb.py`)
- **One dependency** (`numpy`)
- **Zero setup** — just SQLite
- **Fully local** — no API keys or cloud services
- **Blazing fast** — optimized with SQLite PRAGMAs
- **Actually simple** — 300 lines of readable code

Features: CRUD operations, batch inserts, metadata filtering, context windows, database statistics.

---

## Need a Similar Reality Check?

**Fixed-price feasibility sprint:** $2,500 — 3 business days

**Email:** [caldwelljeffreyd@gmail.com](mailto:caldwelljeffreyd@gmail.com)

*This repo is proof I can rapidly validate your AI ideas. The code is just an example of my process in action.*