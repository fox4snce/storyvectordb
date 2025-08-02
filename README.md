# Jeff — Rapid AI Validation Engineer [![Bench](https://img.shields.io/badge/bench-pass-brightgreen)](benchmarks/results.csv)
*Ex-systems engineer*

**I turn a "Could we just…?" AI idea into go / pivot / drop data in 72 hours.**

---

### Recent client question  
> "Could SQLite replace FAISS for our early-stage vector search?"

I built this repo in two days, ran the numbers, and the answer was: **yes up to ~200 k vectors; switch to FAISS for sub-100 ms at million scale.**

### Micro-benchmark (CPU-only, i7-14700KF)

| Vectors | Insert&nbsp;(s) | SQL&nbsp;query&nbsp;(ms) | Fast (NumPy) ms |
|---------|---------------:|------------------------:|----------------:|
| 1 000   | 0.13 | 270 | **80** |
| 10 000  | 1.43 | 2 940 | **875** |
| 100 000 | 16.3 | 29 456 | **9 447** |

<sub>Measured on Windows 10, Intel i7-14700KF, CPU-only.</sub>

*Fast method = thin NumPy loop, no SQL UDF.*

**Key findings**

- ✅ SQLite fine for < 200 k vectors or batch look-ups  
- ✅ Thin-loop optimisation → 3-4 × speed-up  
- ⚠️ Plan migration to FAISS / pgvector for ≥ 1 M vectors or < 100 ms latency

---

## Fixed-price feasibility sprint — US $2 500 — 3 business days

You get:  

- 🛠️ working prototype & code  
- 📊 benchmark CSV + plots  
- 📝 3-page go / pivot / drop memo  
- 📬 async Q&A support (email or Slack)

**Email me:** [caldwelljeffreyd@gmail.com](mailto:caldwelljeffreyd@gmail.com) (reply < 24 h, Pacific)

---

Copy-paste to run the demo:
```bash
git clone https://github.com/fox4snce/storyvectordb && \
cd storyvectordb && \
pip install numpy && \
python benchmarks/benchmark_sqlite.py
```

---

## Implementation Details

- **Single Python script** (`src/sqlite_vectordb.py`)
- **One dependency** (`numpy`)
- **Zero setup** — just SQLite
- **Fully local** — no API keys or cloud services
- **PRAGMA-tuned; ~0.9 s top-5 query at 10 k vectors on laptop CPU** — optimized with SQLite PRAGMAs
- **Actually simple** — 300 lines of readable code

Features: CRUD operations, batch inserts, metadata filtering, context windows, database statistics.

**License:** MIT
