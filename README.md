# Jeff — Rapid AI Validation Engineer  [![Bench](https://img.shields.io/badge/bench-pass-brightgreen)](benchmarks/results.csv)

I turn a **“Could we just…?”** AI idea into **go / pivot / drop data** in 72 hours.

---

### Recent client question  
> “Could SQLite replace FAISS for our early-stage vector search?”

I built this repo in two days, ran the numbers, and the answer was:

*Yes for ≤ 200 k vectors; move to FAISS / pgvector for sub-100 ms at million-scale.*

---

### Micro-benchmark  (CPU-only • i7-14700KF)

| Vectors | Insert&nbsp;(s) | SQL&nbsp;query&nbsp;(ms) | Fast&nbsp;NumPy&nbsp;(ms) |
|--------:|---------------:|-------------------------:|-------------------------:|
| 1 000   | 0.13 | 270   | **80** |
| 10 000  | 1.43 | 2 940 | **875** |
| 100 000 | 16.3 | 29 456 | **9 447** |

*Fast method = thin NumPy loop (no SQL UDF).*

**Key findings**

- ✅ SQLite is fine for prototypes & KBs < 200 k vectors  
- ✅ Thin-loop optimisation → 3-4 × speed-up on CPU  
- ⚠️ Plan FAISS / pgvector for ≥ 1 M vectors **or** < 100 ms latency

---

## Fixed-price feasibility sprint — **US $2 500 — 3 business days**

**You get**

- 🛠️ working prototype & code  
- 📊 benchmark CSV + plots  
- 📝 3-page go / pivot / drop memo  
- 📬 async Q&A support (email or Slack)  

*I reply to enquiries within 24 h (Pacific).*  
**Email:** <mailto:caldwelljeffreyd@gmail.com>

---

### Quick demo

```bash
git clone https://github.com/fox4snce/storyvectordb
cd storyvectordb
pip install numpy
python benchmarks/benchmark_sqlite.py
```
## Implementation details

- Single Python file (`src/sqlite_vectordb.py`) + NumPy
- Pure SQLite (no external services, no API keys)  
- PRAGMA-tuned; fast enough for 100k-vector look-ups on laptop CPU
- Features: CRUD, batch insert, metadata filter, context window, DB stats

-----
This repo isn't a product—it's proof I can turn an AI idea into hard numbers in three days.
