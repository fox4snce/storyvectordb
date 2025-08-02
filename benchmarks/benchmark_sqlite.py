"""
Benchmark: insert + search for 1 k, 10 k, 100 k vectors on CPU.
Runs in < 10 min on a 4-core laptop.
"""

import time
import random
import numpy as np
import sys
import os
import sqlite3

# Add src to path so we can import the module
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from sqlite_vectordb import SQLiteVectorDB

SIZES = [1_000, 10_000, 100_000]
DIM = 256
TOP_K = 5
TMPDB = "tmp_vectors.db"
np.random.seed(42)

def random_vec(dim=DIM):
    v = np.random.randn(dim).astype(float)
    return (v / np.linalg.norm(v)).tolist()

def init_db():
    """Initialize database with performance optimizations."""
    conn = sqlite3.connect(TMPDB)
    conn.execute("PRAGMA journal_mode = MEMORY")
    conn.execute("PRAGMA synchronous = OFF")
    conn.execute("PRAGMA temp_store = MEMORY")
    conn.execute("PRAGMA cache_size = 100000")
    conn.close()

def clear_db():
    """Clear the database completely."""
    if os.path.exists(TMPDB):
        os.remove(TMPDB)

def run():
    print("SQLite VectorDB Benchmark")
    print("=" * 40)
    print(f"Testing sizes: {SIZES}")
    print(f"Vector dimension: {DIM}")
    print(f"Top-K: {TOP_K}")
    print()
    
    # Clear and initialize database
    clear_db()
    init_db()
    
    results = []
    for n in SIZES:
        print(f"Testing {n:,} vectors...")
        
        # Create fresh database for each size
        db = SQLiteVectorDB(TMPDB)
            
        vecs = [random_vec() for _ in range(n)]
        
        # Insert benchmark
        t0 = time.time()
        items = [{"content": f"dummy {i}", "vector": v} for i, v in enumerate(vecs)]
        db.insert_batch(items)
        insert_t = time.time() - t0

        # Query benchmark - SQL method
        q = random_vec()
        t0 = time.time()
        _ = db.search(q, top_n=TOP_K)
        query_sql_t = time.time() - t0

        # Query benchmark - Fast Python method
        t0 = time.time()
        _ = db.search_fast(q, top_n=TOP_K)
        query_fast_t = time.time() - t0

        results.append((n, insert_t, query_sql_t, query_fast_t))
        print(f"{n:,}\t{insert_t:.2f}s\t{query_sql_t*1000:.2f}ms (SQL)\t{query_fast_t*1000:.2f}ms (Fast)")

    print("\nResults Summary:")
    print("Vectors\tInsert (s)\tQuery SQL (ms)\tQuery Fast (ms)")
    print("-" * 55)
    for n, it, qs, qf in results:
        print(f"{n:,}\t{it:.2f}\t{qs*1000:.2f}\t{qf*1000:.2f}")

    # write CSV for README table
    results_dir = os.path.dirname(__file__)
    csv_path = os.path.join(results_dir, "results.csv")
    with open(csv_path, "w") as f:
        f.write("vectors,insert_s,query_sql_ms,query_fast_ms\n")
        for n, it, qs, qf in results:
            f.write(f"{n},{it:.2f},{qs*1000:.2f},{qf*1000:.2f}\n")
    
    print(f"\nResults saved to: {csv_path}")

if __name__ == "__main__":
    run() 