.PHONY: benchmark clean

benchmark:
	python benchmarks/benchmark_sqlite.py

clean:
	rm -f tmp_vectors.db benchmarks/results.csv 