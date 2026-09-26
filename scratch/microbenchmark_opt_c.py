"""
Microbenchmark: String-ID vs Integer-ID Operations for Candidate Generation
Measures:
1. Candidate tie-breaking and top-k selection (string vs integer keys)
2. DataFrame candidate sorting (string columns vs integer columns)
3. Memory footprint (string objects vs primitive integers)
"""

import time, heapq, sys
import numpy as np
import pandas as pd

def run_microbenchmark():
    print("=" * 60)
    print("MICROBENCHMARK: STRING-ID VS INTEGER-ID OPERATIONS")
    print("=" * 60)

    N_cand = 200000
    needed = 400

    # 1. Candidate tie-breaking & top-k selection
    print(f"\n[Test 1] Top-{needed} selection over {N_cand:,} candidates:")
    eids_str = [f'S2-{i:07d}' for i in range(N_cand)]
    np.random.seed(42)
    np.random.shuffle(eids_str)
    cand_indices = list(range(N_cand))

    order = np.argsort(eids_str)
    lex_rank = np.empty(N_cand, dtype=np.int32)
    lex_rank[order] = np.arange(N_cand, dtype=np.int32)

    # String selection
    t0 = time.time()
    res_str = heapq.nsmallest(needed, cand_indices, key=lambda idx: eids_str[idx])
    t_str_sel = time.time() - t0

    # Integer selection
    t0 = time.time()
    res_int = heapq.nsmallest(needed, cand_indices, key=lambda idx: lex_rank[idx])
    t_int_sel = time.time() - t0

    assert res_str == res_int, "Top-K selection results differ!"
    print(f"  String ID selection:  {t_str_sel*1000:.2f} ms")
    print(f"  Integer ID selection: {t_int_sel*1000:.2f} ms")
    print(f"  Equivalence:          100% Identical")

    # 2. DataFrame Candidate Sorting
    print(f"\n[Test 2] DataFrame sorting over {N_cand:,} rows (2 IDs + score):")
    s1_strs = [f'S1-{i%500:07d}' for i in range(N_cand)]
    cand_strs = [f'S2-{i%100000:07d}' for i in range(N_cand)]
    scores = np.random.choice([30, 50, 60, 70, 100, 120], N_cand).astype(np.float32)

    s1_ints = np.array([int(s.split('-')[1]) for s in s1_strs], dtype=np.int32)
    cand_ints = np.array([int(s.split('-')[1]) for s in cand_strs], dtype=np.int32)

    df_str = pd.DataFrame({'entity_id_s1': s1_strs, 'evidence_score': scores, 'entity_id_cand': cand_strs})
    df_int = pd.DataFrame({'s1_int': s1_ints, 'evidence_score': scores, 'cand_int': cand_ints})

    t0 = time.time()
    _ = df_str.sort_values(by=['entity_id_s1', 'evidence_score', 'entity_id_cand'], ascending=[True, False, True])
    t_str_sort = time.time() - t0

    t0 = time.time()
    _ = df_int.sort_values(by=['s1_int', 'evidence_score', 'cand_int'], ascending=[True, False, True])
    t_int_sort = time.time() - t0

    sort_speedup = t_str_sort / t_int_sort
    print(f"  String DataFrame sort:  {t_str_sort*1000:.2f} ms")
    print(f"  Integer DataFrame sort: {t_int_sort*1000:.2f} ms")
    print(f"  Speedup:                {sort_speedup:.2f}x faster ({((t_str_sort - t_int_sort)/t_str_sort)*100:.1f}% reduction)")

    # 3. Memory Footprint
    mem_str_mb = df_str.memory_usage(deep=True).sum() / (1024 * 1024)
    mem_int_mb = df_int.memory_usage(deep=True).sum() / (1024 * 1024)
    mem_reduction = ((mem_str_mb - mem_int_mb) / mem_str_mb) * 100
    print(f"\n[Test 3] In-Memory Footprint ({N_cand:,} candidate rows):")
    print(f"  String representation:  {mem_str_mb:.2f} MB")
    print(f"  Integer representation: {mem_int_mb:.2f} MB")
    print(f"  Memory reduction:       {mem_reduction:.1f}% smaller")
    print("=" * 60)

if __name__ == '__main__':
    run_microbenchmark()
