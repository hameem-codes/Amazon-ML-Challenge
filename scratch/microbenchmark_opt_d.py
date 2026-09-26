"""
Clean Microbenchmark for Optimization D: Blocking Evidence Bitmask
Measures:
1. Per-candidate evidence aggregation time (5 separate lists vs 1 uint8 bitmask)
2. In-memory DataFrame footprint per chunk / retained candidates
3. Vectorized bitmask decoding throughput (reconstructing all 5 channels + num_keys)
"""

import time, sys
import numpy as np
import pandas as pd

def run_microbenchmark():
    print("=" * 65)
    print("MICROBENCHMARK: SEPARATE EVIDENCE LISTS VS COMPACT BITMASK")
    print("=" * 65)

    N_cand = 200000

    # Test 1: Aggregation & List Appends
    print(f"\n[Test 1] Storing evidence for {N_cand:,} candidate pairs:")
    # Simulate random channel firings
    np.random.seed(42)
    b_name = np.random.choice([0, 1], N_cand)
    b_p3 = np.random.choice([0, 1], N_cand)
    b_p4 = np.random.choice([0, 1], N_cand)
    b_addr = np.random.choice([0, 1], N_cand)

    # Baseline: 5 separate lists
    t0 = time.time()
    out_b_ctry = []
    out_b_name = []
    out_b_p3 = []
    out_b_p4 = []
    out_b_addr = []
    out_num_keys = []
    for i in range(N_cand):
        n, p3, p4, a = b_name[i], b_p3[i], b_p4[i], b_addr[i]
        out_b_ctry.append(1)
        out_b_name.append(n)
        out_b_p3.append(p3)
        out_b_p4.append(p4)
        out_b_addr.append(a)
        out_num_keys.append(n + p3 + p4 + a)
    t_base_agg = time.time() - t0

    # Optimization D: 1 bitmask list
    t0 = time.time()
    out_mask = []
    for i in range(N_cand):
        n, p3, p4, a = b_name[i], b_p3[i], b_p4[i], b_addr[i]
        mask = 1 | (n << 1) | (p3 << 2) | (p4 << 3) | (a << 4)
        out_mask.append(mask)
    t_opt_agg = time.time() - t0

    agg_speedup = t_base_agg / t_opt_agg
    print(f"  Baseline (6 list appends):   {t_base_agg*1000:.2f} ms")
    print(f"  Opt D (1 bitmask append):    {t_opt_agg*1000:.2f} ms")
    print(f"  Speedup:                     {agg_speedup:.2f}x faster ({((t_base_agg - t_opt_agg)/t_base_agg)*100:.1f}% reduction)")

    # Test 2: DataFrame Memory Footprint
    print(f"\n[Test 2] Intermediate Candidate Chunk Memory Footprint ({N_cand:,} rows):")
    df_base = pd.DataFrame({
        'blocked_country': np.array(out_b_ctry, dtype=np.int8),
        'blocked_name_token': np.array(out_b_name, dtype=np.int8),
        'blocked_prefix_3': np.array(out_b_p3, dtype=np.int8),
        'blocked_prefix_4': np.array(out_b_p4, dtype=np.int8),
        'blocked_address_token': np.array(out_b_addr, dtype=np.int8),
        'num_blocking_keys': np.array(out_num_keys, dtype=np.int8)
    })
    df_opt = pd.DataFrame({
        'evidence_mask': np.array(out_mask, dtype=np.uint8)
    })

    mem_base = df_base.memory_usage(deep=True).sum() / (1024 * 1024)
    mem_opt = df_opt.memory_usage(deep=True).sum() / (1024 * 1024)
    mem_reduc = ((mem_base - mem_opt) / mem_base) * 100
    print(f"  Baseline (6 evidence columns):  {mem_base:.2f} MB")
    print(f"  Opt D (1 bitmask column):       {mem_opt:.2f} MB")
    print(f"  Memory reduction:               {mem_reduc:.1f}% smaller")

    # Test 3: Vectorized Bitmask Decoding Throughput
    print(f"\n[Test 3] Vectorized decoding of {N_cand:,} bitmasks into 5 channels + num_keys:")
    mask_arr = df_opt['evidence_mask'].values
    t0 = time.time()
    dec_ctry = ((mask_arr >> 0) & 1).astype(np.int8)
    dec_name = ((mask_arr >> 1) & 1).astype(np.int8)
    dec_p3 = ((mask_arr >> 2) & 1).astype(np.int8)
    dec_p4 = ((mask_arr >> 3) & 1).astype(np.int8)
    dec_addr = ((mask_arr >> 4) & 1).astype(np.int8)
    dec_num_keys = (dec_name + dec_p3 + dec_p4 + dec_addr).astype(np.int8)
    t_decode = time.time() - t0

    print(f"  Decoding duration: {t_decode*1000:.2f} ms ({N_cand/t_decode:,.0f} rows/sec)")
    
    # Assert exact bit-for-bit equivalence
    assert np.array_equal(dec_ctry, df_base['blocked_country'].values)
    assert np.array_equal(dec_name, df_base['blocked_name_token'].values)
    assert np.array_equal(dec_p3, df_base['blocked_prefix_3'].values)
    assert np.array_equal(dec_p4, df_base['blocked_prefix_4'].values)
    assert np.array_equal(dec_addr, df_base['blocked_address_token'].values)
    assert np.array_equal(dec_num_keys, df_base['num_blocking_keys'].values)
    print("  Decoded channels equivalence:   100% Bit-for-Bit Identical!")
    print("=" * 65)

if __name__ == '__main__':
    run_microbenchmark()
