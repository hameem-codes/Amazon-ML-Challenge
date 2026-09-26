"""
Phase 6.1 Profiling Investigation: Detailed Runtime & Component Breakdown
Profiles the Phase 6.1 blocking engine against the full 1.43M France targets.
Uses microsecond component instrumentation + cProfile.
"""

import sys, os, time, gc, cProfile, pstats, io
sys.path.insert(0, 'src')
import pandas as pd
import numpy as np
import psutil

from normalize import normalize_business_name, normalize_country, normalize_business_address
from blocking import ConfigABlockingEngine
from candidate_safety import apply_candidate_safety_cap
from config import BLOCKING_CHUNK_SIZE, CANDIDATE_CAP_CONFIG, get_config_fingerprint


def profile_blocking_pipeline(num_s1=25):
    print("=" * 70)
    print(f"PHASE 6.1 PROFILING INVESTIGATION ({num_s1} FRANCE S1 vs 1.43M TARGETS)")
    print("=" * 70)

    # 1. Load France targets
    print("\n--- Phase A: Target Ingestion & Normalization ---")
    t0 = time.time()
    s2_chunks = [
        chunk[chunk['country'].apply(normalize_country) == 'france']
        for chunk in pd.read_csv('data/test/test_source2.tsv', sep='\t', dtype=str, na_filter=False, chunksize=500000)
    ]
    s2_fr = pd.concat(s2_chunks, ignore_index=True)
    s2_fr['source'] = 'S2'
    del s2_chunks
    gc.collect()

    s3_chunks = [
        chunk[chunk['country'].apply(normalize_country) == 'france']
        for chunk in pd.read_csv('data/test/test_source3.tsv', sep='\t', dtype=str, na_filter=False, chunksize=500000)
    ]
    s3_fr = pd.concat(s3_chunks, ignore_index=True)
    s3_fr['source'] = 'S3'
    del s3_chunks
    gc.collect()

    targets = pd.concat([s2_fr, s3_fr], ignore_index=True)
    del s2_fr, s3_fr
    gc.collect()

    t_ingest = time.time() - t0
    print(f"Target ingestion: {len(targets):,} records in {t_ingest:.2f}s")

    t0 = time.time()
    targets['norm_name'] = targets['business_name'].apply(normalize_business_name)
    targets['norm_address'] = targets['business_address'].apply(normalize_business_address)
    targets['norm_country'] = 'france'
    t_norm = time.time() - t0
    print(f"Target normalization: {t_norm:.2f}s")

    # 2. Index Construction Breakdown
    print("\n--- Phase B: Index Construction Component Breakdown ---")
    t0_idx_total = time.time()
    
    n_target = len(targets)
    t_names = targets['norm_name'].values
    t_addrs = targets['norm_address'].values
    t_countries = targets['norm_country'].values
    stop_tokens = set(ConfigABlockingEngine.__init__.__defaults__[1])
    stop_addrs = set(ConfigABlockingEngine.__init__.__defaults__[2])

    from collections import defaultdict
    c_name_idx = defaultdict(lambda: defaultdict(list))
    c_p3_idx = defaultdict(lambda: defaultdict(list))
    c_p4_idx = defaultdict(lambda: defaultdict(list))
    c_addr_idx = defaultdict(lambda: defaultdict(list))

    t0_c2 = time.time()
    for i in range(n_target):
        c = t_countries[i]
        name = str(t_names[i]) if pd.notna(t_names[i]) else ''
        for t in name.split():
            if len(t) > 1 and t not in stop_tokens:
                c_name_idx[c][t].append(i)
    t_c2_build = time.time() - t0_c2

    t0_c3 = time.time()
    for i in range(n_target):
        c = t_countries[i]
        name = str(t_names[i]) if pd.notna(t_names[i]) else ''
        if len(name) >= 3:
            c_p3_idx[c][name[:3]].append(i)
    t_c3_build = time.time() - t0_c3

    t0_c4 = time.time()
    for i in range(n_target):
        c = t_countries[i]
        name = str(t_names[i]) if pd.notna(t_names[i]) else ''
        if len(name) >= 4:
            c_p4_idx[c][name[:4]].append(i)
    t_c4_build = time.time() - t0_c4

    t0_c5 = time.time()
    for i in range(n_target):
        c = t_countries[i]
        addr = str(t_addrs[i]) if pd.notna(t_addrs[i]) else ''
        if addr:
            addr_toks = set([t for t in addr.split() if len(t) >= 4 and t not in stop_addrs])
            for t in addr_toks:
                c_addr_idx[c][t].append(i)
    t_c5_build = time.time() - t0_c5

    t_idx_total = time.time() - t0_idx_total
    print(f"Total Index Construction: {t_idx_total:.2f}s")
    print(f"  - Name token index:    {t_c2_build:.2f}s ({100*t_c2_build/t_idx_total:.1f}%)")
    print(f"  - Prefix-3 index:      {t_c3_build:.2f}s ({100*t_c3_build/t_idx_total:.1f}%)")
    print(f"  - Prefix-4 index:      {t_c4_build:.2f}s ({100*t_c4_build/t_idx_total:.1f}%)")
    print(f"  - Address token index: {t_c5_build:.2f}s ({100*t_c5_build/t_idx_total:.1f}%)")

    # Build engine for candidate generation
    engine = ConfigABlockingEngine(targets)
    del c_name_idx, c_p3_idx, c_p4_idx, c_addr_idx
    gc.collect()

    # 3. Load S1 records
    print(f"\n--- Phase C: Loading {num_s1} France S1 Sample ---")
    s1_df = pd.read_csv('data/test/test_source1.tsv', sep='\t', dtype=str, nrows=5000, na_filter=False)
    s1_fr = s1_df[s1_df['country'].apply(normalize_country) == 'france'].head(num_s1).copy().reset_index(drop=True)
    s1_fr['norm_name'] = s1_fr['business_name'].apply(normalize_business_name)
    s1_fr['norm_address'] = s1_fr['business_address'].apply(normalize_business_address)
    s1_fr['norm_country'] = 'france'
    del s1_df
    gc.collect()

    # 4. Detailed Step-by-Step Profiling of Candidate Generation & Capping
    print(f"\n--- Phase D: Step-by-Step Candidate Generation Profiling ({num_s1} S1) ---")
    
    # Sub-component timers
    t_name_token_lookup = 0.0
    t_prefix4_lookup = 0.0
    t_prefix3_lookup = 0.0
    t_address_token_lookup = 0.0
    t_cand_evidence_iter = 0.0
    t_list_append = 0.0
    t_df_creation = 0.0
    t_cap_sort = 0.0
    t_cap_head = 0.0
    t_gc_cleanup = 0.0

    total_raw_pairs = 0
    total_post_cap = 0

    s1_eids = s1_fr['entity_id'].values
    s1_names = s1_fr['norm_name'].values
    s1_addrs = s1_fr['norm_address'].values
    s1_countries = s1_fr['norm_country'].values

    t_bench_start = time.time()

    out_s1 = []
    out_cand = []
    out_src = []
    out_b_ctry = []
    out_b_name = []
    out_b_p3 = []
    out_b_p4 = []
    out_b_addr = []
    out_num_keys = []
    out_score = []
    out_exact = []

    for s1_id, name, addr, country in zip(s1_eids, s1_names, s1_addrs, s1_countries):
        name = str(name) if pd.notna(name) else ''
        addr = str(addr) if pd.notna(addr) else ''
        country = str(country) if pd.notna(country) else ''

        cand_flags = {}
        cand_scores = {}

        # 1. Name token lookup
        t0 = time.time()
        for t in name.split():
            if len(t) > 1 and t not in engine.stop_tokens:
                for idx in engine.country_name_token_idx[country].get(t, ()):
                    cand_flags[idx] = cand_flags.get(idx, 0) | 1
                    cand_scores[idx] = cand_scores.get(idx, 0) + 70
        t_name_token_lookup += (time.time() - t0)

        # 2. Prefix 4 lookup
        t0 = time.time()
        if len(name) >= 4:
            p4 = name[:4]
            for idx in engine.country_prefix4_idx[country].get(p4, ()):
                cand_flags[idx] = cand_flags.get(idx, 0) | 2
                cand_scores[idx] = cand_scores.get(idx, 0) + 50
        t_prefix4_lookup += (time.time() - t0)

        # 3. Prefix 3 lookup
        t0 = time.time()
        if len(name) >= 3:
            p3 = name[:3]
            for idx in engine.country_prefix3_idx[country].get(p3, ()):
                cand_flags[idx] = cand_flags.get(idx, 0) | 4
                cand_scores[idx] = cand_scores.get(idx, 0) + 30
        t_prefix3_lookup += (time.time() - t0)

        # 4. Address token lookup
        t0 = time.time()
        if addr:
            addr_toks = set([t for t in addr.split() if len(t) >= 4 and t not in engine.stop_addrs])
            for t in addr_toks:
                for idx in engine.country_addr_token_idx[country].get(t, ()):
                    cand_flags[idx] = cand_flags.get(idx, 0) | 8
                    cand_scores[idx] = cand_scores.get(idx, 0) + 60
        t_address_token_lookup += (time.time() - t0)

        # 5 & 6. Evidence iteration & list appends
        t0 = time.time()
        total_raw_pairs += len(cand_flags)
        
        for idx, mask in cand_flags.items():
            b_name = 1 if (mask & 1) else 0
            b_p4 = 1 if (mask & 2) else 0
            b_p3 = 1 if (mask & 4) else 0
            b_addr = 1 if (mask & 8) else 0
            num_keys = b_name + b_p4 + b_p3 + b_addr

            if num_keys > 0:
                c_name = engine.target_names[idx]
                exact_match = 1 if (name and name == c_name) else 0

                out_s1.append(s1_id)
                out_cand.append(engine.target_eids[idx])
                out_src.append(engine.target_sources[idx])
                out_b_ctry.append(1)
                out_b_name.append(b_name)
                out_b_p3.append(b_p3)
                out_b_p4.append(b_p4)
                out_b_addr.append(b_addr)
                out_num_keys.append(num_keys)
                out_score.append(cand_scores[idx])
                out_exact.append(exact_match)
        t_list_append += (time.time() - t0)

        del cand_flags, cand_scores

    # 7. DataFrame creation
    t0 = time.time()
    N = len(out_s1)
    df_chunk = pd.DataFrame({
        'entity_id_s1': out_s1,
        'entity_id_cand': out_cand,
        'source': out_src,
        'blocked_country': np.array(out_b_ctry, dtype=np.int8),
        'blocked_name_token': np.array(out_b_name, dtype=np.int8),
        'blocked_prefix_3': np.array(out_b_p3, dtype=np.int8),
        'blocked_prefix_4': np.array(out_b_p4, dtype=np.int8),
        'blocked_address_token': np.array(out_b_addr, dtype=np.int8),
        'num_blocking_keys': np.array(out_num_keys, dtype=np.int8),
        'evidence_score': np.array(out_score, dtype=np.float32),
        'blocked_exact_name': np.array(out_exact, dtype=np.int8),
        'blocked_selective_token': np.array(out_b_name, dtype=np.int8),
        'blocked_rare_token': np.zeros(N, dtype=np.int8),
        'blocked_token_pair': np.zeros(N, dtype=np.int8),
        'blocked_char_ngram': np.zeros(N, dtype=np.int8),
        'shared_ngram_count': np.zeros(N, dtype=np.int8)
    })
    t_df_creation = time.time() - t0

    # 8 & 9. Candidate safety cap (sorting & grouping)
    t0 = time.time()
    df_sorted = df_chunk.sort_values(
        by=['entity_id_s1', 'evidence_score', 'entity_id_cand'],
        ascending=[True, False, True]
    )
    t_cap_sort = time.time() - t0

    t0 = time.time()
    filtered_df = df_sorted.groupby('entity_id_s1').head(400).reset_index(drop=True)
    t_cap_head = time.time() - t0
    total_post_cap = len(filtered_df)

    # 10. GC cleanup
    t0 = time.time()
    del out_s1, out_cand, out_src, out_b_ctry, out_b_name, out_b_p3, out_b_p4, out_b_addr, out_num_keys, out_score, out_exact
    del df_chunk, df_sorted, filtered_df
    gc.collect()
    t_gc_cleanup = time.time() - t0

    t_bench_total = time.time() - t_bench_start

    print(f"\nCandidate Generation Total ({num_s1} S1): {t_bench_total:.2f}s")
    print(f"Raw candidates: {total_raw_pairs:,} | Post-cap: {total_post_cap:,}")
    
    # 5. Component Timing Table
    components = [
        ("Prefix-3 lookup & bitmask union", t_prefix3_lookup),
        ("Prefix-4 lookup & bitmask union", t_prefix4_lookup),
        ("Name-token lookup & bitmask union", t_name_token_lookup),
        ("Address-token lookup & bitmask union", t_address_token_lookup),
        ("Candidate evidence extraction & list appends", t_list_append),
        ("DataFrame construction from columnar lists", t_df_creation),
        ("Candidate ranking (sort_values by 3 cols)", t_cap_sort),
        ("Candidate capping (groupby.head(400))", t_cap_head),
        ("Garbage collection & memory release", t_gc_cleanup)
    ]
    
    print("\n" + "=" * 70)
    print("DETAILED COMPONENT RUNTIME BREAKDOWN")
    print("=" * 70)
    print(f"{'Component':<45} | {'Time (s)':<10} | {'% of Gen Time':<12}")
    print("-" * 70)
    for name, dur in sorted(components, key=lambda x: x[1], reverse=True):
        pct = (dur / t_bench_total) * 100 if t_bench_total > 0 else 0
        print(f"{name:<45} | {dur:<10.3f} | {pct:<12.1f}%")
    print("-" * 70)
    print(f"{'Sum of Measured Components':<45} | {sum(c[1] for c in components):<10.3f} | {100*sum(c[1] for c in components)/t_bench_total:<12.1f}%")

    # 6. cProfile on a second run (5 S1 records) for detailed function call profiler
    print("\n" + "=" * 70)
    print("cProfile FUNCTION-LEVEL PROFILING (TOP 10 CUMULATIVE FUNCTIONS)")
    print("=" * 70)
    
    profiler = cProfile.Profile()
    s1_profile_sample = s1_fr.head(5).copy()
    
    profiler.enable()
    capped_p, _ = engine.generate_bounded_candidates(s1_profile_sample, max_candidates_per_s1=400, chunk_size=5, return_metrics=True)
    profiler.disable()
    
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(15)
    print(s.getvalue())

    s_tot = io.StringIO()
    ps_tot = pstats.Stats(profiler, stream=s_tot).sort_stats('tottime')
    ps_tot.print_stats(15)
    print("--- Top Functions by Internal (Self) Time ---")
    print(s_tot.getvalue())


if __name__ == '__main__':
    profile_blocking_pipeline(num_s1=25)
