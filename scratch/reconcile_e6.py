import sys, os, time, gc, json
from collections import defaultdict
sys.path.insert(0, 'src')
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
import psutil

from normalize import normalize_business_name, normalize_country, normalize_business_address
from blocking import ConfigABlockingEngine
from config import BLOCKING_CHUNK_SIZE, CANDIDATE_CAP_CONFIG, get_config_fingerprint
from scratch.run_experiment_opt_e6 import generate_bounded_candidates_e6

def main():
    print("=" * 80, flush=True)
    print("RECONCILIATION AUDIT: E6-0 VS E6-A CANDIDATE SET & RECALL", flush=True)
    print("=" * 80, flush=True)

    fp = get_config_fingerprint()
    print(f"Production Fingerprint: {fp}", flush=True)
    assert fp == "87f20ceeb84ccc6ea2d48678c7810ac5"

    # 1. Load France Targets
    print("\n[1/4] Loading 1,434,993 France targets (S2 + S3)...", flush=True)
    t0 = time.time()
    targets_cache = 'scratch/targets_france.pkl'
    if os.path.exists(targets_cache):
        print(f"Loading cached targets from {targets_cache}...", flush=True)
        targets = pd.read_pickle(targets_cache)
    else:
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

        targets['norm_name'] = targets['business_name'].apply(normalize_business_name)
        targets['norm_address'] = targets['business_address'].apply(normalize_business_address)
        targets['norm_country'] = 'france'
        print(f"Saving targets to cache {targets_cache}...", flush=True)
        pd.to_pickle(targets, targets_cache)
    print(f"Targets loaded: {len(targets):,} in {time.time()-t0:.2f}s", flush=True)

    # 2. Build Engine
    print("\n[2/4] Building ConfigABlockingEngine...", flush=True)
    t0 = time.time()
    engine = ConfigABlockingEngine(targets)
    print(f"Engine built in {time.time()-t0:.2f}s", flush=True)

    # 3. Load 500 France S1
    print("\n[3/4] Loading 500 France S1...", flush=True)
    s1_df = pd.read_csv('data/test/test_source1.tsv', sep='\t', dtype=str, nrows=10000, na_filter=False)
    s1_fr = s1_df[s1_df['country'].apply(normalize_country) == 'france'].head(500).copy().reset_index(drop=True)
    del s1_df
    gc.collect()
    s1_fr['norm_name'] = s1_fr['business_name'].apply(normalize_business_name)
    s1_fr['norm_address'] = s1_fr['business_address'].apply(normalize_business_address)
    s1_fr['norm_country'] = 'france'
    print(f"S1 France loaded: {len(s1_fr)}", flush=True)

    # Reference Ground Truth (Exact Name Matches)
    s1_name_map = {}
    for idx, r in s1_fr.iterrows():
        s1_name_map.setdefault(r['norm_name'], []).append(r['entity_id'])

    ground_truth_pairs = set()
    for idx in range(len(engine.target_names)):
        t_name = engine.target_names[idx]
        if t_name in s1_name_map:
            t_eid = engine.target_eids[idx]
            for s1_id in s1_name_map[t_name]:
                ground_truth_pairs.add((s1_id, t_eid))
    print(f"Reference Ground Truth (exact name matches): {len(ground_truth_pairs):,} pairs", flush=True)

    # 4. Run E6-0 (Control)
    print("\n[4/4] Executing E6-0 (Control, no DF cap)...", flush=True)
    t0 = time.time()
    df_e6_0, m_e6_0 = generate_bounded_candidates_e6(
        engine, s1_fr, max_candidates_per_s1=400, chunk_size=BLOCKING_CHUNK_SIZE, addr_df_cap=None
    )
    t_e6_0 = time.time() - t0
    pairs_e6_0 = set(zip(df_e6_0['entity_id_s1'], df_e6_0['entity_id_cand']))
    print(f"E6-0 finished in {t_e6_0:.2f}s | Raw: {m_e6_0['raw_candidate_count']:,} | Post-cap: {len(pairs_e6_0):,}", flush=True)

    # Run E6-A (Selective 50,000)
    print("\nExecuting E6-A (Selective DF cap 50,000)...", flush=True)
    high_tokens = {k for k, v in engine.country_addr_token_idx['france'].items() if len(v) > 50000}
    high_sets = {k: set(engine.country_addr_token_idx['france'][k]) for k in high_tokens}
    t0 = time.time()
    df_e6_a, m_e6_a = generate_bounded_candidates_e6(
        engine, s1_fr, max_candidates_per_s1=400, chunk_size=BLOCKING_CHUNK_SIZE, addr_df_cap=50000, high_df_postings_sets=high_sets
    )
    t_e6_a = time.time() - t0
    pairs_e6_a = set(zip(df_e6_a['entity_id_s1'], df_e6_a['entity_id_cand']))
    print(f"E6-A finished in {t_e6_a:.2f}s | Raw: {m_e6_a['raw_candidate_count']:,} | Post-cap: {len(pairs_e6_a):,}", flush=True)

    # Also run Production Blocking Engine on the exact same 500 S1 to see production baseline (E0/E5-0)
    print("\nExecuting Production Engine baseline (E0 / E5-0)...", flush=True)
    t0 = time.time()
    df_prod, m_prod = engine.generate_bounded_candidates(
        s1_fr, max_candidates_per_s1=400, chunk_size=BLOCKING_CHUNK_SIZE, return_metrics=True
    )
    t_prod = time.time() - t0
    pairs_prod = set(zip(df_prod['entity_id_s1'], df_prod['entity_id_cand']))
    print(f"Production Engine finished in {t_prod:.2f}s | Raw: {m_prod['raw_candidate_count']:,} | Post-cap: {len(pairs_prod):,}", flush=True)

    # RECONCILIATION CALCULATIONS
    print("\n" + "=" * 80, flush=True)
    print("DETAILED SET-THEORETIC COMPARISONS", flush=True)
    print("=" * 80, flush=True)

    # 1. Total Candidates Comparison
    total_e6_0 = len(pairs_e6_0)
    total_e6_a = len(pairs_e6_a)
    cand_intersection = pairs_e6_0 & pairs_e6_a
    cand_xor = pairs_e6_0 ^ pairs_e6_a
    cand_only_e6_0 = pairs_e6_0 - pairs_e6_a
    cand_only_e6_a = pairs_e6_a - pairs_e6_0

    print(f"Total post-cap candidates E6-0: {total_e6_0:,}", flush=True)
    print(f"Total post-cap candidates E6-A: {total_e6_a:,}", flush=True)
    print(f"Candidate Overlap (E6-0 & E6-A): {len(cand_intersection):,} ({len(cand_intersection)/total_e6_0*100:.2f}%)", flush=True)
    print(f"Candidate Symmetric Difference (E6-0 XOR E6-A): {len(cand_xor):,}", flush=True)
    print(f"Candidates unique to E6-0: {len(cand_only_e6_0):,}", flush=True)
    print(f"Candidates unique to E6-A: {len(cand_only_e6_a):,}", flush=True)

    # 2. True Pairs Recall Comparison
    e6_0_true_post = pairs_e6_0 & ground_truth_pairs
    e6_a_true_post = pairs_e6_a & ground_truth_pairs
    prod_true_post = pairs_prod & ground_truth_pairs

    lost_vs_control = e6_0_true_post - e6_a_true_post
    gained_vs_control = e6_a_true_post - e6_0_true_post

    print(f"\nGround Truth Reference Pairs: {len(ground_truth_pairs):,}", flush=True)
    print(f"Production (E0 / E5-0) True Pairs Captured: {len(prod_true_post):,} (Recall: {len(prod_true_post)/len(ground_truth_pairs):.4%})", flush=True)
    print(f"E6-0 True Pairs Captured: {len(e6_0_true_post):,} (Recall: {len(e6_0_true_post)/len(ground_truth_pairs):.4%})", flush=True)
    print(f"E6-A True Pairs Captured: {len(e6_a_true_post):,} (Recall: {len(e6_a_true_post)/len(ground_truth_pairs):.4%})", flush=True)
    print(f"True Pairs Lost vs E6-0 (E6-0 - E6-A): {len(lost_vs_control)}", flush=True)
    print(f"True Pairs Gained vs E6-0 (E6-A - E6-0): {len(gained_vs_control)}", flush=True)
    print(f"True Pairs Lost vs Production (Prod - E6-A): {len(prod_true_post - e6_a_true_post)}", flush=True)
    print(f"True Pairs Gained vs Production (E6-A - Prod): {len(e6_a_true_post - prod_true_post)}", flush=True)

    # Write summary json
    reconciliation_data = {
        'e6_0': {
            'raw_candidates': m_e6_0['raw_candidate_count'],
            'post_cap_candidates': total_e6_0,
            'true_pairs_captured': len(e6_0_true_post),
            'post_cap_recall': len(e6_0_true_post) / len(ground_truth_pairs),
            'runtime_s': t_e6_0
        },
        'e6_a': {
            'raw_candidates': m_e6_a['raw_candidate_count'],
            'post_cap_candidates': total_e6_a,
            'true_pairs_captured': len(e6_a_true_post),
            'post_cap_recall': len(e6_a_true_post) / len(ground_truth_pairs),
            'runtime_s': t_e6_a
        },
        'production_baseline': {
            'raw_candidates': m_prod['raw_candidate_count'],
            'post_cap_candidates': len(pairs_prod),
            'true_pairs_captured': len(prod_true_post),
            'post_cap_recall': len(prod_true_post) / len(ground_truth_pairs),
            'runtime_s': t_prod
        },
        'candidate_set_comparison': {
            'overlap_count': len(cand_intersection),
            'overlap_pct': len(cand_intersection) / total_e6_0 * 100,
            'xor_count': len(cand_xor),
            'unique_to_e6_0': len(cand_only_e6_0),
            'unique_to_e6_a': len(cand_only_e6_a),
            'true_pairs_lost_vs_e6_0': len(lost_vs_control),
            'true_pairs_gained_vs_e6_0': len(gained_vs_control)
        }
    }

    with open('scratch/reconciliation_data.json', 'w') as f:
        json.dump(reconciliation_data, f, indent=2)
    print("\nReconciliation data saved to scratch/reconciliation_data.json", flush=True)

    print("Saving candidate sets to compressed TSVs...", flush=True)
    df_e6_0.to_csv('scratch/candidates_e6_0.tsv.gz', sep='\t', compression='gzip', index=False)
    df_e6_a.to_csv('scratch/candidates_e6_a.tsv.gz', sep='\t', compression='gzip', index=False)
    print("Saved candidate sets to scratch/candidates_e6_0.tsv.gz and scratch/candidates_e6_a.tsv.gz", flush=True)

if __name__ == '__main__':
    main()
