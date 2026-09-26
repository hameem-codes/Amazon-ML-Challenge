import sys, os, time, gc
from collections import defaultdict
sys.path.insert(0, 'src')
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from normalize import normalize_business_name, normalize_country, normalize_business_address
from blocking import ConfigABlockingEngine
from config import get_config_fingerprint

def verify_baseline():
    fp = get_config_fingerprint()
    print(f"Production Fingerprint: {fp}")
    assert fp == "87f20ceeb84ccc6ea2d48678c7810ac5", f"Fingerprint mismatch: {fp}"

    t0 = time.time()
    targets = pd.read_pickle('scratch/targets_france.pkl')
    print(f"Loaded {len(targets):,} targets in {time.time()-t0:.2f}s")
    engine = ConfigABlockingEngine(targets)

    target_names = engine.target_names
    target_eids = engine.target_eids
    target_lex = engine.target_lex_rank

    s1_df = pd.read_csv('data/test/test_source1.tsv', sep='\t', dtype=str, nrows=10000, na_filter=False)
    s1_fr = s1_df[s1_df['country'].apply(normalize_country) == 'france'].head(500).copy().reset_index(drop=True)
    del s1_df
    gc.collect()
    s1_fr['norm_name'] = s1_fr['business_name'].apply(normalize_business_name)
    s1_fr['norm_address'] = s1_fr['business_address'].apply(normalize_business_address)
    s1_fr['norm_country'] = 'france'
    val_s1_ids = list(s1_fr['entity_id'].values)

    s1_name_map = defaultdict(list)
    for idx, r in s1_fr.iterrows():
        s1_name_map[r['norm_name']].append(r['entity_id'])

    exact_matches_set = set()
    for idx in range(len(target_names)):
        t_name = target_names[idx]
        if t_name in s1_name_map:
            t_eid = target_eids[idx]
            for s1_id in s1_name_map[t_name]:
                exact_matches_set.add((s1_id, t_eid))

    TOTAL_GT_PAIRS = len(exact_matches_set)
    print(f"Ground truth true pairs: {TOTAL_GT_PAIRS} (Expected: 4,191)")
    assert TOTAL_GT_PAIRS == 4191, f"Expected 4191 GT pairs, got {TOTAL_GT_PAIRS}"

    BIT_NAME = 1 << 1
    BIT_P3 = 1 << 2
    BIT_P4 = 1 << 3
    BIT_ADDR = 1 << 4

    country_name_tokens = engine.country_name_token_idx['france']
    country_p3 = engine.country_prefix3_idx['france']
    country_p4 = engine.country_prefix4_idx['france']
    country_addr_tokens = engine.country_addr_token_idx['france']

    stage0_pairs = set()

    for s1_idx in range(len(s1_fr)):
        s1_row = s1_fr.iloc[s1_idx]
        s1_id = s1_row['entity_id']
        name = s1_row['norm_name']
        addr = s1_row['norm_address']

        s1_tokens = [t for t in name.split() if len(t) > 1 and t not in engine.stop_tokens]
        s1_p3 = name[:3] if len(name) >= 3 else ''
        s1_p4 = name[:4] if len(name) >= 4 else ''
        s1_addr_tokens = set([t for t in addr.split() if len(t) >= 4 and t not in engine.stop_addrs]) if addr else set()

        cand_flags = {}
        cand_scores = {}

        for t in s1_tokens:
            if t in country_name_tokens:
                for idx in country_name_tokens[t]:
                    cand_flags[idx] = cand_flags.get(idx, 0) | BIT_NAME
                    cand_scores[idx] = cand_scores.get(idx, 0) + 70

        if s1_p4 and s1_p4 in country_p4:
            for idx in country_p4[s1_p4]:
                cand_flags[idx] = cand_flags.get(idx, 0) | BIT_P4
                cand_scores[idx] = cand_scores.get(idx, 0) + 50

        if s1_p3 and s1_p3 in country_p3:
            for idx in country_p3[s1_p3]:
                cand_flags[idx] = cand_flags.get(idx, 0) | BIT_P3
                cand_scores[idx] = cand_scores.get(idx, 0) + 30

        for atok in s1_addr_tokens:
            if atok in country_addr_tokens:
                for idx in country_addr_tokens[atok]:
                    f = cand_flags.get(idx, 0)
                    if not (f & BIT_ADDR):
                        cand_flags[idx] = f | BIT_ADDR
                        cand_scores[idx] = cand_scores.get(idx, 0) + 60

        n_cands = len(cand_flags)
        if n_cands == 0:
            continue

        cand_indices_all = np.array(list(cand_flags.keys()), dtype=np.int32)
        legacy_sc_all = np.array([float(cand_scores[idx]) for idx in cand_indices_all], dtype=np.float32)

        if n_cands <= 400:
            selected_eids = [target_eids[idx] for idx in cand_indices_all]
        else:
            # Deterministic tie-breaking
            b_name = np.array([1 if (cand_flags[idx] & BIT_NAME) else 0 for idx in cand_indices_all], dtype=np.int8)
            b_p3 = np.array([1 if (cand_flags[idx] & BIT_P3) else 0 for idx in cand_indices_all], dtype=np.int8)
            b_p4 = np.array([1 if (cand_flags[idx] & BIT_P4) else 0 for idx in cand_indices_all], dtype=np.int8)
            b_addr = np.array([1 if (cand_flags[idx] & BIT_ADDR) else 0 for idx in cand_indices_all], dtype=np.int8)
            b_count = b_name + b_p3 + b_p4 + b_addr

            target_lex_cand = [target_lex[idx] for idx in cand_indices_all]

            sort_keys = (np.array(target_lex_cand, dtype=np.int32), -b_count, -legacy_sc_all)
            order = np.lexsort(sort_keys)
            selected_eids = [target_eids[cand_indices_all[i]] for i in order[:400]]

        for ceid in selected_eids:
            stage0_pairs.add((s1_id, ceid))

    tp = len(stage0_pairs & exact_matches_set)
    rec400 = (tp / TOTAL_GT_PAIRS) * 100
    print(f"Stage 0 Retained True Pairs: {tp} (Expected: 3,470)")
    print(f"Stage 0 Recall@400: {rec400:.4f}% (Expected: 82.7965%)")
    assert tp == 3470, f"Expected 3470 retained true pairs, got {tp}"
    assert abs(rec400 - 82.7965) < 0.001, f"Expected 82.7965% recall, got {rec400:.4f}%"
    print("STAGE 0 BASELINE VERIFIED SUCCESSFULLY!")

if __name__ == '__main__':
    verify_baseline()
