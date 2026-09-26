import sys, time, gc
sys.path.insert(0, 'src')
import pandas as pd
import numpy as np
from collections import defaultdict

from normalize import normalize_business_name, normalize_country, normalize_business_address
from blocking import ConfigABlockingEngine

# Load France targets
s2_chunks = [chunk[chunk['country'].apply(normalize_country) == 'france'] for chunk in pd.read_csv('data/test/test_source2.tsv', sep='\t', dtype=str, na_filter=False, chunksize=500000)]
s3_chunks = [chunk[chunk['country'].apply(normalize_country) == 'france'] for chunk in pd.read_csv('data/test/test_source3.tsv', sep='\t', dtype=str, na_filter=False, chunksize=500000)]
targets = pd.concat(s2_chunks + s3_chunks, ignore_index=True)
del s2_chunks, s3_chunks
gc.collect()

targets['norm_name'] = targets['business_name'].apply(normalize_business_name)
targets['norm_address'] = targets['business_address'].apply(normalize_business_address)
targets['norm_country'] = 'france'
targets['source'] = ['S2' if i < 703378 else 'S3' for i in range(len(targets))]

engine = ConfigABlockingEngine(targets)

# Load 25 France S1
s1_df = pd.read_csv('data/test/test_source1.tsv', sep='\t', dtype=str, nrows=5000, na_filter=False)
s1_fr = s1_df[s1_df['country'].apply(normalize_country) == 'france'].head(25).copy().reset_index(drop=True)
s1_fr['norm_name'] = s1_fr['business_name'].apply(normalize_business_name)
s1_fr['norm_address'] = s1_fr['business_address'].apply(normalize_business_address)
s1_fr['norm_country'] = 'france'

# 1. Run baseline
t0 = time.time()
base_cands, base_metrics = engine.generate_bounded_candidates(s1_fr, max_candidates_per_s1=400, chunk_size=25, return_metrics=True)
t_base = time.time() - t0
print(f"BASELINE: {t_base:.2f}s | Raw: {base_metrics['raw_candidate_count']:,} | Capped: {len(base_cands):,}")

# 2. Run Optimization A
def generate_bounded_opt_a(engine, df_s1, max_candidates_per_s1=400, chunk_size=25):
    c_size = chunk_size
    post_cap_chunks = []
    
    total_raw_candidates = 0
    overflowing_s1_count = 0
    zero_candidate_s1_count = 0
    raw_counts_per_s1 = {}
    s1_ids = list(df_s1['entity_id'].values)

    eids_target = engine.target_eids
    sources_target = engine.target_sources
    names_target = engine.target_names
    
    for i in range(0, len(df_s1), c_size):
        chunk_s1 = df_s1.iloc[i:i + c_size]
        
        s1_eids = chunk_s1['entity_id'].values
        s1_names = chunk_s1['norm_name'].values if 'norm_name' in chunk_s1.columns else np.array([''] * len(chunk_s1))
        s1_addrs = chunk_s1['norm_address'].values if 'norm_address' in chunk_s1.columns else np.array([''] * len(chunk_s1))
        s1_countries = chunk_s1['norm_country'].values if 'norm_country' in chunk_s1.columns else np.array([''] * len(chunk_s1))

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

            # Channel 2: Name token
            for t in name.split():
                if len(t) > 1 and t not in engine.stop_tokens:
                    for idx in engine.country_name_token_idx[country].get(t, ()):
                        cand_flags[idx] = cand_flags.get(idx, 0) | 1
                        cand_scores[idx] = cand_scores.get(idx, 0) + 70

            # Channel 3: Prefix 4
            if len(name) >= 4:
                p4 = name[:4]
                for idx in engine.country_prefix4_idx[country].get(p4, ()):
                    cand_flags[idx] = cand_flags.get(idx, 0) | 2
                    cand_scores[idx] = cand_scores.get(idx, 0) + 50

            # Channel 4: Prefix 3
            if len(name) >= 3:
                p3 = name[:3]
                for idx in engine.country_prefix3_idx[country].get(p3, ()):
                    cand_flags[idx] = cand_flags.get(idx, 0) | 4
                    cand_scores[idx] = cand_scores.get(idx, 0) + 30

            # Channel 5: Address token
            if addr:
                addr_toks = set([t for t in addr.split() if len(t) >= 4 and t not in engine.stop_addrs])
                for t in addr_toks:
                    for idx in engine.country_addr_token_idx[country].get(t, ()):
                        cand_flags[idx] = cand_flags.get(idx, 0) | 8
                        cand_scores[idx] = cand_scores.get(idx, 0) + 60

            n_raw = len(cand_flags)
            raw_counts_per_s1[s1_id] = n_raw
            total_raw_candidates += n_raw
            if n_raw == 0:
                zero_candidate_s1_count += 1
            if n_raw > max_candidates_per_s1:
                overflowing_s1_count += 1

            if n_raw == 0:
                continue

            if n_raw <= max_candidates_per_s1:
                # All candidates fit; sort them by (-score, eid)
                top_idxs = list(cand_flags.keys())
                top_idxs.sort(key=lambda idx: (-cand_scores[idx], eids_target[idx]))
            else:
                # Fast score-bucketing top-K selection
                buckets = defaultdict(list)
                for idx, sc in cand_scores.items():
                    buckets[sc].append(idx)

                top_idxs = []
                for sc in sorted(buckets.keys(), reverse=True):
                    b_idxs = buckets[sc]
                    b_idxs.sort(key=lambda idx: eids_target[idx])
                    if len(top_idxs) + len(b_idxs) <= max_candidates_per_s1:
                        top_idxs.extend(b_idxs)
                    else:
                        needed = max_candidates_per_s1 - len(top_idxs)
                        top_idxs.extend(b_idxs[:needed])
                        break

            # Materialize ONLY the retained top candidates
            for idx in top_idxs:
                mask = cand_flags[idx]
                b_name = 1 if (mask & 1) else 0
                b_p4 = 1 if (mask & 2) else 0
                b_p3 = 1 if (mask & 4) else 0
                b_addr = 1 if (mask & 8) else 0
                num_keys = b_name + b_p4 + b_p3 + b_addr

                c_name = names_target[idx]
                exact_match = 1 if (name and name == c_name) else 0

                out_s1.append(s1_id)
                out_cand.append(eids_target[idx])
                out_src.append(sources_target[idx])
                out_b_ctry.append(1)
                out_b_name.append(b_name)
                out_b_p3.append(b_p3)
                out_b_p4.append(b_p4)
                out_b_addr.append(b_addr)
                out_num_keys.append(num_keys)
                out_score.append(cand_scores[idx])
                out_exact.append(exact_match)

            del cand_flags, cand_scores

        N_chunk = len(out_s1)
        if N_chunk > 0:
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
                'blocked_rare_token': np.zeros(N_chunk, dtype=np.int8),
                'blocked_token_pair': np.zeros(N_chunk, dtype=np.int8),
                'blocked_char_ngram': np.zeros(N_chunk, dtype=np.int8),
                'shared_ngram_count': np.zeros(N_chunk, dtype=np.int8)
            })
            post_cap_chunks.append(df_chunk)

        del out_s1, out_cand, out_src, out_b_ctry, out_b_name, out_b_p3, out_b_p4, out_b_addr, out_num_keys, out_score, out_exact
        gc.collect()

    if post_cap_chunks:
        capped_df = pd.concat(post_cap_chunks, ignore_index=True)
        capped_df = capped_df.sort_values(
            by=['entity_id_s1', 'evidence_score', 'entity_id_cand'],
            ascending=[True, False, True]
        ).reset_index(drop=True)
    else:
        capped_df = pd.DataFrame()

    post_cap_counts_per_s1 = capped_df.groupby('entity_id_s1').size().to_dict() if len(capped_df) > 0 else {}
    post_counts_list = [post_cap_counts_per_s1.get(sid, 0) for sid in s1_ids]
    raw_counts_list = [raw_counts_per_s1.get(sid, 0) for sid in s1_ids]

    metrics = {
        'num_s1_evaluated': len(df_s1),
        'raw_candidate_count': total_raw_candidates,
        'post_cap_candidate_count': len(capped_df),
        'candidates_per_s1': float(np.mean(post_counts_list)) if post_counts_list else 0.0,
        'median_candidates_per_s1': float(np.median(post_counts_list)) if post_counts_list else 0.0,
        'max_candidates_per_s1': int(np.max(post_counts_list)) if post_counts_list else 0,
        'raw_candidates_per_s1': float(np.mean(raw_counts_list)) if raw_counts_list else 0.0,
        'raw_median_candidates_per_s1': float(np.median(raw_counts_list)) if raw_counts_list else 0.0,
        'raw_max_candidates_per_s1': int(np.max(raw_counts_list)) if raw_counts_list else 0,
        'zero_candidate_s1_count': zero_candidate_s1_count,
        'overflowing_s1_count': overflowing_s1_count,
        'max_candidate_cap': max_candidates_per_s1,
        'chunk_size': c_size
    }
    return capped_df, metrics

t0 = time.time()
opt_cands, opt_metrics = generate_bounded_opt_a(engine, s1_fr, max_candidates_per_s1=400, chunk_size=25)
t_opt = time.time() - t0
print(f"OPTIMIZED A: {t_opt:.2f}s | Raw: {opt_metrics['raw_candidate_count']:,} | Capped: {len(opt_cands):,}")
print(f"SPEEDUP: {t_base / t_opt:.2f}x faster!")

# VERIFY 100% BIT-FOR-BIT EQUIVALENCE
assert len(base_cands) == len(opt_cands), f"Length mismatch: {len(base_cands)} vs {len(opt_cands)}"
assert base_metrics['raw_candidate_count'] == opt_metrics['raw_candidate_count'], "Raw count mismatch"
assert base_metrics['overflowing_s1_count'] == opt_metrics['overflowing_s1_count'], "Overflow mismatch"
assert base_metrics['zero_candidate_s1_count'] == opt_metrics['zero_candidate_s1_count'], "Zero cand mismatch"

for col in base_cands.columns:
    if col == 'evidence_score':
        diff = np.abs(base_cands[col].values - opt_cands[col].values).max()
        assert diff == 0, f"Diff in {col}: {diff}"
    else:
        mismatch = (base_cands[col].values != opt_cands[col].values).sum()
        assert mismatch == 0, f"Mismatch in {col}: {mismatch}"

print("ALL 16 COLUMNS MATCH EXACTLY 100% BIT-FOR-BIT IN EXACT ORDER!")
