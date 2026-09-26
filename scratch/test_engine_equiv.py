import sys, time
sys.path.insert(0, 'src')
import pandas as pd
import numpy as np

from normalize import normalize_business_name, normalize_country, normalize_business_address
from blocking import ConfigABlockingEngine as OldEngine
from candidate_safety import apply_candidate_safety_cap

# Create test target and s1
s1 = pd.read_csv('data/train/train_source1.tsv', sep='\t', dtype=str, nrows=200, na_filter=False)
s1['norm_name'] = s1['business_name'].apply(normalize_business_name)
s1['norm_address'] = s1['business_address'].apply(normalize_business_address)
s1['norm_country'] = s1['country'].apply(normalize_country)

s2 = pd.read_csv('data/train/train_source2.tsv', sep='\t', dtype=str, nrows=2000, na_filter=False)
s2['source'] = 'S2'
s2['norm_name'] = s2['business_name'].apply(normalize_business_name)
s2['norm_address'] = s2['business_address'].apply(normalize_business_address)
s2['norm_country'] = s2['country'].apply(normalize_country)

s3 = pd.read_csv('data/train/train_source3.tsv', sep='\t', dtype=str, nrows=2000, na_filter=False)
s3['source'] = 'S3'
s3['norm_name'] = s3['business_name'].apply(normalize_business_name)
s3['norm_address'] = s3['business_address'].apply(normalize_business_address)
s3['norm_country'] = s3['country'].apply(normalize_country)

targets = pd.concat([s2, s3], ignore_index=True)

old_engine = OldEngine(targets)
t0 = time.time()
old_cands = old_engine.generate_candidates_with_evidence(s1)
old_time = time.time() - t0
print(f"Old engine: {len(old_cands):,} candidates in {old_time:.3f}s")

# Implement new fast & memory-safe chunked candidate generation
def new_generate_chunk(engine, s1_chunk):
    s1_eids = s1_chunk['entity_id'].values
    s1_names = s1_chunk['norm_name'].values
    s1_addrs = s1_chunk['norm_address'].values if 'norm_address' in s1_chunk.columns else np.array([''] * len(s1_chunk))
    s1_countries = s1_chunk['norm_country'].values if 'norm_country' in s1_chunk.columns else np.array([''] * len(s1_chunk))

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

        # Channel 2: Name token (selective)
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

        for idx, mask in cand_flags.items():
            b_name = 1 if (mask & 1) else 0
            b_p4 = 1 if (mask & 2) else 0
            b_p3 = 1 if (mask & 4) else 0
            b_addr = 1 if (mask & 8) else 0
            num_keys = b_name + b_p4 + b_p3 + b_addr

            if num_keys > 0:
                c_name = engine.target_names[idx]
                exact = 1 if (name and name == c_name) else 0
                score = cand_scores[idx]

                out_s1.append(s1_id)
                out_cand.append(engine.target_eids[idx])
                out_src.append(engine.target_sources[idx])
                out_b_ctry.append(1)
                out_b_name.append(b_name)
                out_b_p3.append(b_p3)
                out_b_p4.append(b_p4)
                out_b_addr.append(b_addr)
                out_num_keys.append(num_keys)
                out_score.append(score)
                out_exact.append(exact)

    N = len(out_s1)
    if N == 0:
        return pd.DataFrame(columns=[
            'entity_id_s1', 'entity_id_cand', 'source', 'blocked_country',
            'blocked_name_token', 'blocked_prefix_3', 'blocked_prefix_4',
            'blocked_address_token', 'num_blocking_keys', 'evidence_score',
            'blocked_exact_name', 'blocked_selective_token', 'blocked_rare_token',
            'blocked_token_pair', 'blocked_char_ngram', 'shared_ngram_count'
        ])

    return pd.DataFrame({
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

def new_generate_all(engine, df_s1, chunk_size=25):
    chunks = []
    for i in range(0, len(df_s1), chunk_size):
        chunk_s1 = df_s1.iloc[i:i+chunk_size]
        chunk_cands = new_generate_chunk(engine, chunk_s1)
        chunks.append(chunk_cands)
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()

t0 = time.time()
new_cands = new_generate_all(old_engine, s1, chunk_size=25)
new_time = time.time() - t0
print(f"New chunked engine (chunk=25): {len(new_cands):,} candidates in {new_time:.3f}s")

# Test equality
assert len(old_cands) == len(new_cands), f"Count mismatch: {len(old_cands)} vs {len(new_cands)}"

cols_to_check = [
    'entity_id_s1', 'entity_id_cand', 'source', 'blocked_country',
    'blocked_name_token', 'blocked_prefix_3', 'blocked_prefix_4',
    'blocked_address_token', 'num_blocking_keys', 'evidence_score',
    'blocked_exact_name', 'blocked_selective_token', 'blocked_rare_token',
    'blocked_token_pair', 'blocked_char_ngram', 'shared_ngram_count'
]

old_sorted = old_cands.sort_values(by=['entity_id_s1', 'entity_id_cand']).reset_index(drop=True)
new_sorted = new_cands.sort_values(by=['entity_id_s1', 'entity_id_cand']).reset_index(drop=True)

for col in cols_to_check:
    if col in ['evidence_score']:
        diff = np.abs(old_sorted[col].values.astype(float) - new_sorted[col].values.astype(float)).max()
        assert diff == 0, f"Score diff in {col}: {diff}"
    else:
        mismatch = (old_sorted[col].values != new_sorted[col].values).sum()
        assert mismatch == 0, f"Mismatch in {col}: {mismatch} rows"

print("ALL 16 COLUMNS MATCH EXACTLY 100% BIT-FOR-BIT!")

# Test with chunk_size = 7
new_cands_7 = new_generate_all(old_engine, s1, chunk_size=7)
new_sorted_7 = new_cands_7.sort_values(by=['entity_id_s1', 'entity_id_cand']).reset_index(drop=True)
for col in cols_to_check:
    mismatch = (new_sorted[col].values != new_sorted_7[col].values).sum()
    assert mismatch == 0, f"Chunk size 7 mismatch in {col}: {mismatch} rows"

print("DETERMINISM ACROSS CHUNK SIZES (25 vs 7) VERIFIED 100%!")

# Test capping
old_capped, _ = apply_candidate_safety_cap(old_cands, max_candidates_per_s1=400)
new_capped, _ = apply_candidate_safety_cap(new_cands, max_candidates_per_s1=400)
old_cap_pairs = set(zip(old_capped['entity_id_s1'], old_capped['entity_id_cand']))
new_cap_pairs = set(zip(new_capped['entity_id_s1'], new_capped['entity_id_cand']))
assert old_cap_pairs == new_cap_pairs
print("POST-CAP CANDIDATES MATCH 100%!")

# Test capping per-chunk vs whole
chunks_capped = []
for i in range(0, len(s1), 25):
    s1_chunk = s1.iloc[i:i+25]
    c_raw = new_generate_chunk(old_engine, s1_chunk)
    c_cap, _ = apply_candidate_safety_cap(c_raw, max_candidates_per_s1=400)
    chunks_capped.append(c_cap)
chunk_capped_all = pd.concat(chunks_capped, ignore_index=True)
chunk_cap_pairs = set(zip(chunk_capped_all['entity_id_s1'], chunk_capped_all['entity_id_cand']))
assert chunk_cap_pairs == old_cap_pairs
print("PER-CHUNK BOUNDED CAPPING MATCHES WHOLE-CAP 100%!")
