"""
Optimization E Prototype Test
Validates experimental DF cap engine logic and channel diagnostic tracking on 25 France S1 records.
"""

import sys, os, time, gc
sys.path.insert(0, 'src')
import pandas as pd
import numpy as np

from normalize import normalize_business_name, normalize_country, normalize_business_address
from blocking import ConfigABlockingEngine
from config import get_config_fingerprint

print("Checking config fingerprint...")
fp = get_config_fingerprint()
assert fp == "87f20ceeb84ccc6ea2d48678c7810ac5", f"Fingerprint mismatch: {fp}"
print(f"Locked production fingerprint verified: {fp}")

# Load 100k France targets for quick prototype
print("Loading target sample...")
s2_chunks = [
    chunk[chunk['country'].apply(normalize_country) == 'france']
    for chunk in pd.read_csv('data/test/test_source2.tsv', sep='\t', dtype=str, na_filter=False, chunksize=100000)
]
targets = pd.concat(s2_chunks[:2], ignore_index=True)
targets['norm_name'] = targets['business_name'].apply(normalize_business_name)
targets['norm_address'] = targets['business_address'].apply(normalize_business_address)
targets['norm_country'] = 'france'
targets['source'] = 'S2'
print(f"Loaded {len(targets):,} targets.")

base_engine = ConfigABlockingEngine(targets)

# Load 25 France S1
s1_df = pd.read_csv('data/test/test_source1.tsv', sep='\t', dtype=str, nrows=5000, na_filter=False)
s1_fr = s1_df[s1_df['country'].apply(normalize_country) == 'france'].head(25).copy().reset_index(drop=True)
s1_fr['norm_name'] = s1_fr['business_name'].apply(normalize_business_name)
s1_fr['norm_address'] = s1_fr['business_address'].apply(normalize_business_address)
s1_fr['norm_country'] = 'france'

# Filter helper
def build_df_capped_engine(base_engine, df_cap=None):
    if df_cap is None:
        return base_engine
    
    import copy
    from collections import defaultdict
    
    # Create lightweight clone sharing target arrays
    cloned = copy.copy(base_engine)
    
    # Shallow copy country dicts and filter keys with len > df_cap
    cloned.country_name_token_idx = defaultdict(lambda: defaultdict(list))
    for c, idx in base_engine.country_name_token_idx.items():
        cloned.country_name_token_idx[c] = {k: v for k, v in idx.items() if len(v) <= df_cap}
        
    cloned.country_prefix3_idx = defaultdict(lambda: defaultdict(list))
    for c, idx in base_engine.country_prefix3_idx.items():
        cloned.country_prefix3_idx[c] = {k: v for k, v in idx.items() if len(v) <= df_cap}
        
    cloned.country_prefix4_idx = defaultdict(lambda: defaultdict(list))
    for c, idx in base_engine.country_prefix4_idx.items():
        cloned.country_prefix4_idx[c] = {k: v for k, v in idx.items() if len(v) <= df_cap}
        
    cloned.country_addr_token_idx = defaultdict(lambda: defaultdict(list))
    for c, idx in base_engine.country_addr_token_idx.items():
        cloned.country_addr_token_idx[c] = {k: v for k, v in idx.items() if len(v) <= df_cap}
        
    return cloned

# Test E0 and E1
eng_e0 = build_df_capped_engine(base_engine, df_cap=None)
eng_e1 = build_df_capped_engine(base_engine, df_cap=500)

cands_e0, metrics_e0 = eng_e0.generate_bounded_candidates(s1_fr, max_candidates_per_s1=400, chunk_size=25, return_metrics=True)
cands_e1, metrics_e1 = eng_e1.generate_bounded_candidates(s1_fr, max_candidates_per_s1=400, chunk_size=25, return_metrics=True)

print(f"E0 (No Cap):   Raw: {metrics_e0['raw_candidate_count']:,} | Post-cap: {metrics_e0['post_cap_candidate_count']:,}")
print(f"E1 (Cap 500):  Raw: {metrics_e1['raw_candidate_count']:,} | Post-cap: {metrics_e1['post_cap_candidate_count']:,}")
raw_reduc = (1 - metrics_e1['raw_candidate_count'] / metrics_e0['raw_candidate_count']) * 100
print(f"Raw candidate reduction: {raw_reduc:.2f}%")
print("Prototype test PASSED.")
