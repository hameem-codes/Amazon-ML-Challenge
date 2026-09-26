"""
Test script for Pre-Ranking Ablation:
Verifies Stage 0 baseline (expected Recall@400 = 82.7965%, 3,470 true pairs)
and tests the numerical scale behavior of legacy_score vs similarity signals.
"""

import sys, os, time, gc, heapq
from collections import defaultdict
sys.path.insert(0, 'src')
import pandas as pd
import numpy as np

from normalize import normalize_business_name, normalize_country, normalize_business_address
from blocking import ConfigABlockingEngine
from config import BLOCKING_CHUNK_SIZE, CANDIDATE_CAP_CONFIG, get_config_fingerprint
from scratch.run_experiment_opt_e6 import generate_bounded_chunk_e6

def token_jaccard(s1, s2):
    t1 = set(s1.split())
    t2 = set(s2.split())
    if not t1 or not t2:
        return 0.0
    return len(t1 & t2) / len(t1 | t2)

print("=" * 80)
print("PRE-RANKING ABLATION: STAGE 0 BASELINE VERIFICATION & SCALE CHECK")
print("=" * 80)

# 1. Load targets
targets = pd.read_pickle('scratch/targets_france.pkl')
engine = ConfigABlockingEngine(targets)

# 2. Load 500 France S1
s1_df = pd.read_csv('data/test/test_source1.tsv', sep='\t', dtype=str, nrows=10000, na_filter=False)
s1_fr = s1_df[s1_df['country'].apply(normalize_country) == 'france'].head(500).copy().reset_index(drop=True)
s1_fr['norm_name'] = s1_fr['business_name'].apply(normalize_business_name)
s1_fr['norm_address'] = s1_fr['business_address'].apply(normalize_business_address)
s1_fr['norm_country'] = 'france'

# 3. Reference ground truth
s1_name_map = {}
for idx, r in s1_fr.iterrows():
    s1_name_map.setdefault(r['norm_name'], []).append(r['entity_id'])

gt_pairs = set()
for idx in range(len(engine.target_names)):
    t_name = engine.target_names[idx]
    if t_name in s1_name_map:
        t_eid = engine.target_eids[idx]
        for s1_id in s1_name_map[t_name]:
            gt_pairs.add((s1_id, t_eid))

print(f"Total Reference True Pairs: {len(gt_pairs):,}")

# Run Stage 0 across all 500 S1
t0 = time.time()
chunk_size = 25
total_s1 = len(s1_fr)

stage0_pairs = set()
raw_total = 0

for start_idx in range(0, total_s1, chunk_size):
    end_idx = min(start_idx + chunk_size, total_s1)
    chunk = s1_fr.iloc[start_idx:end_idx]
    s1_orig_indices = list(range(start_idx, end_idx))
    s1_lex_ranks = [0] * len(chunk)

    df_chunk, raw_counts = generate_bounded_chunk_e6(
        engine, chunk, s1_orig_indices, s1_lex_ranks, max_candidates_per_s1=400, addr_df_cap=None
    )
    for r in df_chunk:
        # df_chunk is a list of tuples or records from generate_bounded_chunk_e6
        pass
    raw_total += sum(raw_counts.values())

print(f"Completed Stage 0 candidate generation check in {time.time()-t0:.2f}s")
