import sys, os, time, gc, re
from collections import defaultdict, Counter
sys.path.insert(0, 'src')
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from normalize import normalize_business_name, normalize_country, normalize_business_address
from blocking import ConfigABlockingEngine
from config import GENERIC_BUSINESS_TOKENS, GENERIC_ADDRESS_TOKENS

print("Loading targets...")
t0 = time.time()
targets = pd.read_pickle('scratch/targets_france.pkl')
engine = ConfigABlockingEngine(targets)
target_names = engine.target_names
target_addrs = engine.target_addrs
target_eids = engine.target_eids
target_lex = engine.target_lex_rank
n_targets = len(targets)
print(f"Loaded {n_targets:,} targets in {time.time()-t0:.2f}s")

# Load 500 France benchmark
s1_df = pd.read_csv('data/test/test_source1.tsv', sep='\t', dtype=str, nrows=10000, na_filter=False)
s1_fr = s1_df[s1_df['country'].apply(normalize_country) == 'france'].head(500).copy().reset_index(drop=True)
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
assert TOTAL_GT_PAIRS == 4191, f"Expected 4191 GT pairs, got {TOTAL_GT_PAIRS}"
print(f"Ground truth true pairs: {TOTAL_GT_PAIRS}")

# Channel 1: Exact Name
t0 = time.time()
name_to_target_idxs = defaultdict(list)
for idx, name in enumerate(target_names):
    if name:
        name_to_target_idxs[name].append(idx)

ch1_pairs = set()
ch1_counts = []
for s1_idx, r in s1_fr.iterrows():
    s1_id = r['entity_id']
    name = r['norm_name']
    cands = name_to_target_idxs.get(name, [])
    ch1_counts.append(len(cands))
    for cidx in cands:
        ch1_pairs.add((s1_id, target_eids[cidx]))

tp1 = len(ch1_pairs & exact_matches_set)
print(f"\nChannel 1 (Exact Name):")
print(f"  Total candidates: {len(ch1_pairs):,}, Mean/S1: {np.mean(ch1_counts):.2f}, Median: {np.median(ch1_counts)}, Max: {np.max(ch1_counts)}")
print(f"  TP: {tp1} / {TOTAL_GT_PAIRS} ({tp1/TOTAL_GT_PAIRS*100:.4f}%), Time: {time.time()-t0:.2f}s")

# Channel 2: Rare Name Token
# Calculate token document frequencies across targets
t0 = time.time()
token_doc_freq = Counter()
token_to_targets = defaultdict(list)
for idx, name in enumerate(target_names):
    if name:
        toks = set([t for t in name.split() if len(t) > 1 and t not in GENERIC_BUSINESS_TOKENS])
        for t in toks:
            token_doc_freq[t] += 1
            token_to_targets[t].append(idx)

# Rare token filter: doc frequency <= 0.002 * n_targets (~2870 targets) or <= 0.005 (~7175 targets)
# Let's inspect distribution
print(f"\nTotal unique informative tokens: {len(token_doc_freq):,}")
freq_vals = list(token_doc_freq.values())
print(f"Token freq percentiles: 50%={np.percentile(freq_vals, 50)}, 90%={np.percentile(freq_vals, 90)}, 99%={np.percentile(freq_vals, 99)}, max={np.max(freq_vals)}")

# Evaluate rare token channel with cap = 0.002 * n_targets (2,870) and min count 1
cap_count_002 = int(0.002 * n_targets)
rare_tokens_002 = {t for t, cnt in token_doc_freq.items() if cnt <= cap_count_002}
print(f"Tokens with freq <= 0.002 ({cap_count_002}): {len(rare_tokens_002):,} / {len(token_doc_freq):,}")

ch2_pairs = set()
ch2_counts = []
for s1_idx, r in s1_fr.iterrows():
    s1_id = r['entity_id']
    name = r['norm_name']
    toks = [t for t in name.split() if len(t) > 1 and t in rare_tokens_002]
    cand_set = set()
    for t in toks:
        cand_set.update(token_to_targets[t])
    ch2_counts.append(len(cand_set))
    for cidx in cand_set:
        ch2_pairs.add((s1_id, target_eids[cidx]))

tp2 = len(ch2_pairs & exact_matches_set)
print(f"Channel 2 (Rare Name Token, cap <= 0.2%):")
print(f"  Total candidates: {len(ch2_pairs):,}, Mean/S1: {np.mean(ch2_counts):.2f}, Median: {np.median(ch2_counts)}, Max: {np.max(ch2_counts)}")
print(f"  TP: {tp2} / {TOTAL_GT_PAIRS} ({tp2/TOTAL_GT_PAIRS*100:.4f}%), Time: {time.time()-t0:.2f}s")

# Channel 3: Postal / House Number
# Index by structural postal / house number
t0 = time.time()
num_to_targets = defaultdict(list)
for idx, addr in enumerate(target_addrs):
    if addr:
        # 5-digit postal code or house number
        post = re.findall(r'\b\d{5}\b', addr)
        # house number: leading number or number before street type
        house = re.findall(r'^\d{1,4}\b|\b\d{1,4}(?=\s+(?:rue|avenue|av|boulevard|bd|chemin|impasse|place|route|cours|allee))\b', addr)
        nums = set(post + house)
        for n in nums:
            num_to_targets[n].append(idx)

ch3_pairs = set()
ch3_counts = []
for s1_idx, r in s1_fr.iterrows():
    s1_id = r['entity_id']
    addr = r['norm_address']
    post = re.findall(r'\b\d{5}\b', addr)
    house = re.findall(r'^\d{1,4}\b|\b\d{1,4}(?=\s+(?:rue|avenue|av|boulevard|bd|chemin|impasse|place|route|cours|allee))\b', addr)
    nums = set(post + house)
    cand_set = set()
    for n in nums:
        # Note: to prevent explosion on generic numbers, we can limit or take all
        cand_set.update(num_to_targets[n])
    ch3_counts.append(len(cand_set))
    for cidx in cand_set:
        ch3_pairs.add((s1_id, target_eids[cidx]))

tp3 = len(ch3_pairs & exact_matches_set)
print(f"\nChannel 3 (Postal / House Number):")
print(f"  Total candidates: {len(ch3_pairs):,}, Mean/S1: {np.mean(ch3_counts):.2f}, Median: {np.median(ch3_counts)}, Max: {np.max(ch3_counts)}")
print(f"  TP: {tp3} / {TOTAL_GT_PAIRS} ({tp3/TOTAL_GT_PAIRS*100:.4f}%), Time: {time.time()-t0:.2f}s")

# Channel 4: Address Token (existing Config A semantics)
t0 = time.time()
country_addr_tokens = engine.country_addr_token_idx['france']
ch4_pairs = set()
ch4_counts = []
for s1_idx, r in s1_fr.iterrows():
    s1_id = r['entity_id']
    addr = r['norm_address']
    s1_addr_tokens = set([t for t in addr.split() if len(t) >= 4 and t not in engine.stop_addrs]) if addr else set()
    cand_set = set()
    for atok in s1_addr_tokens:
        if atok in country_addr_tokens:
            cand_set.update(country_addr_tokens[atok])
    ch4_counts.append(len(cand_set))
    for cidx in cand_set:
        ch4_pairs.add((s1_id, target_eids[cidx]))

tp4 = len(ch4_pairs & exact_matches_set)
print(f"\nChannel 4 (Address Token):")
print(f"  Total candidates: {len(ch4_pairs):,}, Mean/S1: {np.mean(ch4_counts):.2f}, Median: {np.median(ch4_counts)}, Max: {np.max(ch4_counts)}")
print(f"  TP: {tp4} / {TOTAL_GT_PAIRS} ({tp4/TOTAL_GT_PAIRS*100:.4f}%), Time: {time.time()-t0:.2f}s")

# Channel 5: Character N-Gram
# Char 3-grams with doc frequency filtering (e.g. df <= 0.02 * n_targets = 28,699, min shared ngrams >= 3)
t0 = time.time()
ngram_to_targets = defaultdict(list)
for idx, name in enumerate(target_names):
    if name:
        clean = name.replace(' ', '')
        if len(clean) >= 3:
            ngs = set(clean[i:i+3] for i in range(len(clean)-2))
            for ng in ngs:
                ngram_to_targets[ng].append(idx)

# filter ngrams with high frequency
max_ng_cnt = int(0.01 * n_targets) # 1% = 14,350
filtered_ngram_index = {ng: idxs for ng, idxs in ngram_to_targets.items() if len(idxs) <= max_ng_cnt}
print(f"\nChar 3-grams total: {len(ngram_to_targets):,}, filtered (<=1%): {len(filtered_ngram_index):,}")

ch5_pairs = set()
ch5_counts = []
min_shared_ng = 3
for s1_idx, r in s1_fr.iterrows():
    s1_id = r['entity_id']
    name = r['norm_name']
    clean = name.replace(' ', '')
    cand_hits = Counter()
    if len(clean) >= 3:
        ngs = set(clean[i:i+3] for i in range(len(clean)-2))
        for ng in ngs:
            if ng in filtered_ngram_index:
                cand_hits.update(filtered_ngram_index[ng])
    cand_set = {cidx for cidx, cnt in cand_hits.items() if cnt >= min_shared_ng}
    ch5_counts.append(len(cand_set))
    for cidx in cand_set:
        ch5_pairs.add((s1_id, target_eids[cidx]))

tp5 = len(ch5_pairs & exact_matches_set)
print(f"Channel 5 (Char 3-gram >= 3 shared, cap <= 1%):")
print(f"  Total candidates: {len(ch5_pairs):,}, Mean/S1: {np.mean(ch5_counts):.2f}, Median: {np.median(ch5_counts)}, Max: {np.max(ch5_counts)}")
print(f"  TP: {tp5} / {TOTAL_GT_PAIRS} ({tp5/TOTAL_GT_PAIRS*100:.4f}%), Time: {time.time()-t0:.2f}s")
