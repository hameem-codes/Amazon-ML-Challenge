import sys
sys.path.append('src')
import pandas as pd
import numpy as np
from normalize import normalize_business_name, normalize_business_address, normalize_country
from test_blocking_scale import parse_matched_ids

print("Loading 10,000 S1 records...")
s1 = pd.read_csv('data/train/train_source1.tsv', sep='\t', nrows=10000, dtype=str, na_filter=False)
s1['norm_name'] = s1['business_name'].apply(normalize_business_name)
s1['norm_address'] = s1['business_address'].apply(normalize_business_address)
s1['norm_country'] = s1['country'].apply(normalize_country)
s1_dict = s1.set_index('entity_id').to_dict('index')

print("Loading ground truth...")
gt = pd.read_csv('data/train/train_ground_truth.tsv', sep='\t', dtype=str, na_filter=False)
gt_s1 = gt[gt['source1_entity_id'].isin(s1['entity_id'])]

true_pairs = []
s2_needed = set()
s3_needed = set()

for _, row in gt_s1.iterrows():
    s1_id = row['source1_entity_id']
    matches = parse_matched_ids(row['matched_entity_ids'])
    for m in matches:
        true_pairs.append((s1_id, m))
        if m.startswith('S2'):
            s2_needed.add(m)
        elif m.startswith('S3'):
            s3_needed.add(m)

print(f"Total True Pairs: {len(true_pairs)}")
print(f"S2 unique true targets: {len(s2_needed)}, S3 unique true targets: {len(s3_needed)}")

# Load needed S2 and S3 targets
def load_needed(filepath, needed_ids):
    rows = {}
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        header = f.readline()
        for line in f:
            tab1 = line.find('\t')
            eid = line[:tab1] if tab1 != -1 else line.strip()
            if eid in needed_ids:
                parts = line.rstrip('\r\n').split('\t')
                while len(parts) < 4:
                    parts.append('')
                rows[eid] = {
                    'entity_id': parts[0],
                    'business_name': parts[1],
                    'business_address': parts[2],
                    'country': parts[3],
                    'norm_name': normalize_business_name(parts[1]),
                    'norm_address': normalize_business_address(parts[2]),
                    'norm_country': normalize_country(parts[3])
                }
                if len(rows) == len(needed_ids):
                    break
    return rows

print("Scanning needed S2...")
s2_dict = load_needed('data/train/train_source2.tsv', s2_needed)
print(f"Found {len(s2_dict)} / {len(s2_needed)} S2 targets")

print("Scanning needed S3...")
s3_dict = load_needed('data/train/train_source3.tsv', s3_needed)
print(f"Found {len(s3_dict)} / {len(s3_needed)} S3 targets")

target_dict = {**s2_dict, **s3_dict}

# Now analyze true-pair coverage by channel
channel_hits = {
    'same_country': 0,
    'exact_name': 0,
    'exact_name_and_country': 0,
    'name_token': 0,
    'name_prefix_3': 0,
    'name_prefix_4': 0,
    'address_token': 0,
    'config_a_union': 0,
    'filtered_char_3gram': 0,
    'selective_token': 0,
    'token_pair': 0,
    'config_b_union': 0,
    'config_c_union': 0,
    'config_d_union': 0,
}

from config import GENERIC_BUSINESS_TOKENS, GENERIC_ADDRESS_TOKENS

missed_by_a = []

for s1_id, cand_id in true_pairs:
    s1_row = s1_dict[s1_id]
    cand_row = target_dict.get(cand_id)
    if not cand_row:
        continue
        
    s1_name = s1_row['norm_name']
    c_name = cand_row['norm_name']
    s1_addr = s1_row['norm_address']
    c_addr = cand_row['norm_address']
    s1_country = s1_row['norm_country']
    c_country = cand_row['norm_country']
    
    same_country = (s1_country == c_country and s1_country != '')
    exact_name = (s1_name == c_name and s1_name != '')
    
    s1_toks = set(s1_name.split())
    c_toks = set(c_name.split())
    name_token = len(s1_toks.intersection(c_toks)) > 0
    
    s1_p3 = s1_name[:3] if len(s1_name) >= 3 else ''
    c_p3 = c_name[:3] if len(c_name) >= 3 else ''
    p3_match = (s1_p3 == c_p3 and s1_p3 != '')
    
    s1_p4 = s1_name[:4] if len(s1_name) >= 4 else ''
    c_p4 = c_name[:4] if len(c_name) >= 4 else ''
    p4_match = (s1_p4 == c_p4 and s1_p4 != '')
    
    s1_addr_toks = set(s1_addr.split()) - GENERIC_ADDRESS_TOKENS
    c_addr_toks = set(c_addr.split()) - GENERIC_ADDRESS_TOKENS
    addr_match = len(s1_addr_toks.intersection(c_addr_toks)) > 0
    
    # Selective token
    s1_sel = s1_toks - GENERIC_BUSINESS_TOKENS
    c_sel = c_toks - GENERIC_BUSINESS_TOKENS
    sel_match = len(s1_sel.intersection(c_sel)) > 0
    
    # Token pair
    tok_pair_match = len(s1_sel.intersection(c_sel)) >= 2
    
    # Char 3-gram >= 3
    s1_clean = s1_name.replace(' ', '')
    c_clean = c_name.replace(' ', '')
    s1_ng = {s1_clean[i:i+3] for i in range(len(s1_clean)-2)} if len(s1_clean) >= 3 else set()
    c_ng = {c_clean[i:i+3] for i in range(len(c_clean)-2)} if len(c_clean) >= 3 else set()
    ng_match = len(s1_ng.intersection(c_ng)) >= 3
    
    # Config A: Country + Name-token + Name-prefix 3 + Name-prefix 4 + Address-token
    # Check both with country filter and without country filter
    hit_a = (exact_name and same_country) or name_token or p3_match or p4_match or addr_match
    # Config B: Filtered Char 3-gram + Selective Token + Token Pair + Exact Name
    hit_b = exact_name or sel_match or tok_pair_match or ng_match
    # Config C: Config B + Address Token
    hit_c = hit_b or addr_match
    # Config D: Union of A + B
    hit_d = hit_a or hit_b
    
    if same_country: channel_hits['same_country'] += 1
    if exact_name: channel_hits['exact_name'] += 1
    if exact_name and same_country: channel_hits['exact_name_and_country'] += 1
    if name_token: channel_hits['name_token'] += 1
    if p3_match: channel_hits['name_prefix_3'] += 1
    if p4_match: channel_hits['name_prefix_4'] += 1
    if addr_match: channel_hits['address_token'] += 1
    if hit_a: 
        channel_hits['config_a_union'] += 1
    else:
        missed_by_a.append((s1_id, cand_id, s1_name, c_name, s1_addr, c_addr))
    if ng_match: channel_hits['filtered_char_3gram'] += 1
    if sel_match: channel_hits['selective_token'] += 1
    if tok_pair_match: channel_hits['token_pair'] += 1
    if hit_b: channel_hits['config_b_union'] += 1
    if hit_c: channel_hits['config_c_union'] += 1
    if hit_d: channel_hits['config_d_union'] += 1

print("\n--- Channel Coverage on True Pairs (Total: {}) ---".format(len(true_pairs)))
for k, v in channel_hits.items():
    print(f"  {k:25s}: {v:6d} / {len(true_pairs)} ({v/len(true_pairs)*100:.2f}%)")

print(f"\nMissed by Config A union: {len(missed_by_a)}")
if missed_by_a:
    print("First 5 missed by Config A:")
    for m in missed_by_a[:5]:
        print(f"  S1: {m[0]} ({m[2]}) | Cand: {m[1]} ({m[3]}) | S1 Addr: ({m[4]}) | Cand Addr: ({m[5]})")
