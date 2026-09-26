import sys
sys.path.append('src')
import time
import pandas as pd
from collections import defaultdict
from normalize import normalize_business_name, normalize_business_address, normalize_country
from test_blocking_scale import parse_matched_ids
from config import GENERIC_ADDRESS_TOKENS

# Load 10k S1
print("Loading 10k S1...")
s1 = pd.read_csv('data/train/train_source1.tsv', sep='\t', nrows=10000, dtype=str, na_filter=False)
s1['norm_name'] = s1['business_name'].apply(normalize_business_name)
s1['norm_address'] = s1['business_address'].apply(normalize_business_address)
s1['norm_country'] = s1['country'].apply(normalize_country)

gt = pd.read_csv('data/train/train_ground_truth.tsv', sep='\t', dtype=str, na_filter=False)
gt_s1 = gt[gt['source1_entity_id'].isin(s1['entity_id'])]

target_s2_ids = set()
target_s3_ids = set()
true_pairs = set()

for _, row in gt_s1.iterrows():
    s1_id = row['source1_entity_id']
    matches = parse_matched_ids(row['matched_entity_ids'])
    for m in matches:
        true_pairs.add((s1_id, m))
        if m.startswith('S2'):
            target_s2_ids.add(m)
        elif m.startswith('S3'):
            target_s3_ids.add(m)

print(f"Total True links: {len(true_pairs)}")

# Fast scan of S2 and S3 once
def scan_fast(filepath, needed_ids, bg_count):
    rows = []
    rem = set(needed_ids)
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        f.readline()
        idx = 0
        for line in f:
            idx += 1
            tab1 = line.find('\t')
            eid = line[:tab1] if tab1 != -1 else line.strip()
            is_needed = eid in rem
            if idx <= bg_count or is_needed:
                parts = line.rstrip('\r\n').split('\t')
                while len(parts) < 4:
                    parts.append('')
                rows.append({
                    'entity_id': parts[0],
                    'business_name': parts[1],
                    'business_address': parts[2],
                    'country': parts[3],
                    'norm_name': normalize_business_name(parts[1]),
                    'norm_address': normalize_business_address(parts[2]),
                    'norm_country': normalize_country(parts[3])
                })
                if is_needed:
                    rem.remove(eid)
                    if not rem and idx >= bg_count:
                        break
    return pd.DataFrame(rows).drop_duplicates(subset=['entity_id'])

bg_size = 30000
print(f"Scanning S2 with bg={bg_size}...")
s2 = scan_fast('data/train/train_source2.tsv', target_s2_ids, bg_size)
s2['source'] = 'S2'
print(f"S2 loaded: {len(s2)}")

print(f"Scanning S3 with bg={bg_size}...")
s3 = scan_fast('data/train/train_source3.tsv', target_s3_ids, bg_size)
s3['source'] = 'S3'
print(f"S3 loaded: {len(s3)}")

target_df = pd.concat([s2, s3], ignore_index=True)
print(f"Total Target pool: {len(target_df)}")

# Now test Option 1: Country-partitioned Team Baseline on first 1,000 S1 records
# S1 -> Country -> (Name-token, Name-prefix 3, Name-prefix 4, Address-token) -> Union + Dedup
s1_test = s1.iloc[:1000]
s1_test_eids = set(s1_test['entity_id'])
test_true_pairs = {p for p in true_pairs if p[0] in s1_test_eids}
print(f"\nTesting on 1,000 S1 (true pairs: {len(test_true_pairs)})...")

# Build inverted indexes by country
country_target_indices = defaultdict(list)
for i, c in enumerate(target_df['norm_country']):
    country_target_indices[c].append(i)

# Build inverted indexes within country
name_token_index = defaultdict(lambda: defaultdict(list))
p3_index = defaultdict(lambda: defaultdict(list))
p4_index = defaultdict(lambda: defaultdict(list))
addr_token_index = defaultdict(lambda: defaultdict(list))

for i in range(len(target_df)):
    c = target_df['norm_country'].iloc[i]
    name = target_df['norm_name'].iloc[i]
    addr = target_df['norm_address'].iloc[i]
    
    # Name tokens
    for t in name.split():
        if len(t) > 1:
            name_token_index[c][t].append(i)
    # Prefix 3
    if len(name) >= 3:
        p3_index[c][name[:3]].append(i)
    # Prefix 4
    if len(name) >= 4:
        p4_index[c][name[:4]].append(i)
    # Address tokens
    for t in addr.split():
        if t not in GENERIC_ADDRESS_TOKENS and len(t) > 3:
            addr_token_index[c][t].append(i)

print("Indexes built! Generating candidates for 1k S1...")
cand_pairs_country_filtered = set()
for _, row in s1_test.iterrows():
    s1_id = row['entity_id']
    c = row['norm_country']
    name = row['norm_name']
    addr = row['norm_address']
    
    cands_idx = set()
    # Name tokens
    for t in name.split():
        if len(t) > 1 and t in name_token_index[c]:
            cands_idx.update(name_token_index[c][t])
    # Prefix 3
    p3 = name[:3] if len(name) >= 3 else ''
    if p3 and p3 in p3_index[c]:
        cands_idx.update(p3_index[c][p3])
    # Prefix 4
    p4 = name[:4] if len(name) >= 4 else ''
    if p4 and p4 in p4_index[c]:
        cands_idx.update(p4_index[c][p4])
    # Address tokens
    for t in addr.split():
        if t not in GENERIC_ADDRESS_TOKENS and len(t) > 3 and t in addr_token_index[c]:
            cands_idx.update(addr_token_index[c][t])
            
    for idx in cands_idx:
        cand_pairs_country_filtered.add((s1_id, target_df['entity_id'].iloc[idx]))

retrieved = cand_pairs_country_filtered.intersection(test_true_pairs)
print(f"Country-Filtered Team Baseline on 1k S1:")
print(f"  Candidate pairs: {len(cand_pairs_country_filtered):,}")
print(f"  Avg candidates/S1: {len(cand_pairs_country_filtered)/1000:.2f}")
print(f"  True pairs retrieved: {len(retrieved)} / {len(test_true_pairs)} ({len(retrieved)/len(test_true_pairs)*100:.2f}%)")
print(f"  Missed: {len(test_true_pairs) - len(retrieved)}")
