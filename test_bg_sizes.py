import sys
sys.path.append('src')
import time
import pandas as pd
from collections import defaultdict
from normalize import normalize_business_name, normalize_business_address, normalize_country
from test_blocking_scale import parse_matched_ids, scan_target_file
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
print(f"Unique true targets: S2={len(target_s2_ids)}, S3={len(target_s3_ids)}")

for bg in [20000, 25000, 30000]:
    s2 = scan_target_file('data/train/train_source2.tsv', target_s2_ids, bg)
    s3 = scan_target_file('data/train/train_source3.tsv', target_s3_ids, bg)
    print(f"BG={bg}: total S2={len(s2)}, total S3={len(s3)}, total target={len(s2)+len(s3)}")
