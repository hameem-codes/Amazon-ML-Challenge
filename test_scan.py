import pandas as pd
import time

t0 = time.time()
gt = pd.read_csv('data/train/train_ground_truth.tsv', sep='\t', dtype=str, nrows=5000, na_filter=False)
s1 = pd.read_csv('data/train/train_source1.tsv', sep='\t', dtype=str, nrows=5000, na_filter=False)
gt_s1 = gt[gt['source1_entity_id'].isin(s1['entity_id'])]

true_s2_ids = set()
for val in gt_s1['matched_entity_ids']:
    for m in str(val).replace('[','').replace(']','').replace("'",'').split(','):
        m = m.strip()
        if m.startswith('S2'):
            true_s2_ids.add(m)

print(f"True S2 count for 5000 S1: {len(true_s2_ids)}")

s2_bg = pd.read_csv('data/train/train_source2.tsv', sep='\t', dtype=str, nrows=50000, na_filter=False)
in_bg = len(set(s2_bg['entity_id']).intersection(true_s2_ids))
print(f"S2 bg (50k rows) contains {in_bg} of {len(true_s2_ids)} true matches")
missing = len(true_s2_ids) - in_bg
print(f"Missing from head 50k: {missing}")

# Check line numbers of missing matches
print("Scanning chunks...")
s2_chunks = pd.read_csv('data/train/train_source2.tsv', sep='\t', dtype=str, na_filter=False, chunksize=100000)
found_count = in_bg
chunk_idx = 0
for chunk in s2_chunks:
    chunk_idx += 1
    chunk_found = chunk[chunk['entity_id'].isin(true_s2_ids)]
    found_count += len(chunk_found)
    print(f"Chunk {chunk_idx} checked, found so far: {found_count} / {len(true_s2_ids)} (time: {time.time()-t0:.1f}s)")
    if found_count >= len(true_s2_ids) or chunk_idx >= 5:
        break
