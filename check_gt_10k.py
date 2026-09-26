import pandas as pd
import ast

def parse_ids(v):
    if not v or pd.isna(v) or v == '[]':
        return []
    try:
        if isinstance(v, str) and v.startswith('['):
            return ast.literal_eval(v)
    except:
        pass
    clean = str(v).replace('[', '').replace(']', '').replace("'", "").replace('"', '')
    return [x.strip() for x in clean.split(',') if x.strip()]

s1 = pd.read_csv('data/train/train_source1.tsv', sep='\t', nrows=10000, dtype=str, na_filter=False)
s1_set = set(s1['entity_id'])

# Stream ground truth to be fast and safe
total_links = 0
found_s1 = 0
for chunk in pd.read_csv('data/train/train_ground_truth.tsv', sep='\t', dtype=str, na_filter=False, chunksize=50000):
    subset = chunk[chunk['source1_entity_id'].isin(s1_set)]
    found_s1 += len(subset)
    for m in subset['matched_entity_ids']:
        ids = parse_ids(m)
        total_links += len(ids)
    if found_s1 >= 10000:
        break

print(f"S1 count: {len(s1)}, Matched GT S1 rows: {found_s1}, Total true links: {total_links}")
