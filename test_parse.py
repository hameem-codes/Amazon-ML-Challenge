import pandas as pd

def parse_matched_ids(val):
    if pd.isna(val) or str(val).strip() == '' or str(val).strip() == '[]':
        return []
    val_str = str(val).strip()
    if val_str.startswith('[') and val_str.endswith(']'):
        val_str = val_str[1:-1]
    items = []
    if ',' in val_str:
        raw_items = val_str.split(',')
    elif ' ' in val_str:
        raw_items = val_str.split(' ')
    else:
        raw_items = [val_str]
    for x in raw_items:
        clean = x.strip().strip("'").strip('"')
        if clean:
            items.append(clean)
    return items

for n in [1000, 2500, 5000]:
    s1 = pd.read_csv('data/train/train_source1.tsv', sep='\t', dtype=str, nrows=n, na_filter=False)
    gt = pd.read_csv('data/train/train_ground_truth.tsv', sep='\t', dtype=str, na_filter=False)
    gt_s1 = gt[gt['source1_entity_id'].isin(s1['entity_id'])]
    s2_m = set()
    s3_m = set()
    total_pairs = 0
    for _, row in gt_s1.iterrows():
        matches = parse_matched_ids(row['matched_entity_ids'])
        for m in matches:
            if m.startswith('S2'):
                s2_m.add(m)
            elif m.startswith('S3'):
                s3_m.add(m)
            total_pairs += 1
    print(f"N={n}: Total True Pairs={total_pairs}, Unique S2={len(s2_m)}, Unique S3={len(s3_m)}")
