"""
Tournament Data Preparation

Loads exactly the 10,000 S1 entities, their complete ground truth,
and builds a standardized target pool (all true matches + 30k background per source).
Caches the normalized data for instantaneous reuse by all tournament configurations.
"""

import os
import sys
import time
import pandas as pd
import numpy as np

sys.path.append('src')
from normalize import normalize_business_name, normalize_business_address, normalize_country
from test_blocking_scale import parse_matched_ids

def main():
    print("="*60)
    print("PREPARING STANDARDIZED TOURNAMENT DATASET (10,000 S1)")
    print("="*60)
    t_start = time.time()
    
    os.makedirs('scratch', exist_ok=True)
    os.makedirs('experiments', exist_ok=True)
    
    # 1. Load S1 IDs
    s1_ids_path = 'experiments/blocking_10k_sample_ids.txt'
    if not os.path.exists(s1_ids_path):
        print("Generating blocking_10k_sample_ids.txt...")
        s1_raw = pd.read_csv('data/train/train_source1.tsv', sep='\t', nrows=10000, dtype=str)
        s1_raw['entity_id'].to_csv(s1_ids_path, index=False, header=False)
        
    with open(s1_ids_path, 'r', encoding='utf-8') as f:
        selected_s1_ids = [line.strip() for line in f if line.strip()]
        
    assert len(selected_s1_ids) == 10000, f"Expected 10,000 S1 IDs, got {len(selected_s1_ids)}"
    selected_s1_set = set(selected_s1_ids)
    print(f"Loaded {len(selected_s1_ids)} S1 IDs from {s1_ids_path}")
    
    # 2. Load and normalize S1
    print("Loading and normalizing Source 1 records...")
    s1_df = pd.read_csv('data/train/train_source1.tsv', sep='\t', nrows=10000, dtype=str, na_filter=False)
    s1_df = s1_df[s1_df['entity_id'].isin(selected_s1_set)].copy()
    s1_df['norm_name'] = s1_df['business_name'].apply(normalize_business_name)
    s1_df['norm_address'] = s1_df['business_address'].apply(normalize_business_address)
    s1_df['norm_country'] = s1_df['country'].apply(normalize_country)
    print(f"S1 normalized: {len(s1_df)} records")
    
    # 3. Ground Truth extraction
    print("Extracting ground-truth true pairs...")
    gt = pd.read_csv('data/train/train_ground_truth.tsv', sep='\t', dtype=str, na_filter=False)
    gt_s1 = gt[gt['source1_entity_id'].isin(selected_s1_set)]
    
    true_pairs = []
    target_s2_needed = set()
    target_s3_needed = set()
    
    for _, row in gt_s1.iterrows():
        s1_id = row['source1_entity_id']
        matches = parse_matched_ids(row['matched_entity_ids'])
        for m in matches:
            true_pairs.append({'source1_entity_id': s1_id, 'matched_entity_id': m})
            if m.startswith('S2'):
                target_s2_needed.add(m)
            elif m.startswith('S3'):
                target_s3_needed.add(m)
                
    df_true_pairs = pd.DataFrame(true_pairs)
    print(f"Total True Pairs: {len(df_true_pairs)} (S2: {len(target_s2_needed)}, S3: {len(target_s3_needed)})")
    
    # Save true pairs for tournament
    gt_cache_path = 'scratch/tournament_true_pairs.tsv'
    df_true_pairs.to_csv(gt_cache_path, sep='\t', index=False)
    
    # 4. Stream Source 2 (30k background + all true matches)
    bg_per_source = 30000
    print(f"\nScanning Source 2 (bg={bg_per_source} + true matches)...")
    rem_s2 = set(target_s2_needed)
    s2_rows = []
    chunk_size = 100000
    total_scanned_s2 = 0
    
    for chunk in pd.read_csv('data/train/train_source2.tsv', sep='\t', dtype=str, na_filter=False, chunksize=chunk_size):
        chunk_eids = set(chunk['entity_id'])
        
        # BG portion
        if total_scanned_s2 < bg_per_source:
            take_bg = min(bg_per_source - total_scanned_s2, len(chunk))
            s2_rows.append(chunk.iloc[:take_bg])
            
        # True matches in chunk
        needed_in_chunk = chunk[chunk['entity_id'].isin(rem_s2)]
        if len(needed_in_chunk) > 0:
            s2_rows.append(needed_in_chunk)
            rem_s2.difference_update(set(needed_in_chunk['entity_id']))
            
        total_scanned_s2 += len(chunk)
        if total_scanned_s2 >= bg_per_source and not rem_s2:
            break
            
    df_s2 = pd.concat(s2_rows, ignore_index=True).drop_duplicates(subset=['entity_id'])
    df_s2['source'] = 'S2'
    print(f"Normalizing S2 ({len(df_s2)} records)...")
    df_s2['norm_name'] = df_s2['business_name'].apply(normalize_business_name)
    df_s2['norm_address'] = df_s2['business_address'].apply(normalize_business_address)
    df_s2['norm_country'] = df_s2['country'].apply(normalize_country)
    print(f"S2 ready: {len(df_s2)} records (remaining unfound: {len(rem_s2)})")
    
    # 5. Stream Source 3 (30k background + all true matches)
    print(f"\nScanning Source 3 (bg={bg_per_source} + true matches)...")
    rem_s3 = set(target_s3_needed)
    s3_rows = []
    total_scanned_s3 = 0
    
    for chunk in pd.read_csv('data/train/train_source3.tsv', sep='\t', dtype=str, na_filter=False, chunksize=chunk_size):
        if total_scanned_s3 < bg_per_source:
            take_bg = min(bg_per_source - total_scanned_s3, len(chunk))
            s3_rows.append(chunk.iloc[:take_bg])
            
        needed_in_chunk = chunk[chunk['entity_id'].isin(rem_s3)]
        if len(needed_in_chunk) > 0:
            s3_rows.append(needed_in_chunk)
            rem_s3.difference_update(set(needed_in_chunk['entity_id']))
            
        total_scanned_s3 += len(chunk)
        if total_scanned_s3 >= bg_per_source and not rem_s3:
            break
            
    df_s3 = pd.concat(s3_rows, ignore_index=True).drop_duplicates(subset=['entity_id'])
    df_s3['source'] = 'S3'
    print(f"Normalizing S3 ({len(df_s3)} records)...")
    df_s3['norm_name'] = df_s3['business_name'].apply(normalize_business_name)
    df_s3['norm_address'] = df_s3['business_address'].apply(normalize_business_address)
    df_s3['norm_country'] = df_s3['country'].apply(normalize_country)
    print(f"S3 ready: {len(df_s3)} records (remaining unfound: {len(rem_s3)})")
    
    df_targets = pd.concat([df_s2, df_s3], ignore_index=True)
    print(f"\nTotal Target records: {len(df_targets):,} ({len(df_s2):,} S2, {len(df_s3):,} S3)")
    
    # Save cached data
    s1_cache = 'scratch/tournament_s1.tsv'
    targets_cache = 'scratch/tournament_targets.tsv'
    s1_df.to_csv(s1_cache, sep='\t', index=False)
    df_targets.to_csv(targets_cache, sep='\t', index=False)
    
    print(f"\nTournament data saved:")
    print(f"  S1: {s1_cache} ({len(s1_df)} records)")
    print(f"  Targets: {targets_cache} ({len(df_targets)} records)")
    print(f"  True Pairs: {gt_cache_path} ({len(df_true_pairs)} pairs)")
    print(f"Preparation completed in {time.time()-t_start:.1f}s!")

if __name__ == '__main__':
    main()
