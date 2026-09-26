"""
Test Batch Pipeline Verification
Verifies that batching S1 entities with standard ConfigABlockingEngine,
apply_candidate_safety_cap, build_feature_table, and XGBoost scoring
produces exact, clean outputs and maintains stable memory.
"""

import sys, os, time, gc, psutil
sys.path.insert(0, 'src')
import pandas as pd
import numpy as np
import xgboost as xgb

from normalize import normalize_business_name, normalize_country, normalize_business_address
from blocking import ConfigABlockingEngine
from candidate_safety import apply_candidate_safety_cap
from build_features import build_feature_table
from tfidf_model import TransductiveTFIDF
from features import FEATURE_NAMES

def test_batch_run():
    print("=== TESTING BATCH PIPELINE ON 1,000 TEST S1 RECORDS ===")
    
    # 1. Load model & verify
    model = xgb.XGBClassifier()
    model.load_model('models/xgboost_entity_resolution_phase5_configA.json')
    assert model.n_features_in_ == 57
    print("Loaded locked Phase 5 XGBoost model successfully.")
    
    # 2. Transductive TF-IDF
    s1_tr = pd.read_csv('data/train/train_source1.tsv', sep='\t', dtype=str, nrows=2500, na_filter=False)
    s1_tr['norm_name'] = s1_tr['business_name'].apply(normalize_business_name)
    s1_tr['norm_address'] = s1_tr['business_address'].apply(normalize_business_address)

    test_s1_text = pd.read_csv('data/test/test_source1.tsv', sep='\t', dtype=str, nrows=500, na_filter=False)
    test_s2_text = pd.read_csv('data/test/test_source2.tsv', sep='\t', dtype=str, nrows=1000, na_filter=False)
    test_s3_text = pd.read_csv('data/test/test_source3.tsv', sep='\t', dtype=str, nrows=1000, na_filter=False)

    unlabeled_names = (list(s1_tr['norm_name']) +
                       list(test_s1_text['business_name'].apply(normalize_business_name)) +
                       list(test_s2_text['business_name'].apply(normalize_business_name)) +
                       list(test_s3_text['business_name'].apply(normalize_business_name)))
    unlabeled_addrs = (list(s1_tr['norm_address']) +
                       list(test_s1_text['business_address'].apply(normalize_business_address)) +
                       list(test_s2_text['business_address'].apply(normalize_business_address)) +
                       list(test_s3_text['business_address'].apply(normalize_business_address)))

    tfidf = TransductiveTFIDF(max_features=12000)
    tfidf.fit_from_text_iterables(unlabeled_names, unlabeled_addrs)
    tfidf_models = tfidf.get_models_dict()
    print("Fitted transductive TF-IDF successfully.")

    # 3. Load 1,000 S1 records
    s1_df = pd.read_csv('data/test/test_source1.tsv', sep='\t', dtype=str, nrows=1000, na_filter=False)
    s1_df['norm_name'] = s1_df['business_name'].apply(normalize_business_name)
    s1_df['norm_address'] = s1_df['business_address'].apply(normalize_business_address)
    s1_df['norm_country'] = s1_df['country'].apply(normalize_country)
    
    # Load 50,000 targets each from S2 and S3
    s2 = pd.read_csv('data/test/test_source2.tsv', sep='\t', dtype=str, nrows=50000, na_filter=False)
    s2['source'] = 'S2'
    s3 = pd.read_csv('data/test/test_source3.tsv', sep='\t', dtype=str, nrows=50000, na_filter=False)
    s3['source'] = 'S3'
    targets = pd.concat([s2, s3], ignore_index=True)
    targets['norm_name'] = targets['business_name'].apply(normalize_business_name)
    targets['norm_address'] = targets['business_address'].apply(normalize_business_address)
    targets['norm_country'] = targets['country'].apply(normalize_country)
    
    print(f"Loaded {len(s1_df)} test S1, {len(targets)} test targets.")
    
    # Build engine
    t0 = time.time()
    engine = ConfigABlockingEngine(targets)
    print(f"Engine built in {time.time()-t0:.2f}s.")
    
    # Process in batches of 250 S1
    batch_size = 250
    all_preds_dict = {}
    total_raw_cands = 0
    total_capped_cands = 0
    all_cand_pairs = set()
    
    test_out_cand = 'scratch/test_batch_cands.tsv'
    test_out_match = 'scratch/test_batch_matches.tsv'
    
    with open(test_out_cand, 'w', encoding='utf-8') as f_cand:
        f_cand.write("entity_id_s1\tentity_id_cand\tsource\tnum_blocking_keys\tevidence_score\n")
        
    t_start = time.time()
    
    for start_idx in range(0, len(s1_df), batch_size):
        end_idx = min(start_idx + batch_size, len(s1_df))
        s1_batch = s1_df.iloc[start_idx:end_idx].copy()
        
        t_b0 = time.time()
        cands_raw = engine.generate_candidates_with_evidence(s1_batch)
        total_raw_cands += len(cands_raw)
        
        capped_cands, cap_metrics = apply_candidate_safety_cap(cands_raw, max_candidates_per_s1=400)
        total_capped_cands += len(capped_cands)
        
        # Stream candidate pairs
        if len(capped_cands) > 0:
            for _, r in capped_cands[['entity_id_s1', 'entity_id_cand', 'source', 'num_blocking_keys', 'evidence_score']].iterrows():
                all_cand_pairs.add((r['entity_id_s1'], r['entity_id_cand']))
                
            capped_cands[['entity_id_s1', 'entity_id_cand', 'source', 'num_blocking_keys', 'evidence_score']].to_csv(
                test_out_cand, sep='\t', index=False, header=False, mode='a'
            )
            
            # Features
            feat_df = build_feature_table(capped_cands, s1_batch, targets, tfidf_models=tfidf_models)
            X = feat_df[FEATURE_NAMES].values
            
            # Score
            probs = model.predict_proba(X)[:, 1]
            match_mask = (probs >= 0.910)
            
            matched_pairs = feat_df[match_mask][['source1_entity_id', 'candidate_entity_id']]
            for s1_id in s1_batch['entity_id']:
                all_preds_dict[s1_id] = []
            for _, r in matched_pairs.iterrows():
                all_preds_dict[r['source1_entity_id']].append(r['candidate_entity_id'])
        else:
            for s1_id in s1_batch['entity_id']:
                all_preds_dict[s1_id] = []
                
        mem_mb = psutil.virtual_memory().used / (1024**2)
        print(f"Batch [{start_idx}:{end_idx}] ({end_idx-start_idx} S1) complete in {time.time()-t_b0:.2f}s | Raw: {len(cands_raw):,}, Capped: {len(capped_cands):,} | Mem: {mem_mb:.1f} MB")
        
        del cands_raw, capped_cands
        gc.collect()

    print(f"\nAll batches complete in {time.time()-t_start:.2f}s!")
    print(f"Total raw candidates: {total_raw_cands:,}")
    print(f"Total post-cap candidates: {total_capped_cands:,}")
    
    # Write matching_results.tsv
    with open(test_out_match, 'w', encoding='utf-8') as f_match:
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in s1_df['entity_id']:
            matches = all_preds_dict.get(s1_id, [])
            f_match.write(f"{s1_id}\t{','.join(matches)}\n")
            
    print(f"Wrote matching results to {test_out_match}")
    
    # Verify Sanity Checks
    print("\n--- RUNNING SANITY CHECKS ---")
    df_match = pd.read_csv(test_out_match, sep='\t', dtype=str, na_filter=False)
    assert len(df_match) == len(s1_df), f"Expected {len(s1_df)} rows, got {len(df_match)}"
    assert df_match['source1_entity_id'].nunique() == len(s1_df), "Duplicate S1 IDs found!"
    assert list(df_match['source1_entity_id']) == list(s1_df['entity_id']), "S1 IDs or ordering mismatch!"
    
    pred_pair_count = 0
    zero_count = 0
    for _, r in df_match.iterrows():
        s1_id = r['source1_entity_id']
        m_str = r['matched_entity_ids']
        if not m_str:
            zero_count += 1
        else:
            m_list = m_str.split(',')
            assert len(m_list) == len(set(m_list)), f"Duplicate matches in row {s1_id}!"
            for tgt in m_list:
                pred_pair_count += 1
                assert (s1_id, tgt) in all_cand_pairs, f"Prediction ({s1_id}, {tgt}) not in candidate set!"
                assert not tgt.startswith('S1-'), f"Target {tgt} is an S1 ID!"
                
    print(f"Sanity Check PASS! Zero matches: {zero_count}/{len(s1_df)}, Predicted matches: {pred_pair_count}")

if __name__ == '__main__':
    test_batch_run()
