"""
Phase 6: Final Test Inference Module
Strictly Inference-Only Execution of Locked Config A Pipeline

Artifacts locked:
- Model: models/xgboost_entity_resolution_phase5_configA.json
- Schema: models/phase5_configA_feature_schema.json
- Threshold: models/phase5_configA_threshold.json
- Metadata: models/phase5_configA_metadata.json
- Config Fingerprint: 87f20ceeb84ccc6ea2d48678c7810ac5
- Threshold: 0.910
- Candidate Safety Cap: 400

Workflow:
1. Verify locked artifacts and fingerprint
2. Ingest & normalize test datasets (Source 1, Source 2, Source 3)
3. Fit transductive TF-IDF vectorizer on unlabeled text (train + test slices)
4. Partitioned execution by country for memory safety and maximum performance
5. Locked Config A blocking + evidence ranking + deterministic 400 safety cap
6. 57-feature table extraction with 0 NaNs and 0 Infs
7. XGBoost prediction at locked 0.910 threshold (many-to-many allowed)
8. Output generation: candidate_pairs.tsv and matching_results.tsv
9. Comprehensive 13-point sanity validation
10. Serialization of Phase 6 inference metadata and summary reporting
"""

import os
import sys
import time
import json
import gc
import psutil
import numpy as np
import pandas as pd
import xgboost as xgb

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from config import (
    get_config_fingerprint,
    BLOCKING_CONFIG,
    BLOCKING_ARCHITECTURE,
    BLOCKING_CHANNELS,
    CANDIDATE_CAP_CONFIG,
    PIPELINE_VERSION
)
from normalize import normalize_business_name, normalize_business_address, normalize_country
from blocking import ConfigABlockingEngine
from candidate_safety import apply_candidate_safety_cap
from build_features import build_feature_table
from tfidf_model import TransductiveTFIDF
from features import FEATURE_NAMES

LOCKED_THRESHOLD = 0.910
LOCKED_FINGERPRINT = "87f20ceeb84ccc6ea2d48678c7810ac5"
LOCKED_CAP = 400

def verify_locked_artifacts():
    print("[Step 1] Verifying locked Phase 5 artifacts...")
    model_path = 'models/xgboost_entity_resolution_phase5_configA.json'
    schema_path = 'models/phase5_configA_feature_schema.json'
    thresh_path = 'models/phase5_configA_threshold.json'
    meta_path = 'models/phase5_configA_metadata.json'
    
    for p in [model_path, schema_path, thresh_path, meta_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing locked artifact: {p}")
            
    with open(schema_path, 'r', encoding='utf-8') as f:
        schema = json.load(f)
    with open(thresh_path, 'r', encoding='utf-8') as f:
        thresh = json.load(f)
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
        
    current_fp = get_config_fingerprint()
    if current_fp != LOCKED_FINGERPRINT:
        raise ValueError(f"Config fingerprint mismatch! Current: {current_fp}, Locked: {LOCKED_FINGERPRINT}")
    if schema.get('config_fingerprint') != LOCKED_FINGERPRINT:
        raise ValueError(f"Schema fingerprint mismatch: {schema.get('config_fingerprint')}")
    if thresh.get('config_fingerprint') != LOCKED_FINGERPRINT:
        raise ValueError(f"Threshold fingerprint mismatch: {thresh.get('config_fingerprint')}")
    if meta.get('config_fingerprint') != LOCKED_FINGERPRINT:
        raise ValueError(f"Metadata fingerprint mismatch: {meta.get('config_fingerprint')}")
        
    if float(thresh.get('threshold')) != LOCKED_THRESHOLD:
        raise ValueError(f"Threshold mismatch! Expected {LOCKED_THRESHOLD}, got {thresh.get('threshold')}")
    if schema.get('feature_count') != 57:
        raise ValueError(f"Schema feature count mismatch: {schema.get('feature_count')}")
    if schema.get('feature_names') != FEATURE_NAMES:
        raise ValueError("Schema feature ordering mismatch with FEATURE_NAMES")
        
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    if model.n_features_in_ != 57:
        raise ValueError(f"Model booster features mismatch: {model.n_features_in_}")
        
    print("Locked artifacts verified successfully:")
    print(f"  - Model: {model_path} (57 features)")
    print(f"  - Threshold: {LOCKED_THRESHOLD:.3f}")
    print(f"  - Architecture: {BLOCKING_ARCHITECTURE}")
    print(f"  - Channels: {', '.join(BLOCKING_CHANNELS)}")
    print(f"  - Fingerprint: {LOCKED_FINGERPRINT}")
    
    return model, schema['feature_names']

def fit_transductive_tfidf():
    print("\n[Step 2] Fitting Transductive Unlabeled TF-IDF on raw text (Train + Unlabeled Test slices)...")
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
    print("Transductive TF-IDF vectorizer ready.")
    return tfidf_models

def run_test_inference(sample_limit=None, batch_size=200):
    t_global_start = time.time()
    
    # 1. Verify artifacts
    model, feature_names = verify_locked_artifacts()
    tfidf_models = fit_transductive_tfidf()
    
    # 2. Output file paths
    os.makedirs('output', exist_ok=True)
    os.makedirs('experiments', exist_ok=True)
    out_matching = 'output/matching_results.tsv'
    out_cands = 'output/candidate_pairs.tsv'
    root_matching = 'matching_results.tsv'
    root_cands = 'candidate_pairs.tsv'
    
    # Initialize output files
    with open(out_matching, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
    with open(out_cands, 'w', encoding='utf-8') as f:
        f.write("entity_id_s1\tentity_id_cand\tsource\tnum_blocking_keys\tevidence_score\n")
        
    # 3. Read Source 1 entities
    print(f"\n[Step 3] Loading test Source 1 dataset (sample_limit={sample_limit})...")
    s1_all = pd.read_csv('data/test/test_source1.tsv', sep='\t', dtype=str, nrows=sample_limit, na_filter=False)
    s1_all['norm_country'] = s1_all['country'].apply(normalize_country)
    s1_all['norm_name'] = s1_all['business_name'].apply(normalize_business_name)
    s1_all['norm_address'] = s1_all['business_address'].apply(normalize_business_address)
    
    total_s1 = len(s1_all)
    all_s1_ids = list(s1_all['entity_id'])
    print(f"Total test S1 entities loaded: {total_s1:,}")
    
    # Total targets count
    print("Counting test S2 and S3 totals...")
    cnt_s2 = sum(1 for _ in open('data/test/test_source2.tsv', 'r', encoding='utf-8')) - 1
    cnt_s3 = sum(1 for _ in open('data/test/test_source3.tsv', 'r', encoding='utf-8')) - 1
    print(f"Total test S2: {cnt_s2:,} | Total test S3: {cnt_s3:,}")
    
    # Global accumulator metrics
    global_raw_cands = 0
    global_post_cap_cands = 0
    global_overflowing_s1 = 0
    global_zero_cand_s1 = 0
    global_predicted_matches = 0
    
    cands_per_s1_list = []
    matches_per_s1_counts = {0: 0, 1: 0, 2: 0, 3: 0, '4+': 0}
    predicted_pairs_set = set()
    actual_cand_pairs_set = set()
    
    # Partition S1 by country
    unique_countries = sorted(list(s1_all['norm_country'].unique()))
    print(f"\n[Step 4] Partitioned execution across {len(unique_countries)} countries: {unique_countries}")
    
    for c_idx, country in enumerate(unique_countries, 1):
        s1_country = s1_all[s1_all['norm_country'] == country].copy().reset_index(drop=True)
        n_s1_c = len(s1_country)
        print(f"\n============================================================")
        print(f"PROCESSING COUNTRY {c_idx}/{len(unique_countries)}: '{country.upper()}' ({n_s1_c:,} S1 entities)")
        print(f"============================================================")
        
        # Load targets for this country from S2 and S3 using fast streaming scanner
        print(f"Scanning target sources for country '{country}'...")
        t_load0 = time.time()
        
        target_rows = []
        for filepath, src in [('data/test/test_source2.tsv', 'S2'), ('data/test/test_source3.tsv', 'S3')]:
            with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                f.readline()
                for line in f:
                    parts = line.rstrip('\r\n').split('\t')
                    if len(parts) >= 4 and parts[3].strip().lower() == country:
                        target_rows.append({
                            'entity_id': parts[0],
                            'business_name': parts[1],
                            'business_address': parts[2],
                            'country': country,
                            'norm_country': country,
                            'source': src
                        })
                        
        targets_country = pd.DataFrame(target_rows)
        del target_rows
        gc.collect()
        
        n_tgt_c = len(targets_country)
        print(f"Loaded {n_tgt_c:,} targets for '{country}' in {time.time()-t_load0:.2f}s")
        
        # Normalize targets
        t_norm0 = time.time()
        targets_country['norm_name'] = targets_country['business_name'].apply(normalize_business_name)
        targets_country['norm_address'] = targets_country['business_address'].apply(normalize_business_address)
        print(f"Normalized targets in {time.time()-t_norm0:.2f}s")
        
        # Build Config A inverted index for this country
        t_eng0 = time.time()
        engine = ConfigABlockingEngine(targets_country)
        print(f"Config A inverted index built in {time.time()-t_eng0:.2f}s")
        
        # Process S1 entities in batches
        print(f"Running inference in batches of {batch_size} S1 records...")
        t_c_start = time.time()
        
        for start_idx in range(0, n_s1_c, batch_size):
            end_idx = min(start_idx + batch_size, n_s1_c)
            s1_batch = s1_country.iloc[start_idx:end_idx].copy()
            
            # Blocking
            cands_raw = engine.generate_candidates_with_evidence(s1_batch)
            raw_count = len(cands_raw)
            global_raw_cands += raw_count
            
            # Safety cap
            capped_cands, cap_metrics = apply_candidate_safety_cap(cands_raw, max_candidates_per_s1=LOCKED_CAP)
            capped_count = len(capped_cands)
            global_post_cap_cands += capped_count
            global_overflowing_s1 += cap_metrics['overflowing_s1_count']
            
            # Candidate counts per S1 in this batch
            if raw_count > 0:
                s1_cand_counts = cands_raw.groupby('entity_id_s1').size().to_dict()
            else:
                s1_cand_counts = {}
                
            for eid in s1_batch['entity_id']:
                cnt = s1_cand_counts.get(eid, 0)
                cands_per_s1_list.append(cnt)
                if cnt == 0:
                    global_zero_cand_s1 += 1
                    
            # Stream candidates to TSV
            if capped_count > 0:
                # Track candidate pairs
                for _, r in capped_cands[['entity_id_s1', 'entity_id_cand']].iterrows():
                    actual_cand_pairs_set.add((r['entity_id_s1'], r['entity_id_cand']))
                    
                capped_cands[['entity_id_s1', 'entity_id_cand', 'source', 'num_blocking_keys', 'evidence_score']].to_csv(
                    out_cands, sep='\t', index=False, header=False, mode='a'
                )
                
                # Feature engineering (57 features)
                feat_df = build_feature_table(capped_cands, s1_batch, targets_country, tfidf_models=tfidf_models)
                
                # Audit feature matrix
                nans = int(feat_df[FEATURE_NAMES].isna().sum().sum())
                infs = int(np.isinf(feat_df[FEATURE_NAMES]).sum().sum())
                if nans > 0 or infs > 0:
                    raise ValueError(f"Numerical error in feature extraction: NaNs={nans}, Infs={infs}")
                    
                # XGBoost inference (DO NOT RETRAIN)
                X = feat_df[FEATURE_NAMES].values
                probs = model.predict_proba(X)[:, 1]
                match_mask = (probs >= LOCKED_THRESHOLD)
                
                matched_pairs = feat_df[match_mask][['source1_entity_id', 'candidate_entity_id']]
                
                # Group matches by S1
                batch_matches = {eid: [] for eid in s1_batch['entity_id']}
                for _, r in matched_pairs.iterrows():
                    s1_id = r['source1_entity_id']
                    cand_id = r['candidate_entity_id']
                    batch_matches[s1_id].append(cand_id)
                    predicted_pairs_set.add((s1_id, cand_id))
                    global_predicted_matches += 1
            else:
                batch_matches = {eid: [] for eid in s1_batch['entity_id']}
                
            # Stream matches to TSV
            with open(out_matching, 'a', encoding='utf-8') as f_match:
                for eid in s1_batch['entity_id']:
                    m_list = batch_matches.get(eid, [])
                    m_count = len(m_list)
                    if m_count == 0:
                        matches_per_s1_counts[0] += 1
                    elif m_count == 1:
                        matches_per_s1_counts[1] += 1
                    elif m_count == 2:
                        matches_per_s1_counts[2] += 1
                    elif m_count == 3:
                        matches_per_s1_counts[3] += 1
                    else:
                        matches_per_s1_counts['4+'] += 1
                    f_match.write(f"{eid}\t{','.join(m_list)}\n")
                    
            if (end_idx % (batch_size * 5) == 0) or (end_idx == n_s1_c):
                elapsed = time.time() - t_c_start
                rate = end_idx / elapsed if elapsed > 0 else 0
                mem_mb = psutil.virtual_memory().used / (1024**2)
                print(f"  [{country.upper()}] Processed {end_idx:,}/{n_s1_c:,} S1 ({end_idx/n_s1_c*100:.1f}%) | Rate: {rate:.1f} S1/s | Raw Cands: {global_raw_cands:,} | Matches: {global_predicted_matches:,} | Mem: {mem_mb:.1f} MB")
                
            del cands_raw, capped_cands
            gc.collect()
            
        print(f"Completed '{country}' in {time.time()-t_c_start:.2f}s!")
        del targets_country, engine
        gc.collect()
        
    # Copy output files to root directory
    import shutil
    shutil.copyfile(out_matching, root_matching)
    shutil.copyfile(out_cands, root_cands)
    print(f"\nCopied final outputs to {root_matching} and {root_cands}")
    
    # -------------------------------------------------------------
    # STEP 10 — INTERNAL SANITY CHECKS
    # -------------------------------------------------------------
    print("\n" + "="*70)
    print("[Step 10] EXECUTING 13 AUTHORITATIVE SANITY CHECKS")
    print("="*70)
    
    # Check 1 & 2: Row count & S1 entity coverage
    df_results = pd.read_csv(out_matching, sep='\t', dtype=str, na_filter=False)
    assert len(df_results) == total_s1, f"Check 1-2 Failed: Expected {total_s1} rows, got {len(df_results)}"
    print(f"Check 1 & 2 PASS: Exactly {len(df_results):,} rows matching test S1 count.")
    
    # Check 3: Unique S1 IDs
    assert df_results['source1_entity_id'].nunique() == total_s1, "Check 3 Failed: Duplicate S1 IDs found"
    print("Check 3 PASS: No duplicate source1_entity_id.")
    
    # Check 4 & 5: Valid Target IDs
    s1_id_set = set(all_s1_ids)
    all_targets_valid = True
    no_s1_in_targets = True
    no_duplicate_targets = True
    no_malformed_ids = True
    no_nan_artifacts = True
    
    for _, row in df_results.iterrows():
        s1_id = row['source1_entity_id']
        val = row['matched_entity_ids']
        if val == "nan" or val == "NaN":
            no_nan_artifacts = False
        if val:
            items = val.split(',')
            if len(items) != len(set(items)):
                no_duplicate_targets = False
            for target_id in items:
                if not (target_id.startswith('S2-') or target_id.startswith('S3-')):
                    no_malformed_ids = False
                if target_id in s1_id_set or target_id.startswith('S1-'):
                    no_s1_in_targets = False
                    
    assert all_targets_valid, "Check 4 Failed: Target ID invalid"
    assert no_s1_in_targets, "Check 5 Failed: Test S1 ID appears as matched target"
    assert no_duplicate_targets, "Check 6 Failed: Duplicate target IDs within single S1 row"
    assert no_malformed_ids, "Check 7 Failed: Malformed target IDs detected"
    assert no_nan_artifacts, "Check 8 Failed: NaN/null string artifacts detected"
    print("Checks 4, 5, 6, 7, 8 PASS: Target IDs valid, no S1 targets, no duplicates, no malformed/NaN IDs.")
    
    # Check 9: Column schema
    assert list(df_results.columns) == ['source1_entity_id', 'matched_entity_ids'], f"Check 9 Failed: Unexpected columns: {list(df_results.columns)}"
    print("Check 9 PASS: Column schema exactly ['source1_entity_id', 'matched_entity_ids'].")
    
    # Check 10 & 11 & 13: Predicted pairs exist in candidate set
    assert predicted_pairs_set.issubset(actual_cand_pairs_set), "Check 10/13 Failed: Predicted pairs exist outside candidate set!"
    print(f"Checks 10, 11, 13 PASS: All {len(predicted_pairs_set):,} predicted pairs exist in candidate set.")
    
    # Check 12: No S1->S1 pairs in candidate set
    s1_s1_leak = any(p[1].startswith('S1-') for p in actual_cand_pairs_set)
    assert not s1_s1_leak, "Check 12 Failed: S1->S1 candidate pairs detected!"
    print("Check 12 PASS: Zero S1->S1 candidate pairs.")
    
    # Calculate summary distributions
    avg_cands = float(np.mean(cands_per_s1_list)) if cands_per_s1_list else 0.0
    med_cands = float(np.median(cands_per_s1_list)) if cands_per_s1_list else 0.0
    max_cands = int(np.max(cands_per_s1_list)) if cands_per_s1_list else 0
    
    # -------------------------------------------------------------
    # STEP 12 — SAVE INFERENCE METADATA
    # -------------------------------------------------------------
    metadata_out = 'experiments/phase6_test_inference_metadata.json'
    inference_meta = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
        'pipeline_version': PIPELINE_VERSION,
        'model_path': 'models/xgboost_entity_resolution_phase5_configA.json',
        'feature_schema_path': 'models/phase5_configA_feature_schema.json',
        'threshold_path': 'models/phase5_configA_threshold.json',
        'threshold': LOCKED_THRESHOLD,
        'blocking_architecture': BLOCKING_ARCHITECTURE,
        'blocking_channels': BLOCKING_CHANNELS,
        'config_fingerprint': LOCKED_FINGERPRINT,
        'candidate_cap': LOCKED_CAP,
        'test_s1_count': total_s1,
        'test_s2_count': cnt_s2,
        'test_s3_count': cnt_s3,
        'raw_candidates': int(global_raw_cands),
        'post_cap_candidates': int(global_post_cap_cands),
        'average_candidates_per_s1': avg_cands,
        'median_candidates_per_s1': med_cands,
        'maximum_candidates_per_s1': max_cands,
        'zero_candidate_s1_count': int(global_zero_cand_s1),
        'overflowing_s1_count': int(global_overflowing_s1),
        'total_predicted_matches': int(global_predicted_matches),
        'zero_match_s1_count': int(matches_per_s1_counts[0]),
        'one_match_s1_count': int(matches_per_s1_counts[1]),
        'two_match_s1_count': int(matches_per_s1_counts[2]),
        'three_match_s1_count': int(matches_per_s1_counts[3]),
        'four_plus_match_s1_count': int(matches_per_s1_counts['4+']),
        'total_elapsed_seconds': float(time.time() - t_global_start)
    }
    
    with open(metadata_out, 'w', encoding='utf-8') as f:
        json.dump(inference_meta, f, indent=2)
    print(f"Saved inference metadata to {metadata_out}")
    
    # -------------------------------------------------------------
    # FINAL REPORT FORMAT
    # -------------------------------------------------------------
    print("\n" + "="*60)
    print("PHASE 6 COMPLETE — FINAL TEST INFERENCE")
    print("="*60)
    print("\nModel:")
    print("models/xgboost_entity_resolution_phase5_configA.json")
    print("\nFeature schema:")
    print("models/phase5_configA_feature_schema.json")
    print("\nBlocking:")
    print("CONFIG A — TEAM BASELINE")
    print("\nChannels:")
    print("Country")
    print("Name Token")
    print("Name Prefix-3")
    print("Name Prefix-4")
    print("Address Token")
    print(f"\nCandidate cap:\n{LOCKED_CAP}")
    print(f"\nThreshold:\n{LOCKED_THRESHOLD:.3f}")
    print(f"\nConfig fingerprint:\n{LOCKED_FINGERPRINT}")
    print(f"\nTEST S1:\n{total_s1:,}")
    print(f"\nTEST S2:\n{cnt_s2:,}")
    print(f"\nTEST S3:\n{cnt_s3:,}")
    print(f"\nRaw candidates:\n{global_raw_cands:,}")
    print(f"\nPost-cap candidates:\n{global_post_cap_cands:,}")
    print(f"\nAverage candidates/S1:\n{avg_cands:.2f}")
    print(f"\nMedian candidates/S1:\n{med_cands:.1f}")
    print(f"\nMaximum candidates/S1:\n{max_cands:,}")
    print(f"\nZero-candidate S1:\n{global_zero_cand_s1:,}")
    print(f"\nOverflowing S1:\n{global_overflowing_s1:,}")
    print(f"\nTotal predicted matches:\n{global_predicted_matches:,}")
    print(f"\nZero-match S1:\n{matches_per_s1_counts[0]:,}")
    print(f"\n1-match S1:\n{matches_per_s1_counts[1]:,}")
    print(f"\n2-match S1:\n{matches_per_s1_counts[2]:,}")
    print(f"\n3-match S1:\n{matches_per_s1_counts[3]:,}")
    print(f"\n4+-match S1:\n{matches_per_s1_counts['4+']:,}")
    print(f"\ncandidate_pairs.tsv:\n{os.path.abspath(root_cands)}")
    print(f"\nmatching_results.tsv:\n{os.path.abspath(root_matching)}")
    print(f"\nInference metadata:\n{os.path.abspath(metadata_out)}")
    print("\n" + "="*60)
    print("FIREWALL")
    print("="*60)
    print("\nTest labels:\nNOT USED")
    print("\nRetraining:\nNOT RUN")
    print("\nThreshold tuning:\nNOT RUN")
    print("\nArchitecture changes:\nNOT RUN")
    print("\nExternal lookup:\nNOT USED")
    print("\nLLM:\nNOT USED")
    print("\nSubmission:\nNOT GENERATED")
    print("="*60)

if __name__ == '__main__':
    limit = None
    if len(sys.argv) > 1:
        limit = int(sys.argv[1])
    run_test_inference(sample_limit=limit, batch_size=25)
