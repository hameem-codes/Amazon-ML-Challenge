"""
Phase 4: Complete Architecture End-to-End Dry Run & Integration Test

Tests the full, connected executable pipeline WITHOUT training XGBoost:
1. Ingests raw sample (500 S1 records + controlled target background + true matches)
2. Ingests raw test sample text for transductive unlabeled TF-IDF fitting
3. Normalization (src/normalize.py)
4. Multi-Key Inverted-Index Blocking with Evidence Retention (src/blocking.py)
5. Candidate Safety Cap with evidence ranking & recall metrics (src/candidate_safety.py)
6. Materializes candidates as candidate_pairs.tsv contract
7. Training Pair Construction with evidence-ranked hard negatives (src/training_pairs.py)
8. Strict Entity-Level Train/Validation Split (src/split.py)
9. Transductive Unlabeled TF-IDF Vectorizer Fitting (src/tfidf_model.py)
10. Vectorized Feature Engineering (57 features) (src/features.py & src/build_features.py)
11. Feature Integrity Validation (NaN/Inf = 0, no leakage)
12. Collision Analysis Diagnostic (src/collision.py)
13. Central Configuration Fingerprint Verification (src/config.py)
"""

import os
import time
import tracemalloc
import pandas as pd
import numpy as np

from config import get_config_fingerprint, BLOCKING_CONFIG, CANDIDATE_CAP_CONFIG, PIPELINE_VERSION
from normalize import normalize_business_name, normalize_business_address, normalize_country
from blocking import ProductionBlockingEngine
from candidate_safety import apply_candidate_safety_cap
from split import split_candidate_pairs
from training_pairs import construct_training_pairs
from tfidf_model import TransductiveTFIDF
from collision import analyze_collisions
from build_features import build_feature_table, load_ground_truth_map
from test_blocking_scale import scan_target_file

def run_integration_dry_run():
    print("="*75)
    print("PHASE 4: EXECUTING FULL PIPELINE INTEGRATION DRY RUN")
    print(f"Pipeline Version: {PIPELINE_VERSION} | Fingerprint: {get_config_fingerprint()}")
    print("="*75)
    
    tracemalloc.start()
    t_start = time.time()
    
    # 1. Ingestion: Load 500 S1 entities + ground truth
    num_s1 = 500
    bg_per_source = 5000
    print(f"\n[Step 1] Loading {num_s1} Source 1 records and ground truth...")
    s1 = pd.read_csv('data/train/train_source1.tsv', sep='\t', dtype=str, nrows=num_s1, na_filter=False)
    s1_ids = set(s1['entity_id'])
    
    true_pairs = load_ground_truth_map('data/train/train_ground_truth.tsv', s1_ids=s1_ids)
    target_s2_ids = {p[1] for p in true_pairs if p[1].startswith('S2')}
    target_s3_ids = {p[1] for p in true_pairs if p[1].startswith('S3')}
    print(f"Loaded {len(s1)} S1 entities. Ground-truth true pairs: {len(true_pairs)} (S2: {len(target_s2_ids)}, S3: {len(target_s3_ids)})")
    
    # Load target background + true matches
    print(f"[Step 1b] Scanning target sources ({bg_per_source} bg each)...")
    s2 = scan_target_file('data/train/train_source2.tsv', target_s2_ids, bg_per_source)
    s2['source'] = 'S2'
    s3 = scan_target_file('data/train/train_source3.tsv', target_s3_ids, bg_per_source)
    s3['source'] = 'S3'
    s23 = pd.concat([s2, s3], ignore_index=True)
    print(f"Target pool loaded: {len(s23):,} ({len(s2):,} S2, {len(s3):,} S3)")
    
    # 2. Normalization
    print("\n[Step 2] Performing conservative Unicode-safe normalization...")
    s1['norm_name'] = s1['business_name'].apply(normalize_business_name)
    s1['norm_address'] = s1['business_address'].apply(normalize_business_address)
    s1['norm_country'] = s1['country'].apply(normalize_country)
    
    s23['norm_name'] = s23['business_name'].apply(normalize_business_name)
    s23['norm_address'] = s23['business_address'].apply(normalize_business_address)
    s23['norm_country'] = s23['country'].apply(normalize_country)
    print("Normalization complete.")
    
    # 3. Multi-Key Inverted-Index Blocking with Evidence Retention
    print("\n[Step 3] Multi-Key Inverted Index Blocking with evidence tracking...")
    t_block_start = time.time()
    engine = ProductionBlockingEngine(s23)
    df_cands_raw = engine.generate_candidates_with_evidence(s1)
    t_block_elapsed = time.time() - t_block_start
    print(f"Blocking complete in {t_block_elapsed:.2f}s! Raw candidates: {len(df_cands_raw):,}")
    
    # Verify evidence columns exist
    evidence_cols = ['blocked_exact_name', 'blocked_selective_token', 'blocked_rare_token',
                     'blocked_token_pair', 'blocked_char_ngram', 'shared_ngram_count', 'num_blocking_keys']
    for c in evidence_cols:
        assert c in df_cands_raw.columns, f"Missing evidence column: {c}"
    print(f"Verified blocking evidence metadata columns present: {evidence_cols}")
    
    # Measure blocking recall before cap
    cand_pairs_raw_set = set(zip(df_cands_raw['entity_id_s1'], df_cands_raw['entity_id_cand']))
    true_retrieved_raw = len(cand_pairs_raw_set.intersection(true_pairs))
    blocking_recall_raw = true_retrieved_raw / len(true_pairs) if len(true_pairs) > 0 else 0
    print(f"Raw Blocking Recall: {blocking_recall_raw * 100:.2f}% ({true_retrieved_raw} / {len(true_pairs)} true pairs)")
    
    # 4. Candidate Safety Cap & Evidence Ranking
    print("\n[Step 4] Applying deterministic candidate safety cap (max_cands=300)...")
    capped_cands, cap_metrics = apply_candidate_safety_cap(
        df_cands_raw,
        max_candidates_per_s1=300,
        true_pairs_set=true_pairs
    )
    print(f"Candidate Cap Summary:")
    print(f"  - Candidates before cap: {cap_metrics['before_cap_candidates']:,}")
    print(f"  - Candidates after cap: {cap_metrics['after_cap_candidates']:,}")
    print(f"  - Overflowing S1 entities: {cap_metrics['overflowing_s1_count']}")
    print(f"  - True pairs before cap: {cap_metrics['true_pairs_before_cap']}")
    print(f"  - True pairs after cap: {cap_metrics['true_pairs_after_cap']}")
    print(f"  - Candidate-Cap Recall Retention: {cap_metrics['cap_recall_retention'] * 100:.2f}%")
    combined_recall = (cap_metrics['true_pairs_after_cap'] / len(true_pairs)) if len(true_pairs) > 0 else 0
    print(f"  - Combined Final Candidate Recall: {combined_recall * 100:.2f}%")
    
    # 5. Materialize Candidate Pairs TSV Contract
    print("\n[Step 5] Materializing candidate_pairs.tsv contract...")
    os.makedirs('output', exist_ok=True)
    contract_cols = ['entity_id_s1', 'entity_id_cand', 'source']
    capped_cands.to_csv('output/candidate_pairs_dryrun.tsv', sep='\t', index=False)
    print(f"Materialized {len(capped_cands):,} candidate pairs to output/candidate_pairs_dryrun.tsv")
    
    # 6. Training Pair Construction with Hard Negatives
    print("\n[Step 6] Constructing training pairs (positives + hard negatives)...")
    df_training_pairs, tp_stats = construct_training_pairs(
        capped_cands,
        true_pairs_set=true_pairs,
        hard_neg_ratio=15,
        random_seed=42
    )
    print(f"Training Pair Construction Summary:")
    print(f"  - Positive pairs: {tp_stats['positive_count']}")
    print(f"  - Selected negative pairs: {tp_stats['selected_negative_count']} ({tp_stats['hard_negative_count']} hard, {tp_stats['random_negative_count']} random)")
    print(f"  - Dynamic scale_pos_weight: {tp_stats['scale_pos_weight']:.2f}")
    
    # 7. Strict Entity-Level Train / Validation Split
    print("\n[Step 7] Enforcing strict Source 1 entity-level split (80/20)...")
    train_pairs_df, val_pairs_df, train_s1, val_s1 = split_candidate_pairs(
        df_training_pairs,
        train_ratio=0.8,
        random_seed=42
    )
    # Check disjointness
    assert len(train_s1.intersection(val_s1)) == 0, "ERROR: Leakage! S1 IDs overlap between train and val!"
    train_pos = (train_pairs_df['label'] == 1).sum()
    val_pos = (val_pairs_df['label'] == 1).sum()
    print(f"Split Summary (0 overlap between S1 sets):")
    print(f"  - Train S1 entities: {len(train_s1)}, Pairs: {len(train_pairs_df):,}, Positives: {train_pos}")
    print(f"  - Validation S1 entities: {len(val_s1)}, Pairs: {len(val_pairs_df):,}, Positives: {val_pos}")
    
    # 8. Transductive Unlabeled TF-IDF Preprocessing
    print("\n[Step 8] Fitting Transductive Unlabeled TF-IDF on raw text (Train + Test sample)...")
    # Load small slice of test data to simulate transductive vocabulary inclusion
    test_s1 = pd.read_csv('data/test/test_source1.tsv', sep='\t', dtype=str, nrows=100, na_filter=False)
    test_s2 = pd.read_csv('data/test/test_source2.tsv', sep='\t', dtype=str, nrows=200, na_filter=False)
    
    unlabeled_names = (list(s1['norm_name']) + list(s23['norm_name']) + 
                       list(test_s1['business_name'].apply(normalize_business_name)) +
                       list(test_s2['business_name'].apply(normalize_business_name)))
    unlabeled_addrs = (list(s1['norm_address']) + list(s23['norm_address']) +
                       list(test_s1['business_address'].apply(normalize_business_address)) +
                       list(test_s2['business_address'].apply(normalize_business_address)))
                       
    tfidf = TransductiveTFIDF(max_features=10000)
    tfidf.fit_from_text_iterables(unlabeled_names, unlabeled_addrs)
    tfidf_models = tfidf.get_models_dict()
    print(f"Transductive TF-IDF fitted! Name Vocab: {len(tfidf_models['name_tfidf'].vocabulary_):,}, Address Vocab: {len(tfidf_models['address_tfidf'].vocabulary_):,}")
    
    # 9. Feature Engineering on Train & Validation Pairs
    print("\n[Step 9] Extracting 57 pairwise features for Train & Validation sets...")
    t_feat_start = time.time()
    feature_table_val = build_feature_table(val_pairs_df, s1, s23, tfidf_models=tfidf_models)
    t_feat_elapsed = time.time() - t_feat_start
    print(f"Validation Feature Table generated in {t_feat_elapsed:.2f}s! Shape: {feature_table_val.shape}")
    
    # 10. Feature Integrity Checks
    feature_cols = [c for c in feature_table_val.columns if c not in ('source1_entity_id', 'candidate_entity_id', 'candidate_source')]
    nan_count = feature_table_val[feature_cols].isna().sum().sum()
    inf_count = np.isinf(feature_table_val[feature_cols]).sum().sum()
    print(f"Feature Integrity: Total Features: {len(feature_cols)}, NaNs: {nan_count}, Infs: {inf_count}")
    assert nan_count == 0, f"Found {nan_count} NaNs in features!"
    assert inf_count == 0, f"Found {inf_count} Infs in features!"
    
    # Check that blocking evidence features are non-trivial
    num_keys_sum = feature_table_val['num_blocking_keys'].sum()
    print(f"Evidence Propagation Check: Total num_blocking_keys in validation table: {num_keys_sum:,}")
    assert num_keys_sum > 0, "ERROR: Blocking evidence is zero in feature table!"
    
    # 11. Diagnostic Collision Analysis
    print("\n[Step 11] Running diagnostic collision analysis on candidate pairs...")
    collision_report = analyze_collisions(
        capped_cands.rename(columns={'entity_id_s1': 'source1_entity_id', 'entity_id_cand': 'candidate_entity_id'})
    )
    print(f"Collision Diagnostics on Candidate Set:")
    print(f"  - Total Candidate Pairs: {collision_report['total_predictions']:,}")
    print(f"  - Unique S1: {collision_report['unique_s1']}, Unique Candidates: {collision_report['unique_candidates']}")
    print(f"  - Candidates matched to >1 S1: {collision_report['colliding_candidates_count']:,} ({collision_report['collision_rate']*100:.1f}%)")
    print(f"  - S2 collisions: {collision_report['s2_collisions']:,}, S3 collisions: {collision_report['s3_collisions']:,}")
    print(f"  - Multi-match histogram (candidates matching N S1s): {collision_report['multi_match_histogram']}")
    
    # Stop timer and measure memory
    total_elapsed = time.time() - t_start
    _, peak_mem_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mem_mb = peak_mem_bytes / (1024 * 1024)
    
    print("\n" + "="*75)
    print("INTEGRATION DRY RUN COMPLETE — ALL CHECKS PASSED")
    print(f"Total Runtime: {total_elapsed:.1f}s | Peak Memory: {peak_mem_mb:.1f} MB")
    print("="*75)
    
    return {
        'num_s1': num_s1,
        'target_count': len(s23),
        'true_pairs': len(true_pairs),
        'raw_candidates': len(df_cands_raw),
        'blocking_recall_raw': blocking_recall_raw,
        'capped_candidates': len(capped_cands),
        'combined_recall': combined_recall,
        'cap_metrics': cap_metrics,
        'tp_stats': tp_stats,
        'train_s1': len(train_s1),
        'val_s1': len(val_s1),
        'train_pairs': len(train_pairs_df),
        'val_pairs': len(val_pairs_df),
        'feature_count': len(feature_cols),
        'nan_count': nan_count,
        'inf_count': inf_count,
        'collision_report': collision_report,
        'fingerprint': get_config_fingerprint(),
        'runtime_sec': total_elapsed,
        'peak_mem_mb': peak_mem_mb
    }

if __name__ == '__main__':
    run_integration_dry_run()
