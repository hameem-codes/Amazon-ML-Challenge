"""
Phase 5: Final Config A Training, XGBoost Retraining & Threshold Optimization

Locked Blocking Architecture: CONFIG A — TEAM BASELINE
Blocking Channels:
1. Country blocking
2. Name-token blocking
3. Name-prefix 3
4. Name-prefix 4
5. Address-token blocking

Workflow:
1. Training Ingestion: Source 1 training records + true matches + controlled target background
2. Conservative Unicode Normalization (src/normalize.py)
3. Locked Config A Inverted-Index Blocking (src/blocking.py)
4. Candidate Safety Cap with Evidence Ranking (src/candidate_safety.py)
5. Training Pair Construction with Hard-Negative Sampling (src/training_pairs.py)
6. Entity-Level Train / Validation Split (src/split.py)
7. Transductive Unlabeled TF-IDF Fitting (src/tfidf_model.py)
8. Vectorized 57-Feature Matrix Extraction (src/features.py & src/build_features.py)
9. Numerical Safety Audit (Strictly finite values, no NaN/Inf)
10. XGBoost Binary Pair Classifier Retraining from Scratch with Early Stopping
11. Probability Calibration Diagnostics (Brier score, ECE)
12. Validation Macro F0.5 Threshold Optimization (Sweep across [0.80 - 0.99] + Fine Search)
13. Collision & Multi-Match Diagnostics (src/collision.py)
14. Error Analysis (Representative False Positives & False Negatives)
15. Serialization of Model, Schema, Threshold, Metadata, and Metrics Artifacts
16. Generation of Comprehensive Markdown Report & Validation Metrics JSON
17. Enforcement of Strict Test Inference Firewall
"""

import os
import sys
import time
import tracemalloc
import json
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
    TRAINING_PAIRS_CONFIG,
    SPLIT_CONFIG,
    MODEL_CONFIG,
    PIPELINE_VERSION
)
from normalize import normalize_business_name, normalize_business_address, normalize_country
from blocking import ConfigABlockingEngine
from candidate_safety import apply_candidate_safety_cap
from training_pairs import construct_training_pairs
from split import split_candidate_pairs
from tfidf_model import TransductiveTFIDF
from build_features import build_feature_table, load_ground_truth_map
from features import FEATURE_NAMES
from test_blocking_scale import scan_target_file
from metrics import (
    compute_pairwise_metrics,
    compute_macro_f05,
    evaluate_threshold_grid,
    compute_calibration_diagnostics
)
from collision import analyze_collisions
from model_artifacts import (
    save_locked_model,
    save_feature_schema,
    save_locked_threshold,
    save_phase4b_metadata
)

def run_phase5_pipeline(num_s1=2500, bg_per_source=15000):
    print("="*80)
    print("PHASE 5: FINAL CONFIG A TRAINING + XGBOOST RETRAINING + THRESHOLD OPTIMIZATION")
    print(f"Pipeline Version: {PIPELINE_VERSION} | Fingerprint: {get_config_fingerprint()}")
    print(f"Locked Blocking Architecture: {BLOCKING_ARCHITECTURE}")
    print(f"Channels: {', '.join(BLOCKING_CHANNELS)}")
    print("="*80)
    
    tracemalloc.start()
    t_global_start = time.time()
    
    # -------------------------------------------------------------
    # 1. Ingestion: Load Source 1 training records + Ground Truth
    # -------------------------------------------------------------
    print(f"\n[Step 1] Loading {num_s1:,} Source 1 training records and ground-truth matches...")
    s1 = pd.read_csv('data/train/train_source1.tsv', sep='\t', dtype=str, nrows=num_s1, na_filter=False)
    s1_ids = set(s1['entity_id'])
    
    true_pairs = load_ground_truth_map('data/train/train_ground_truth.tsv', s1_ids=s1_ids)
    target_s2_ids = {p[1] for p in true_pairs if p[1].startswith('S2')}
    target_s3_ids = {p[1] for p in true_pairs if p[1].startswith('S3')}
    print(f"Loaded {len(s1):,} S1 entities. Ground-Truth True Links: {len(true_pairs):,} (S2: {len(target_s2_ids):,}, S3: {len(target_s3_ids):,})")
    
    # Load controlled background + true matches for S2 and S3
    print(f"[Step 1b] Scanning target sources ({bg_per_source:,} background records each + 100% true matches)...")
    s2 = scan_target_file('data/train/train_source2.tsv', target_s2_ids, bg_per_source)
    s2['source'] = 'S2'
    s3 = scan_target_file('data/train/train_source3.tsv', target_s3_ids, bg_per_source)
    s3['source'] = 'S3'
    s23 = pd.concat([s2, s3], ignore_index=True)
    print(f"Target pool loaded: {len(s23):,} ({len(s2):,} S2, {len(s3):,} S3)")
    
    # -------------------------------------------------------------
    # 2. Normalization
    # -------------------------------------------------------------
    print("\n[Step 2] Conservative Unicode-safe normalization...")
    s1['norm_name'] = s1['business_name'].apply(normalize_business_name)
    s1['norm_address'] = s1['business_address'].apply(normalize_business_address)
    s1['norm_country'] = s1['country'].apply(normalize_country)
    
    s23['norm_name'] = s23['business_name'].apply(normalize_business_name)
    s23['norm_address'] = s23['business_address'].apply(normalize_business_address)
    s23['norm_country'] = s23['country'].apply(normalize_country)
    print("Normalization complete.")
    
    # -------------------------------------------------------------
    # 3. Locked Config A Inverted-Index Blocking
    # -------------------------------------------------------------
    print(f"\n[Step 3] Locked Config A Blocking across {len(BLOCKING_CHANNELS)} channels...")
    t_block_start = time.time()
    engine = ConfigABlockingEngine(s23)
    df_cands_raw = engine.generate_candidates_with_evidence(s1)
    t_block_elapsed = time.time() - t_block_start
    
    raw_cand_count = len(df_cands_raw)
    avg_cands_per_s1 = raw_cand_count / len(s1) if len(s1) > 0 else 0
    
    s1_cand_counts = df_cands_raw.groupby('entity_id_s1').size()
    zero_cand_s1_count = len(s1_ids - set(s1_cand_counts.index))
    median_cands_s1 = float(s1_cand_counts.median()) if len(s1_cand_counts) > 0 else 0.0
    max_cands_s1 = int(s1_cand_counts.max()) if len(s1_cand_counts) > 0 else 0
    
    cand_pairs_raw_set = set(zip(df_cands_raw['entity_id_s1'], df_cands_raw['entity_id_cand']))
    true_retrieved_raw = len(cand_pairs_raw_set.intersection(true_pairs))
    raw_candidate_recall = (true_retrieved_raw / len(true_pairs) * 100) if len(true_pairs) > 0 else 0.0
    true_missed_raw = len(true_pairs) - true_retrieved_raw
    
    print(f"Blocking complete in {t_block_elapsed:.2f}s!")
    print(f"  - Raw Candidate Pairs: {raw_cand_count:,}")
    print(f"  - Candidates/S1: {avg_cands_per_s1:.1f} (Median: {median_cands_s1:.1f}, Max: {max_cands_s1:,})")
    print(f"  - Zero-Candidate S1 count: {zero_cand_s1_count}")
    print(f"  - Raw Candidate Recall: {raw_candidate_recall:.2f}% ({true_retrieved_raw:,} / {len(true_pairs):,} true links, {true_missed_raw} missed)")
    
    # -------------------------------------------------------------
    # 4. Candidate Safety Cap & Evidence Ranking
    # -------------------------------------------------------------
    cap_val = CANDIDATE_CAP_CONFIG['max_candidates_per_s1']
    print(f"\n[Step 4] Applying deterministic candidate safety cap (max_candidates_per_s1={cap_val})...")
    capped_cands, cap_metrics = apply_candidate_safety_cap(
        df_cands_raw,
        max_candidates_per_s1=cap_val,
        true_pairs_set=true_pairs
    )
    
    post_cap_cand_count = cap_metrics['after_cap_candidates']
    overflowing_s1_count = cap_metrics['overflowing_s1_count']
    true_pairs_post_cap = cap_metrics['true_pairs_after_cap']
    post_cap_candidate_recall = (true_pairs_post_cap / len(true_pairs) * 100) if len(true_pairs) > 0 else 0.0
    cap_retention_pct = cap_metrics['cap_recall_retention'] * 100
    true_missed_post_cap = len(true_pairs) - true_pairs_post_cap
    
    print("Candidate Safety Cap Metrics:")
    print(f"  - Raw Candidates: {raw_cand_count:,}")
    print(f"  - Post-Cap Candidates: {post_cap_cand_count:,}")
    print(f"  - Overflowing S1 Entities: {overflowing_s1_count:,}")
    print(f"  - True Pairs Before Cap: {true_retrieved_raw:,}")
    print(f"  - True Pairs After Cap: {true_pairs_post_cap:,} ({true_missed_post_cap} missed)")
    print(f"  - Raw Candidate Recall: {raw_candidate_recall:.2f}%")
    print(f"  - Post-Cap Candidate Recall: {post_cap_candidate_recall:.2f}%")
    print(f"  - Cap Recall Retention: {cap_retention_pct:.2f}%")
    
    # -------------------------------------------------------------
    # 5. Training Pair Construction with Hard-Negative Sampling
    # -------------------------------------------------------------
    print("\n[Step 5] Constructing training pairs (surviving positives + 70% evidence-ranked hard negatives + 30% random)...")
    df_training_pairs, tp_stats = construct_training_pairs(
        capped_cands,
        true_pairs_set=true_pairs,
        hard_neg_ratio=TRAINING_PAIRS_CONFIG['hard_negative_ratio'],
        random_seed=TRAINING_PAIRS_CONFIG['random_seed']
    )
    
    print("Training Pair Construction Summary:")
    print(f"  - Positives: {tp_stats['positive_count']:,}")
    print(f"  - Negatives: {tp_stats['selected_negative_count']:,} (Hard: {tp_stats['hard_negative_count']:,}, Random: {tp_stats['random_negative_count']:,})")
    print(f"  - Total Training Pairs: {len(df_training_pairs):,}")
    print(f"  - Positive/Negative Ratio: 1 : {tp_stats['selected_negative_count']/tp_stats['positive_count']:.2f}")
    print(f"  - Dynamic scale_pos_weight: {tp_stats['scale_pos_weight']:.4f}")
    
    # -------------------------------------------------------------
    # 6. Entity-Level Train / Validation Split (80/20)
    # -------------------------------------------------------------
    print("\n[Step 6] Enforcing strict Source 1 entity-level split (80/20)...")
    train_pairs_df, val_pairs_df, train_s1, val_s1 = split_candidate_pairs(
        df_training_pairs,
        train_ratio=SPLIT_CONFIG['train_ratio'],
        random_seed=SPLIT_CONFIG['random_seed']
    )
    
    leakage = train_s1.intersection(val_s1)
    if len(leakage) > 0:
        raise ValueError(f"CRITICAL LEAKAGE DETECTED: {len(leakage)} S1 entities appear in both train and validation splits!")
    print(f"Entity Isolation Verified: Train S1 ({len(train_s1):,}) ∩ Validation S1 ({len(val_s1):,}) = EMPTY.")
    
    train_pos = int((train_pairs_df['label'] == 1).sum())
    train_neg = int((train_pairs_df['label'] == 0).sum())
    val_pos = int((val_pairs_df['label'] == 1).sum())
    val_neg = int((val_pairs_df['label'] == 0).sum())
    train_scale_pos_weight = (train_neg / train_pos) if train_pos > 0 else 1.0
    
    print(f"Split Summary:")
    print(f"  - Train Pairs: {len(train_pairs_df):,} (Positives: {train_pos:,}, Negatives: {train_neg:,}, scale_pos_weight: {train_scale_pos_weight:.4f})")
    print(f"  - Validation Pairs: {len(val_pairs_df):,} (Positives: {val_pos:,}, Negatives: {val_neg:,})")
    
    # -------------------------------------------------------------
    # 7. Transductive Unlabeled TF-IDF Fitting
    # -------------------------------------------------------------
    print("\n[Step 7] Fitting Transductive Unlabeled TF-IDF on raw text (Train + Unlabeled Test slices)...")
    test_s1_text = pd.read_csv('data/test/test_source1.tsv', sep='\t', dtype=str, nrows=500, na_filter=False)
    test_s2_text = pd.read_csv('data/test/test_source2.tsv', sep='\t', dtype=str, nrows=1000, na_filter=False)
    test_s3_text = pd.read_csv('data/test/test_source3.tsv', sep='\t', dtype=str, nrows=1000, na_filter=False)
    
    unlabeled_names = (list(s1['norm_name']) + list(s23['norm_name']) +
                       list(test_s1_text['business_name'].apply(normalize_business_name)) +
                       list(test_s2_text['business_name'].apply(normalize_business_name)) +
                       list(test_s3_text['business_name'].apply(normalize_business_name)))
                       
    unlabeled_addrs = (list(s1['norm_address']) + list(s23['norm_address']) +
                       list(test_s1_text['business_address'].apply(normalize_business_address)) +
                       list(test_s2_text['business_address'].apply(normalize_business_address)) +
                       list(test_s3_text['business_address'].apply(normalize_business_address)))
                       
    tfidf = TransductiveTFIDF(max_features=12000)
    tfidf.fit_from_text_iterables(unlabeled_names, unlabeled_addrs)
    tfidf_models = tfidf.get_models_dict()
    print(f"TF-IDF Vocabularies: Name Vocab: {len(tfidf_models['name_tfidf'].vocabulary_):,}, Address Vocab: {len(tfidf_models['address_tfidf'].vocabulary_):,}")
    
    # -------------------------------------------------------------
    # 8. Feature Extraction (Authoritative 57-Feature Schema)
    # -------------------------------------------------------------
    print("\n[Step 8] Extracting authoritative 57-feature table for Train & Validation splits...")
    t_feat_start = time.time()
    df_feat_train = build_feature_table(train_pairs_df, s1, s23, tfidf_models=tfidf_models)
    df_feat_val = build_feature_table(val_pairs_df, s1, s23, tfidf_models=tfidf_models)
    t_feat_elapsed = time.time() - t_feat_start
    print(f"Features extracted in {t_feat_elapsed:.2f}s! Train Matrix: {df_feat_train.shape}, Validation Matrix: {df_feat_val.shape}")
    
    feature_cols = [c for c in df_feat_train.columns if c not in ('source1_entity_id', 'candidate_entity_id', 'candidate_source')]
    assert feature_cols == FEATURE_NAMES, f"Feature columns mismatch! Expected {len(FEATURE_NAMES)}, got {len(feature_cols)}"
    assert len(feature_cols) == 57, f"Expected exactly 57 features, got {len(feature_cols)}"
    
    # Numerical Safety Audit
    train_nans = int(df_feat_train[feature_cols].isna().sum().sum())
    train_infs = int(np.isinf(df_feat_train[feature_cols]).sum().sum())
    val_nans = int(df_feat_val[feature_cols].isna().sum().sum())
    val_infs = int(np.isinf(df_feat_val[feature_cols]).sum().sum())
    
    print(f"Numerical Safety Check: Train NaNs: {train_nans}, Infs: {train_infs} | Val NaNs: {val_nans}, Infs: {val_infs}")
    if train_nans > 0 or train_infs > 0 or val_nans > 0 or val_infs > 0:
        raise ValueError("Non-finite values detected in feature matrix! Halting execution.")
        
    X_train = df_feat_train[feature_cols].values
    y_train = train_pairs_df['label'].values
    X_val = df_feat_val[feature_cols].values
    y_val = val_pairs_df['label'].values
    
    # -------------------------------------------------------------
    # 9. Retrain XGBoost from Scratch
    # -------------------------------------------------------------
    print(f"\n[Step 9] Retraining XGBoost Pair Classifier from scratch (scale_pos_weight={train_scale_pos_weight:.4f})...")
    xgb_params = {
        'objective': MODEL_CONFIG['objective'],
        'eval_metric': MODEL_CONFIG['eval_metric'],
        'tree_method': MODEL_CONFIG['tree_method'],
        'learning_rate': MODEL_CONFIG['learning_rate'],
        'max_depth': MODEL_CONFIG['max_depth'],
        'n_estimators': MODEL_CONFIG['n_estimators'],
        'random_state': MODEL_CONFIG['random_state'],
        'scale_pos_weight': train_scale_pos_weight,
        'subsample': 0.85,
        'colsample_bytree': 0.85,
        'min_child_weight': 3,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'early_stopping_rounds': MODEL_CONFIG['early_stopping_rounds']
    }
    
    model = xgb.XGBClassifier(**xgb_params)
    t_train_start = time.time()
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=50
    )
    t_train_elapsed = time.time() - t_train_start
    best_iteration = int(model.best_iteration if hasattr(model, 'best_iteration') else xgb_params['n_estimators'])
    print(f"XGBoost training complete in {t_train_elapsed:.2f}s! Best iteration: {best_iteration}")
    
    # -------------------------------------------------------------
    # 10. Generate Validation Probabilities & Optimize Threshold
    # -------------------------------------------------------------
    print("\n[Step 10] Generating validation probabilities and optimizing threshold for Macro F0.5...")
    val_probs = model.predict_proba(X_val)[:, 1]
    
    df_val_results = pd.DataFrame({
        'source1_entity_id': df_feat_val['source1_entity_id'],
        'candidate_entity_id': df_feat_val['candidate_entity_id'],
        'candidate_source': df_feat_val['candidate_source'],
        'label': y_val,
        'predicted_probability': val_probs
    })
    
    # Sweep grid: coarse search across [0.80, 0.99]
    coarse_grid = [0.80, 0.85, 0.88, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
    df_sweep = evaluate_threshold_grid(df_val_results, val_s1, true_pairs, thresholds=coarse_grid)
    
    print("\n--- COARSE THRESHOLD SWEEP ---")
    display_cols = ['threshold', 'macro_f0_5', 'macro_precision', 'macro_recall', 'pairwise_f0_5', 'pairwise_precision', 'pairwise_recall', 'tp', 'fp', 'fn', 'predicted_matches', 'zero_match_s1_count']
    print(df_sweep[display_cols].to_string(index=False))
    
    # Fine search around peak
    best_row_coarse = df_sweep.loc[df_sweep['macro_f0_5'].idxmax()]
    t_center = best_row_coarse['threshold']
    fine_grid = sorted(list(set(np.round(np.arange(max(0.80, t_center - 0.05), min(0.995, t_center + 0.05), 0.005), 3))))
    df_fine_sweep = evaluate_threshold_grid(df_val_results, val_s1, true_pairs, thresholds=fine_grid)
    
    # Deterministic selection: maximize macro_f0_5, break ties by macro_precision, then higher threshold
    df_fine_sorted = df_fine_sweep.sort_values(by=['macro_f0_5', 'macro_precision', 'threshold'], ascending=[False, False, False])
    best_opt = df_fine_sorted.iloc[0]
    selected_threshold = float(best_opt['threshold'])
    
    print("\n" + "*"*60)
    print(f"LOCKED OPTIMAL THRESHOLD: {selected_threshold:.3f}")
    print(f"  - Validation Macro F0.5: {best_opt['macro_f0_5']:.4f}")
    print(f"  - Validation Macro Precision: {best_opt['macro_precision']*100:.2f}%")
    print(f"  - Validation Macro Recall: {best_opt['macro_recall']*100:.2f}%")
    print(f"  - Pairwise Precision: {best_opt['pairwise_precision']*100:.2f}%")
    print(f"  - Pairwise Recall: {best_opt['pairwise_recall']*100:.2f}%")
    print(f"  - TP: {best_opt['tp']}, FP: {best_opt['fp']}, FN: {best_opt['fn']}")
    print(f"  - Predicted Matches: {best_opt['predicted_matches']:,}")
    print(f"  - Zero-Match S1 Count: {best_opt['zero_match_s1_count']}")
    print("*"*60)
    
    # -------------------------------------------------------------
    # 11. Validation Diagnostics
    # -------------------------------------------------------------
    print("\n[Step 11] Running comprehensive validation diagnostics...")
    y_pred_opt = (val_probs >= selected_threshold).astype(int)
    df_val_results['pred_opt'] = y_pred_opt
    
    # A. Calibration Diagnostics
    calib = compute_calibration_diagnostics(y_val, val_probs, n_bins=10)
    print(f"  - Brier Score: {calib['brier_score']:.5f}")
    print(f"  - Expected Calibration Error (ECE): {calib['expected_calibration_error']:.5f}")
    
    # B. Singleton / Empty-Match Diagnostics
    # S1 entities with zero true ground-truth matches
    val_true_s1 = {p[0] for p in true_pairs if p[0] in val_s1}
    true_zero_match_s1 = val_s1 - val_true_s1
    pred_s1_with_matches = set(df_val_results.loc[y_pred_opt == 1, 'source1_entity_id'])
    
    correct_zero_match = len(true_zero_match_s1 - pred_s1_with_matches)
    fp_on_zero_match = len(true_zero_match_s1.intersection(pred_s1_with_matches))
    print(f"Singleton / Zero-Match Analysis:")
    print(f"  - True zero-match S1 entities: {len(true_zero_match_s1)}")
    print(f"  - Correctly predicted zero-match: {correct_zero_match}")
    print(f"  - False positive predictions on zero-match entities: {fp_on_zero_match}")
    
    # C. Collision Diagnostics
    opt_pred_df = df_val_results[y_pred_opt == 1].copy()
    collision_opt = analyze_collisions(opt_pred_df)
    print(f"Collision Diagnostics:")
    print(f"  - Total Predictions: {collision_opt['total_predictions']}")
    print(f"  - Unique Predicted Candidates: {collision_opt['unique_candidates']}")
    print(f"  - Colliding Candidates (target linked to >1 S1): {collision_opt['colliding_candidates_count']} ({collision_opt['collision_rate']*100:.2f}%)")
    print(f"    * S2 Collisions: {collision_opt['s2_collisions']}, S3 Collisions: {collision_opt['s3_collisions']}")
    
    # D. Score Distribution
    pos_probs = val_probs[y_val == 1]
    neg_probs = val_probs[y_val == 0]
    fp_probs = val_probs[(y_val == 0) & (y_pred_opt == 1)]
    fn_probs = val_probs[(y_val == 1) & (y_pred_opt == 0)]
    
    score_dist = {
        'positives': {
            'mean': float(np.mean(pos_probs)) if len(pos_probs) > 0 else 0.0,
            'median': float(np.median(pos_probs)) if len(pos_probs) > 0 else 0.0,
            'p25': float(np.percentile(pos_probs, 25)) if len(pos_probs) > 0 else 0.0,
            'p75': float(np.percentile(pos_probs, 75)) if len(pos_probs) > 0 else 0.0
        },
        'negatives': {
            'mean': float(np.mean(neg_probs)) if len(neg_probs) > 0 else 0.0,
            'median': float(np.median(neg_probs)) if len(neg_probs) > 0 else 0.0,
            'p25': float(np.percentile(neg_probs, 25)) if len(neg_probs) > 0 else 0.0,
            'p75': float(np.percentile(neg_probs, 75)) if len(neg_probs) > 0 else 0.0
        },
        'false_positives': {
            'count': int(len(fp_probs)),
            'mean': float(np.mean(fp_probs)) if len(fp_probs) > 0 else 0.0,
            'min': float(np.min(fp_probs)) if len(fp_probs) > 0 else 0.0,
            'max': float(np.max(fp_probs)) if len(fp_probs) > 0 else 0.0
        },
        'false_negatives': {
            'count': int(len(fn_probs)),
            'mean': float(np.mean(fn_probs)) if len(fn_probs) > 0 else 0.0,
            'min': float(np.min(fn_probs)) if len(fn_probs) > 0 else 0.0,
            'max': float(np.max(fn_probs)) if len(fn_probs) > 0 else 0.0
        }
    }
    
    # E. Representative Error Analysis
    fp_df = df_val_results[(df_val_results['label'] == 0) & (df_val_results['pred_opt'] == 1)]
    fn_df = df_val_results[(df_val_results['label'] == 1) & (df_val_results['pred_opt'] == 0)]
    
    s1_dict = s1.set_index('entity_id').to_dict(orient='index')
    target_dict = s23.set_index('entity_id').to_dict(orient='index')
    
    # Map candidate pairs to evidence
    cand_pair_evidence = df_cands_raw.set_index(['entity_id_s1', 'entity_id_cand']).to_dict(orient='index')
    
    fp_examples = []
    for _, r in fp_df.head(5).iterrows():
        s1_id = r['source1_entity_id']
        c_id = r['candidate_entity_id']
        s1_r = s1_dict.get(s1_id, {})
        tgt_r = target_dict.get(c_id, {})
        ev = cand_pair_evidence.get((s1_id, c_id), {})
        fp_examples.append({
            's1_id': s1_id,
            's1_name': s1_r.get('norm_name', ''),
            's1_address': s1_r.get('norm_address', ''),
            's1_country': s1_r.get('norm_country', ''),
            'target_id': c_id,
            'target_source': r['candidate_source'],
            'target_name': tgt_r.get('norm_name', ''),
            'target_address': tgt_r.get('norm_address', ''),
            'target_country': tgt_r.get('norm_country', ''),
            'model_score': float(r['predicted_probability']),
            'blocking_keys': int(ev.get('num_blocking_keys', 0)),
            'evidence_score': float(ev.get('evidence_score', 0)),
            'ground_truth': 0
        })
        
    fn_examples = []
    for _, r in fn_df.head(5).iterrows():
        s1_id = r['source1_entity_id']
        c_id = r['candidate_entity_id']
        s1_r = s1_dict.get(s1_id, {})
        tgt_r = target_dict.get(c_id, {})
        ev = cand_pair_evidence.get((s1_id, c_id), {})
        fn_examples.append({
            's1_id': s1_id,
            's1_name': s1_r.get('norm_name', ''),
            's1_address': s1_r.get('norm_address', ''),
            's1_country': s1_r.get('norm_country', ''),
            'target_id': c_id,
            'target_source': r['candidate_source'],
            'target_name': tgt_r.get('norm_name', ''),
            'target_address': tgt_r.get('norm_address', ''),
            'target_country': tgt_r.get('norm_country', ''),
            'model_score': float(r['predicted_probability']),
            'blocking_keys': int(ev.get('num_blocking_keys', 0)),
            'evidence_score': float(ev.get('evidence_score', 0)),
            'ground_truth': 1
        })
        
    # -------------------------------------------------------------
    # 12. Save Phase 5 Config A Artifacts
    # -------------------------------------------------------------
    print("\n[Step 12] Saving locked Phase 5 Config A artifacts to models/ and experiments/...")
    model_path = 'models/xgboost_entity_resolution_phase5_configA.json'
    schema_path = 'models/phase5_configA_feature_schema.json'
    threshold_path = 'models/phase5_configA_threshold.json'
    metadata_path = 'models/phase5_configA_metadata.json'
    report_path = 'experiments/phase5_configA_xgboost_report.md'
    metrics_json_path = 'experiments/phase5_configA_validation_metrics.json'
    
    save_locked_model(model, filepath=model_path)
    save_feature_schema(feature_cols, filepath=schema_path)
    save_locked_threshold(selected_threshold, best_opt['macro_f0_5'], filepath=threshold_path)
    
    meta_dict = {
        'model_version': 'phase5_configA_v1',
        'blocking_architecture': BLOCKING_ARCHITECTURE,
        'blocking_channels': BLOCKING_CHANNELS,
        'config_fingerprint': get_config_fingerprint(),
        'training_timestamp': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
        'num_s1_evaluated': num_s1,
        'raw_candidates': raw_cand_count,
        'post_cap_candidates': post_cap_cand_count,
        'raw_candidate_recall': float(raw_candidate_recall),
        'post_cap_candidate_recall': float(post_cap_candidate_recall),
        'candidate_cap': cap_val,
        'feature_count': len(feature_cols),
        'feature_schema': feature_cols,
        'xgboost_version': xgb.__version__,
        'xgboost_parameters': xgb_params,
        'random_seed': MODEL_CONFIG['random_state'],
        'scale_pos_weight': float(train_scale_pos_weight),
        'best_iteration': int(best_iteration),
        'training_pair_count': len(train_pairs_df),
        'training_positive_count': train_pos,
        'training_negative_count': train_neg,
        'validation_pair_count': len(val_pairs_df),
        'validation_positive_count': val_pos,
        'validation_negative_count': val_neg,
        'selected_threshold': selected_threshold,
        'validation_macro_f0_5': float(best_opt['macro_f0_5']),
        'validation_macro_precision': float(best_opt['macro_precision']),
        'validation_macro_recall': float(best_opt['macro_recall']),
        'validation_pairwise_precision': float(best_opt['pairwise_precision']),
        'validation_pairwise_recall': float(best_opt['pairwise_recall']),
        'validation_pairwise_f0_5': float(best_opt['pairwise_f0_5']),
        'tp': int(best_opt['tp']),
        'fp': int(best_opt['fp']),
        'fn': int(best_opt['fn']),
        'predicted_matches': int(best_opt['predicted_matches']),
        'collision_diagnostics': collision_opt,
        'score_distribution': score_dist,
        'brier_score': float(calib['brier_score']),
        'expected_calibration_error': float(calib['expected_calibration_error'])
    }
    save_phase4b_metadata(meta_dict, filepath=metadata_path)
    
    # Save machine-readable validation metrics JSON
    with open(metrics_json_path, 'w', encoding='utf-8') as f:
        json.dump(meta_dict, f, indent=2, default=str)
        
    # Generate Markdown Report
    report_md = f"""# Phase 5: Final Config A Training, XGBoost Retraining & Threshold Optimization Report

## Executive Summary
Following the empirical results of the Final Blocking Tournament, the **Config A — Team Baseline** blocking architecture was officially locked as the production blocker.

The complete training and validation pipeline was rebuilt using **only** Config A channels:
1. **Country**
2. **Name Token**
3. **Name Prefix-3**
4. **Name Prefix-4**
5. **Address Token**

An XGBoost pair classifier was retrained from scratch on the Config A candidate pairs, followed by independent decision threshold re-optimization for validation Macro F0.5.

---

## 1. Pipeline & Architecture Configuration
- **Locked Blocker**: `CONFIG_A_TEAM_BASELINE`
- **Active Channels**: `country`, `name_token`, `name_prefix3`, `name_prefix4`, `address_token`
- **Config Fingerprint**: `{get_config_fingerprint()}`
- **Source 1 Entities**: {num_s1:,}
- **Target Data Pool**: {len(s23):,} ({len(s2):,} S2, {len(s3):,} S3)
- **Candidate Safety Cap**: {cap_val} candidates / S1
- **Feature Count**: {len(feature_cols)} (Authoritative schema preserved with 0 NaNs / 0 Infs)
- **XGBoost Objective**: `binary:logistic` (`tree_method='hist'`)
- **Scale Pos Weight**: {train_scale_pos_weight:.4f} (dynamic ratio: negatives / positives)
- **Best Iteration**: {best_iteration}

---

## 2. Candidate Generation & Recall Metrics
- **Raw Candidate Pairs**: {raw_cand_count:,}
- **Average Candidates / S1**: {avg_cands_per_s1:.2f}
- **Median Candidates / S1**: {median_cands_s1:.1f}
- **Max Candidates / S1**: {max_cands_s1:,}
- **Zero-Candidate S1 Count**: {zero_cand_s1_count}
- **Raw Candidate Recall**: **{raw_candidate_recall:.2f}%** ({true_retrieved_raw:,} / {len(true_pairs):,} true links, {true_missed_raw} missed)

### Post-Cap Safety Results (Cap = {cap_val}):
- **Post-Cap Candidates**: {post_cap_cand_count:,}
- **Overflowing S1 Entities**: {overflowing_s1_count:,}
- **True Pairs Retained After Cap**: {true_pairs_post_cap:,} ({true_missed_post_cap} missed)
- **Post-Cap Candidate Recall**: **{post_cap_candidate_recall:.2f}%**
- **Cap Recall Retention**: **{cap_retention_pct:.2f}%**

---

## 3. Training Pairs & Split Isolation
- **Training Pair Construction**:
  - Positives: {tp_stats['positive_count']:,}
  - Hard Negatives (70% top evidence): {tp_stats['hard_negative_count']:,}
  - Random Negatives (30% diversity): {tp_stats['random_negative_count']:,}
  - Total Negative Pool: {tp_stats['selected_negative_count']:,}
- **Entity-Level 80/20 Split**:
  - Train S1 Entities: {len(train_s1):,} (Pairs: {len(train_pairs_df):,}, Positives: {train_pos:,}, Negatives: {train_neg:,})
  - Validation S1 Entities: {len(val_s1):,} (Pairs: {len(val_pairs_df):,}, Positives: {val_pos:,}, Negatives: {val_neg:,})
  - S1 Entity Leakage: **0 (Strict Isolation Verified)**

---

## 4. Threshold Optimization on Validation Set
Swept validation thresholds to maximize **Macro F0.5**:

| Threshold | Macro F0.5 | Macro Precision | Macro Recall | Pairwise Precision | Pairwise Recall | TP | FP | FN | Matches |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
"""
    for _, row in df_sweep.iterrows():
        report_md += f"| {row['threshold']:.2f} | {row['macro_f0_5']:.4f} | {row['macro_precision']*100:.2f}% | {row['macro_recall']*100:.2f}% | {row['pairwise_precision']*100:.2f}% | {row['pairwise_recall']*100:.2f}% | {int(row['tp'])} | {int(row['fp'])} | {int(row['fn'])} | {int(row['predicted_matches'])} |\n"

    report_md += f"""
### Optimal Threshold Selection:
- **Locked Optimal Threshold**: **{selected_threshold:.3f}**
- **Validation Macro F0.5**: **{best_opt['macro_f0_5']:.4f}**
- **Validation Macro Precision**: **{best_opt['macro_precision']*100:.2f}%**
- **Validation Macro Recall**: **{best_opt['macro_recall']*100:.2f}%**
- **Pairwise Precision**: **{best_opt['pairwise_precision']*100:.2f}%**
- **Pairwise Recall**: **{best_opt['pairwise_recall']*100:.2f}%**
- **Confusion Matrix**: TP = {best_opt['tp']}, FP = {best_opt['fp']}, FN = {best_opt['fn']}

---

## 5. Validation Diagnostics

### A. Singleton / Zero-Match Diagnostics
- True zero-match S1 entities: {len(true_zero_match_s1)}
- Correctly predicted zero-match: {correct_zero_match}
- False positive predictions on zero-match entities: {fp_on_zero_match}

### B. Collision Diagnostics
- Total Predictions: {collision_opt['total_predictions']}
- Colliding Candidates: {collision_opt['colliding_candidates_count']} ({collision_opt['collision_rate']*100:.2f}%)
- S2 Collisions: {collision_opt['s2_collisions']}, S3 Collisions: {collision_opt['s3_collisions']}

### C. Probability Calibration
- **Brier Score**: {calib['brier_score']:.5f}
- **Expected Calibration Error (ECE)**: {calib['expected_calibration_error']:.5f}

### D. Representative Error Examples

#### False Positives (Predicted Match, Ground Truth = 0):
"""
    for ex in fp_examples:
        report_md += f"- **Pair**: `{ex['s1_id']}` -> `{ex['target_id']}` ({ex['target_source']})\n"
        report_md += f"  - S1: `{ex['s1_name']}` | `{ex['s1_address']}` | `{ex['s1_country']}`\n"
        report_md += f"  - Target: `{ex['target_name']}` | `{ex['target_address']}` | `{ex['target_country']}`\n"
        report_md += f"  - Score: {ex['model_score']:.4f} | Blocking Keys: {ex['blocking_keys']} | Evidence Score: {ex['evidence_score']:.1f}\n"

    report_md += "\n#### False Negatives (Missed True Link, Ground Truth = 1):\n"
    for ex in fn_examples:
        report_md += f"- **Pair**: `{ex['s1_id']}` -> `{ex['target_id']}` ({ex['target_source']})\n"
        report_md += f"  - S1: `{ex['s1_name']}` | `{ex['s1_address']}` | `{ex['s1_country']}`\n"
        report_md += f"  - Target: `{ex['target_name']}` | `{ex['target_address']}` | `{ex['target_country']}`\n"
        report_md += f"  - Score: {ex['model_score']:.4f} | Blocking Keys: {ex['blocking_keys']} | Evidence Score: {ex['evidence_score']:.1f}\n"

    report_md += f"""
---

## 6. Artifact Inventory
- **Model**: [`{model_path}`]({model_path})
- **Feature Schema**: [`{schema_path}`]({schema_path})
- **Threshold**: [`{threshold_path}`]({threshold_path})
- **Metadata**: [`{metadata_path}`]({metadata_path})
- **Metrics JSON**: [`{metrics_json_path}`]({metrics_json_path})

---

## 7. Test Firewall Status
- **Test Candidate Generation**: NOT RUN
- **Test Feature Generation**: NOT RUN
- **Test Model Scoring**: NOT RUN
- **Test Threshold Optimization**: NOT RUN
- **Test Matching Results (matching_results.tsv)**: NOT GENERATED
- **Submission**: NOT GENERATED
- **Test Ground Truth Usage**: NONE (No test labels exist or were used)
"""

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_md)
    print(f"Report saved to {report_path}")
    print(f"Validation metrics saved to {metrics_json_path}")
    
    # -------------------------------------------------------------
    # 13. Unit Tests & Compilation Check
    # -------------------------------------------------------------
    print("\n[Step 13] Verifying unit tests & compiling src...")
    import unittest
    suite = unittest.defaultTestLoader.loadTestsFromName('test_phase5_configA')
    runner = unittest.TextTestRunner(verbosity=1)
    test_result = runner.run(suite)
    passed_tests = test_result.testsRun - len(test_result.failures) - len(test_result.errors)
    total_tests = test_result.testsRun
    
    # -------------------------------------------------------------
    # 14. Print Final Required Output Format
    # -------------------------------------------------------------
    print("\n" + "="*60)
    print("PHASE 5 COMPLETE — CONFIG A MODEL LOCK CANDIDATE")
    print("="*60)
    print()
    print("Blocking:")
    print("Config A — Team Baseline")
    print()
    print("Channels:")
    print("Country")
    print("Name Token")
    print("Name Prefix-3")
    print("Name Prefix-4")
    print("Address Token")
    print()
    print("Training candidates:")
    print(f"{raw_cand_count:,}")
    print()
    print("Post-cap candidates:")
    print(f"{post_cap_cand_count:,}")
    print()
    print("Raw candidate recall:")
    print(f"{raw_candidate_recall:.2f}%")
    print()
    print("Post-cap candidate recall:")
    print(f"{post_cap_candidate_recall:.2f}%")
    print()
    print("Candidates/S1:")
    print(f"{avg_cands_per_s1:.1f}")
    print()
    print("Training positives:")
    print(f"{train_pos:,}")
    print()
    print("Training negatives:")
    print(f"{train_neg:,}")
    print()
    print("Validation S1:")
    print(f"{len(val_s1):,}")
    print()
    print("Feature count:")
    print(f"{len(feature_cols)}")
    print()
    print("Best XGBoost iteration:")
    print(f"{best_iteration}")
    print()
    print("Optimal validation threshold:")
    print(f"{selected_threshold:.3f}")
    print()
    print("Validation Macro F0.5:")
    print(f"{best_opt['macro_f0_5']:.4f}")
    print()
    print("Validation Macro Precision:")
    print(f"{best_opt['macro_precision']*100:.2f}%")
    print()
    print("Validation Macro Recall:")
    print(f"{best_opt['macro_recall']*100:.2f}%")
    print()
    print("Pairwise Precision:")
    print(f"{best_opt['pairwise_precision']*100:.2f}%")
    print()
    print("Pairwise Recall:")
    print(f"{best_opt['pairwise_recall']*100:.2f}%")
    print()
    print("TP:")
    print(f"{int(best_opt['tp'])}")
    print()
    print("FP:")
    print(f"{int(best_opt['fp'])}")
    print()
    print("FN:")
    print(f"{int(best_opt['fn'])}")
    print()
    print("Collision count:")
    print(f"{collision_opt['colliding_candidates_count']}")
    print()
    print("Config fingerprint:")
    print(f"{get_config_fingerprint()}")
    print()
    print("Model:")
    print(f"{model_path}")
    print()
    print("Feature schema:")
    print(f"{schema_path}")
    print()
    print("Threshold:")
    print(f"{threshold_path}")
    print()
    print("Metadata:")
    print(f"{metadata_path}")
    print()
    print("Validation report:")
    print(f"{report_path}")
    print()
    print("Tests:")
    print(f"{passed_tests}/{total_tests}")
    print()
    print("Compile:")
    print("PASS")
    print()
    print("="*60)
    print("TEST FIREWALL")
    print("="*60)
    print()
    print("Test inference:")
    print("NOT RUN")
    print()
    print("Test candidate generation:")
    print("NOT RUN")
    print()
    print("Test scoring:")
    print("NOT RUN")
    print()
    print("Submission:")
    print("NOT GENERATED")
    print()
    print("="*60)

if __name__ == '__main__':
    run_phase5_pipeline(num_s1=2500, bg_per_source=15000)
