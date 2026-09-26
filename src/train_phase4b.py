"""
Phase 4B: XGBoost Pair Classifier Training & Validation Pipeline

Executes the complete Phase 4B workflow on TRAIN data:
1. Ingestion: 2,500 Source 1 records + true matches + controlled target background
2. Conservative Normalization (src/normalize.py)
3. Multi-Key Inverted-Index Blocking with Evidence Retention (src/blocking.py)
4. Candidate Safety Cap (src/candidate_safety.py)
5. Candidate Recall Diagnostics (before/after cap)
6. Training Pair Construction with Hard-Negative Sampling (src/training_pairs.py)
7. Entity-Level Train / Validation Split (src/split.py)
8. Transductive Unlabeled TF-IDF Vectorizer Fitting (src/tfidf_model.py)
9. Vectorized 57-Feature Matrix Extraction (src/features.py & src/build_features.py)
10. Numerical Safety Audit (NaN == 0, Inf == 0, strictly no ground-truth leakage)
11. Dynamic Class Imbalance scale_pos_weight computation
12. XGBoost Binary Pair Classifier Training with Early Stopping
13. Probability Calibration Diagnostics (Brier score, ECE, probability bins)
14. Macro F0.5 Threshold Optimization & Fine Search
15. Many-to-Many Collision Diagnostics (src/collision.py)
16. Error Analysis at Selected Threshold (representative FPs and FNs)
17. Model & Artifact Locking:
    - models/xgboost_entity_resolution_phase4b.json
    - models/phase4b_feature_schema.json
    - models/phase4b_threshold.json
    - models/phase4b_metadata.json
18. Generates experiments/phase4b_xgboost_report.md
"""

import os
import sys
import time
import tracemalloc
import json
import numpy as np
import pandas as pd
import xgboost as xgb

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from config import (
    get_config_fingerprint,
    BLOCKING_CONFIG,
    CANDIDATE_CAP_CONFIG,
    TRAINING_PAIRS_CONFIG,
    SPLIT_CONFIG,
    MODEL_CONFIG,
    PIPELINE_VERSION
)
from normalize import normalize_business_name, normalize_business_address, normalize_country
from blocking import ProductionBlockingEngine
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

def run_phase4b_pipeline(num_s1=2500, bg_per_source=15000):
    print("="*80)
    print("PHASE 4B: XGBOOST PAIR CLASSIFIER & MACRO F0.5 OPTIMIZATION")
    print(f"Pipeline Version: {PIPELINE_VERSION} | Fingerprint: {get_config_fingerprint()}")
    print("="*80)
    
    tracemalloc.start()
    t_global_start = time.time()
    
    # -------------------------------------------------------------
    # 1. Ingestion: Load Source 1 training records + Ground Truth
    # -------------------------------------------------------------
    print(f"\n[Step 1] Loading {num_s1} Source 1 records and ground-truth true pairs...")
    s1 = pd.read_csv('data/train/train_source1.tsv', sep='\t', dtype=str, nrows=num_s1, na_filter=False)
    s1_ids = set(s1['entity_id'])
    
    true_pairs = load_ground_truth_map('data/train/train_ground_truth.tsv', s1_ids=s1_ids)
    target_s2_ids = {p[1] for p in true_pairs if p[1].startswith('S2')}
    target_s3_ids = {p[1] for p in true_pairs if p[1].startswith('S3')}
    print(f"Loaded {len(s1)} S1 entities. Total Ground-Truth True Pairs: {len(true_pairs):,} (S2: {len(target_s2_ids):,}, S3: {len(target_s3_ids):,})")
    
    # Load controlled background + true matches for S2 and S3
    print(f"[Step 1b] Scanning target sources ({bg_per_source} background records each)...")
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
    # 3. Multi-Key Blocking with Evidence Retention
    # -------------------------------------------------------------
    print("\n[Step 3] Multi-Key Inverted Index Blocking with evidence tracking...")
    t_block_start = time.time()
    engine = ProductionBlockingEngine(s23)
    df_cands_raw = engine.generate_candidates_with_evidence(s1)
    t_block_elapsed = time.time() - t_block_start
    print(f"Blocking complete in {t_block_elapsed:.2f}s! Raw candidate pairs generated: {len(df_cands_raw):,}")
    
    # Measure raw blocking recall
    cand_pairs_raw_set = set(zip(df_cands_raw['entity_id_s1'], df_cands_raw['entity_id_cand']))
    true_retrieved_raw = len(cand_pairs_raw_set.intersection(true_pairs))
    blocking_recall_raw = true_retrieved_raw / len(true_pairs) if len(true_pairs) > 0 else 0
    print(f"Raw Blocking True Pairs: {true_retrieved_raw:,} / {len(true_pairs):,} (Recall: {blocking_recall_raw*100:.2f}%)")
    
    # -------------------------------------------------------------
    # 4. Candidate Safety Cap & Evidence Ranking
    # -------------------------------------------------------------
    print("\n[Step 4] Applying deterministic candidate safety cap (max_cands=350)...")
    capped_cands, cap_metrics = apply_candidate_safety_cap(
        df_cands_raw,
        max_candidates_per_s1=350,
        true_pairs_set=true_pairs
    )
    print("Candidate Safety Cap Metrics:")
    print(f"  - Candidates before cap: {cap_metrics['before_cap_candidates']:,}")
    print(f"  - Candidates after cap: {cap_metrics['after_cap_candidates']:,}")
    print(f"  - Overflowing S1 entities: {cap_metrics['overflowing_s1_count']}")
    print(f"  - True pairs before cap: {cap_metrics['true_pairs_before_cap']:,}")
    print(f"  - True pairs after cap: {cap_metrics['true_pairs_after_cap']:,}")
    print(f"  - Cap Recall Retention: {cap_metrics['cap_recall_retention']*100:.2f}%")
    combined_recall = (cap_metrics['true_pairs_after_cap'] / len(true_pairs)) if len(true_pairs) > 0 else 0
    print(f"  - Combined Final Candidate Recall: {combined_recall*100:.2f}%")
    
    # -------------------------------------------------------------
    # 5. Training Pair Construction with Hard Negatives
    # -------------------------------------------------------------
    print("\n[Step 5] Constructing training pairs (surviving positives + evidence-ranked hard negatives)...")
    df_training_pairs, tp_stats = construct_training_pairs(
        capped_cands,
        true_pairs_set=true_pairs,
        hard_neg_ratio=TRAINING_PAIRS_CONFIG['hard_negative_ratio'],
        random_seed=TRAINING_PAIRS_CONFIG['random_seed']
    )
    print("Training Pair Construction Summary:")
    print(f"  - Positive pairs: {tp_stats['positive_count']:,}")
    print(f"  - Selected negative pairs: {tp_stats['selected_negative_count']:,}")
    print(f"    * Hard negatives (Top 70% by evidence score): {tp_stats['hard_negative_count']:,}")
    print(f"    * Random negatives (30% for diversity): {tp_stats['random_negative_count']:,}")
    print(f"  - Total training candidate pairs: {len(df_training_pairs):,}")
    print(f"  - Dynamic scale_pos_weight: {tp_stats['scale_pos_weight']:.4f}")
    
    # -------------------------------------------------------------
    # 6. Entity-Level Train / Validation Split
    # -------------------------------------------------------------
    print("\n[Step 6] Enforcing strict Source 1 entity-level split (80/20)...")
    train_pairs_df, val_pairs_df, train_s1, val_s1 = split_candidate_pairs(
        df_training_pairs,
        train_ratio=SPLIT_CONFIG['train_ratio'],
        random_seed=SPLIT_CONFIG['random_seed']
    )
    
    # Programmatic leakage verification
    leakage = train_s1.intersection(val_s1)
    if len(leakage) > 0:
        raise ValueError(f"CRITICAL LEAKAGE DETECTED: {len(leakage)} Source 1 IDs exist in both train and validation sets!")
    print(f"Verified strict entity-level isolation: Train S1 IDs ({len(train_s1)}) intersect Validation S1 IDs ({len(val_s1)}) = EMPTY.")
    
    train_pos = int((train_pairs_df['label'] == 1).sum())
    train_neg = int((train_pairs_df['label'] == 0).sum())
    val_pos = int((val_pairs_df['label'] == 1).sum())
    val_neg = int((val_pairs_df['label'] == 0).sum())
    
    train_scale_pos_weight = (train_neg / train_pos) if train_pos > 0 else 1.0
    print(f"Split Pair Counts:")
    print(f"  - Train: {len(train_pairs_df):,} pairs (Pos: {train_pos:,}, Neg: {train_neg:,}, scale_pos_weight={train_scale_pos_weight:.4f})")
    print(f"  - Validation: {len(val_pairs_df):,} pairs (Pos: {val_pos:,}, Neg: {val_neg:,})")
    
    # -------------------------------------------------------------
    # 7. Transductive Unlabeled TF-IDF Fitting
    # -------------------------------------------------------------
    print("\n[Step 7] Fitting Transductive Unlabeled TF-IDF on raw text (Train + Test slices)...")
    # Load test slices strictly for unlabeled text vocabulary (NO labels, NO predictions)
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
    print(f"Transductive TF-IDF fit complete! Name Vocab: {len(tfidf_models['name_tfidf'].vocabulary_):,}, Address Vocab: {len(tfidf_models['address_tfidf'].vocabulary_):,}")
    
    # -------------------------------------------------------------
    # 8. Feature Extraction on Train & Validation Splits
    # -------------------------------------------------------------
    print("\n[Step 8] Extracting 57 pairwise features for Train & Validation sets...")
    t_feat_start = time.time()
    df_feat_train = build_feature_table(train_pairs_df, s1, s23, tfidf_models=tfidf_models)
    df_feat_val = build_feature_table(val_pairs_df, s1, s23, tfidf_models=tfidf_models)
    t_feat_elapsed = time.time() - t_feat_start
    print(f"Features extracted in {t_feat_elapsed:.2f}s! Train Shape: {df_feat_train.shape}, Validation Shape: {df_feat_val.shape}")
    
    # Feature columns check
    feature_cols = [c for c in df_feat_train.columns if c not in ('source1_entity_id', 'candidate_entity_id', 'candidate_source')]
    assert feature_cols == FEATURE_NAMES, "Feature columns ordering does not match FEATURE_NAMES exactly!"
    
    # Numerical safety audit
    train_nans = df_feat_train[feature_cols].isna().sum().sum()
    train_infs = np.isinf(df_feat_train[feature_cols]).sum().sum()
    val_nans = df_feat_val[feature_cols].isna().sum().sum()
    val_infs = np.isinf(df_feat_val[feature_cols]).sum().sum()
    
    print(f"Numerical Safety Check: Train NaNs: {train_nans}, Infs: {train_infs} | Val NaNs: {val_nans}, Infs: {val_infs}")
    if train_nans > 0 or train_infs > 0 or val_nans > 0 or val_infs > 0:
        raise ValueError("Non-finite values detected in feature matrix! Halting training.")
        
    X_train = df_feat_train[feature_cols].values
    y_train = train_pairs_df['label'].values
    X_val = df_feat_val[feature_cols].values
    y_val = val_pairs_df['label'].values
    
    # -------------------------------------------------------------
    # 9. Train XGBoost Binary Pair Classifier
    # -------------------------------------------------------------
    print(f"\n[Step 9] Training XGBoost Pair Classifier (scale_pos_weight={train_scale_pos_weight:.4f})...")
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
    best_iteration = model.best_iteration if hasattr(model, 'best_iteration') else xgb_params['n_estimators']
    print(f"XGBoost training completed in {t_train_elapsed:.2f}s! Best iteration: {best_iteration}")
    
    # -------------------------------------------------------------
    # 10. Generate Validation Probabilities
    # -------------------------------------------------------------
    val_probs = model.predict_proba(X_val)[:, 1]
    
    df_val_results = pd.DataFrame({
        'source1_entity_id': df_feat_val['source1_entity_id'],
        'candidate_entity_id': df_feat_val['candidate_entity_id'],
        'candidate_source': df_feat_val['candidate_source'],
        'label': y_val,
        'predicted_probability': val_probs
    })
    
    # -------------------------------------------------------------
    # 11. Baseline Evaluation (Threshold = 0.50)
    # -------------------------------------------------------------
    print("\n[Step 11] Evaluating Baseline Validation Metrics at threshold = 0.50...")
    y_pred_50 = (val_probs >= 0.50).astype(int)
    pair_50 = compute_pairwise_metrics(y_val, y_pred_50)
    pred_pairs_50 = set(zip(df_val_results.loc[y_pred_50 == 1, 'source1_entity_id'],
                            df_val_results.loc[y_pred_50 == 1, 'candidate_entity_id']))
    macro_50 = compute_macro_f05(val_s1, true_pairs, pred_pairs_50)
    
    print(f"Baseline (threshold=0.50):")
    print(f"  - Pairwise: Precision: {pair_50['precision']*100:.2f}%, Recall: {pair_50['recall']*100:.2f}%, F0.5: {pair_50['f0_5']:.4f}")
    print(f"  - Macro: Precision: {macro_50['macro_precision']*100:.2f}%, Recall: {macro_50['macro_recall']*100:.2f}%, F0.5: {macro_50['macro_f0_5']:.4f}")
    print(f"  - Predicted Matches: {pair_50['predicted_matches']:,} (TP: {pair_50['tp']}, FP: {pair_50['fp']}, FN: {pair_50['fn']})")
    
    # -------------------------------------------------------------
    # 12. Probability Calibration Diagnostics
    # -------------------------------------------------------------
    print("\n[Step 12] Probability Calibration Diagnostics...")
    calib = compute_calibration_diagnostics(y_val, val_probs, n_bins=10)
    print(f"  - Brier Score: {calib['brier_score']:.5f}")
    print(f"  - Expected Calibration Error (ECE): {calib['expected_calibration_error']:.5f}")
    print("\nProbability Calibration Table:")
    print(calib['bins_table'].to_string(index=False))
    
    # -------------------------------------------------------------
    # 13. Macro F0.5 Threshold Optimization & Sweep
    # -------------------------------------------------------------
    print("\n[Step 13] Optimizing Threshold for Macro F0.5...")
    threshold_grid = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98]
    df_sweep = evaluate_threshold_grid(df_val_results, val_s1, true_pairs, thresholds=threshold_grid)
    
    print("\n--- THRESHOLD SWEEP RESULTS ---")
    display_cols = ['threshold', 'macro_f0_5', 'macro_precision', 'macro_recall', 'pairwise_f0_5', 'pairwise_precision', 'pairwise_recall', 'tp', 'fp', 'fn', 'predicted_matches', 'zero_match_s1_count']
    print(df_sweep[display_cols].to_string(index=False))
    
    # Perform finer search around peak
    best_row_coarse = df_sweep.loc[df_sweep['macro_f0_5'].idxmax()]
    t_center = best_row_coarse['threshold']
    fine_grid = sorted(list(set(np.round(np.arange(max(0.05, t_center - 0.08), min(0.99, t_center + 0.08), 0.01), 3))))
    df_fine_sweep = evaluate_threshold_grid(df_val_results, val_s1, true_pairs, thresholds=fine_grid)
    
    # Deterministic selection rule: maximize macro_f0_5, break ties by higher macro_precision, then higher threshold
    df_fine_sorted = df_fine_sweep.sort_values(by=['macro_f0_5', 'macro_precision', 'threshold'], ascending=[False, False, False])
    best_opt = df_fine_sorted.iloc[0]
    selected_threshold = float(best_opt['threshold'])
    
    print("\n" + "*"*60)
    print(f"LOCKED OPTIMAL THRESHOLD: {selected_threshold:.3f}")
    print(f"  - Validation Macro F0.5: {best_opt['macro_f0_5']:.4f}")
    print(f"  - Validation Macro Precision: {best_opt['macro_precision']*100:.2f}%")
    print(f"  - Validation Macro Recall: {best_opt['macro_recall']*100:.2f}%")
    print(f"  - Pairwise Precision: {best_opt['pairwise_precision']*100:.2f}%, Recall: {best_opt['pairwise_recall']*100:.2f}%, F0.5: {best_opt['pairwise_f0_5']:.4f}")
    print(f"  - TP: {best_opt['tp']}, FP: {best_opt['fp']}, FN: {best_opt['fn']}")
    print(f"  - Predicted Matches: {best_opt['predicted_matches']}, Zero-Match S1 Count: {best_opt['zero_match_s1_count']}")
    print("*"*60)
    
    # -------------------------------------------------------------
    # 14. Collision Analysis at Baseline (0.50) and Optimal Threshold
    # -------------------------------------------------------------
    print("\n[Step 14] Collision Diagnostics at Optimal Threshold...")
    y_pred_opt = (val_probs >= selected_threshold).astype(int)
    opt_pred_df = df_val_results[y_pred_opt == 1].copy()
    
    collision_baseline = analyze_collisions(df_val_results[y_pred_50 == 1])
    collision_opt = analyze_collisions(opt_pred_df)
    
    print(f"Collision Diagnostics Comparison:")
    print(f"  - At threshold=0.50: Predictions={collision_baseline['total_predictions']}, Colliding Candidates={collision_baseline['colliding_candidates_count']} ({collision_baseline['collision_rate']*100:.1f}%), S2={collision_baseline['s2_collisions']}, S3={collision_baseline['s3_collisions']}")
    print(f"  - At threshold={selected_threshold:.2f}: Predictions={collision_opt['total_predictions']}, Colliding Candidates={collision_opt['colliding_candidates_count']} ({collision_opt['collision_rate']*100:.1f}%), S2={collision_opt['s2_collisions']}, S3={collision_opt['s3_collisions']}")
    
    # -------------------------------------------------------------
    # 15. Representative Error Analysis (False Positives & False Negatives)
    # -------------------------------------------------------------
    print("\n[Step 15] Extracting Representative Error Examples...")
    df_val_results['pred_opt'] = y_pred_opt
    
    fp_df = df_val_results[(df_val_results['label'] == 0) & (df_val_results['pred_opt'] == 1)]
    fn_df = df_val_results[(df_val_results['label'] == 1) & (df_val_results['pred_opt'] == 0)]
    
    s1_name_dict = s1.set_index('entity_id')['norm_name'].to_dict()
    s1_addr_dict = s1.set_index('entity_id')['norm_address'].to_dict()
    cand_name_dict = s23.set_index('entity_id')['norm_name'].to_dict()
    cand_addr_dict = s23.set_index('entity_id')['norm_address'].to_dict()
    
    fp_examples = []
    for _, r in fp_df.head(5).iterrows():
        s1_id = r['source1_entity_id']
        c_id = r['candidate_entity_id']
        fp_examples.append({
            'pair': f"{s1_id} -> {c_id} ({r['candidate_source']})",
            'prob': f"{r['predicted_probability']:.4f}",
            's1_name': s1_name_dict.get(s1_id, ''),
            'cand_name': cand_name_dict.get(c_id, ''),
            's1_addr': s1_addr_dict.get(s1_id, ''),
            'cand_addr': cand_addr_dict.get(c_id, '')
        })
        
    fn_examples = []
    for _, r in fn_df.head(5).iterrows():
        s1_id = r['source1_entity_id']
        c_id = r['candidate_entity_id']
        fn_examples.append({
            'pair': f"{s1_id} -> {c_id} ({r['candidate_source']})",
            'prob': f"{r['predicted_probability']:.4f}",
            's1_name': s1_name_dict.get(s1_id, ''),
            'cand_name': cand_name_dict.get(c_id, ''),
            's1_addr': s1_addr_dict.get(s1_id, ''),
            'cand_addr': cand_addr_dict.get(c_id, '')
        })
        
    # -------------------------------------------------------------
    # 16. Lock Model, Feature Schema, Threshold, and Metadata
    # -------------------------------------------------------------
    print("\n[Step 16] Locking Model and Artifacts to models/...")
    model_path = save_locked_model(model, filepath='models/xgboost_entity_resolution_phase4b.json')
    schema_path = save_feature_schema(feature_cols, filepath='models/phase4b_feature_schema.json')
    thresh_path = save_locked_threshold(selected_threshold, best_opt['macro_f0_5'], filepath='models/phase4b_threshold.json')
    
    metadata = {
        'model_version': 'phase4b_v1',
        'training_timestamp': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
        'num_s1_evaluated': num_s1,
        'feature_count': len(feature_cols),
        'feature_schema': feature_cols,
        'config_fingerprint': get_config_fingerprint(),
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
        'validation_pairwise_f0_5': float(best_opt['pairwise_f0_5']),
        'validation_pairwise_precision': float(best_opt['pairwise_precision']),
        'validation_pairwise_recall': float(best_opt['pairwise_recall']),
        'tp': int(best_opt['tp']),
        'fp': int(best_opt['fp']),
        'fn': int(best_opt['fn']),
        'predicted_matches': int(best_opt['predicted_matches']),
        'zero_match_s1_count': int(best_opt['zero_match_s1_count']),
        'brier_score': calib['brier_score'],
        'expected_calibration_error': calib['expected_calibration_error'],
        'collision_diagnostics': collision_opt,
        'blocking_recall_raw': blocking_recall_raw,
        'candidate_cap_recall': combined_recall
    }
    meta_path = save_phase4b_metadata(metadata, filepath='models/phase4b_metadata.json')
    print(f"Artifacts locked successfully:")
    print(f"  - Model: {model_path}")
    print(f"  - Feature Schema: {schema_path}")
    print(f"  - Threshold: {thresh_path}")
    print(f"  - Metadata: {meta_path}")
    
    # -------------------------------------------------------------
    # 17. Generate Detailed Report: experiments/phase4b_xgboost_report.md
    # -------------------------------------------------------------
    print("\n[Step 17] Writing experiments/phase4b_xgboost_report.md...")
    total_pipeline_time = time.time() - t_global_start
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak_bytes / (1024 * 1024)
    
    os.makedirs('experiments', exist_ok=True)
    report_path = 'experiments/phase4b_xgboost_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Phase 4B: XGBoost Pair Classifier & Macro F0.5 Optimization Report\n\n")
        f.write(f"- **Pipeline Version:** {PIPELINE_VERSION}\n")
        f.write(f"- **Config Fingerprint:** `{get_config_fingerprint()}`\n")
        f.write(f"- **Total Runtime:** {total_pipeline_time:.1f}s | **Peak RAM:** {peak_mb:.1f} MB\n")
        f.write("- **Status:** MODEL & THRESHOLD LOCKED (NO TEST INFERENCE RUN)\n\n")
        
        f.write("## 1. Training & Candidate Data Summary\n\n")
        f.write(f"- **Source 1 entities evaluated:** {num_s1:,}\n")
        f.write(f"- **Target records pool (S2+S3):** {len(s23):,} ({len(s2):,} S2, {len(s3):,} S3)\n")
        f.write(f"- **Ground-Truth True Pairs:** {len(true_pairs):,}\n")
        f.write(f"- **Candidate Pairs Before Cap:** {cap_metrics['before_cap_candidates']:,}\n")
        f.write(f"- **Candidate Pairs After Cap (max=350/S1):** {cap_metrics['after_cap_candidates']:,}\n")
        f.write(f"- **Overflowing S1 Entities:** {cap_metrics['overflowing_s1_count']}\n")
        f.write(f"- **Candidate Recall (Blocking):** {blocking_recall_raw*100:.2f}%\n")
        f.write(f"- **Candidate Recall (Post-Cap):** {combined_recall*100:.2f}% (Cap retention: {cap_metrics['cap_recall_retention']*100:.2f}%)\n")
        f.write(f"- **Training Pairs Constructed:** {len(df_training_pairs):,} (Pos: {tp_stats['positive_count']:,}, Neg: {tp_stats['selected_negative_count']:,})\n")
        f.write(f"- **Hard Negatives Selected:** {tp_stats['hard_negative_count']:,} (Top 70% by blocking evidence score)\n")
        f.write(f"- **Random Negatives Selected:** {tp_stats['random_negative_count']:,}\n\n")
        
        f.write("## 2. Strict Entity-Level Train / Validation Split\n\n")
        f.write(f"- **Train S1 Entities:** {len(train_s1):,} ({len(train_pairs_df):,} pairs; Pos: {train_pos:,}, Neg: {train_neg:,})\n")
        f.write(f"- **Validation S1 Entities:** {len(val_s1):,} ({len(val_pairs_df):,} pairs; Pos: {val_pos:,}, Neg: {val_neg:,})\n")
        f.write(f"- **Leakage Verification:** `Train S1 ∩ Val S1 = ∅` (Strictly verified)\n")
        f.write(f"- **Dynamic `scale_pos_weight`:** `{train_scale_pos_weight:.4f}`\n\n")
        
        f.write("## 3. Feature Matrix & Numerical Safety\n\n")
        f.write(f"- **Feature Count:** {len(feature_cols)}\n")
        f.write(f"- **Feature Groups:** Name (14), Address (10), Country (5), Directional Missingness (8), Structural (10), Blocking Evidence (6), Source (2), TF-IDF Cosine (2)\n")
        f.write(f"- **Total NaNs:** 0 | **Total Infs:** 0\n")
        f.write(f"- **Transductive TF-IDF:** Fitted on unlabeled text union across train and test without ground-truth labels.\n\n")
        
        f.write("## 4. XGBoost Model Configuration\n\n")
        f.write("```json\n" + json.dumps(xgb_params, indent=2) + "\n```\n\n")
        
        f.write("## 5. Probability Calibration Diagnostics\n\n")
        f.write(f"- **Brier Score:** `{calib['brier_score']:.5f}`\n")
        f.write(f"- **Expected Calibration Error (ECE):** `{calib['expected_calibration_error']:.5f}`\n\n")
        f.write("```text\n" + calib['bins_table'].to_string(index=False) + "\n```\n\n")
        
        f.write("## 6. Threshold Sweep & Macro F0.5 Optimization\n\n")
        f.write("```text\n" + df_sweep[display_cols].to_string(index=False) + "\n```\n\n")
        
        f.write("## 7. Locked Model & Threshold Selection\n\n")
        f.write(f"- **Selected Classification Threshold:** `{selected_threshold:.3f}`\n")
        f.write(f"- **Validation Macro F0.5:** `{best_opt['macro_f0_5']:.4f}`\n")
        f.write(f"- **Validation Macro Precision:** `{best_opt['macro_precision']*100:.2f}%`\n")
        f.write(f"- **Validation Macro Recall:** `{best_opt['macro_recall']*100:.2f}%`\n")
        f.write(f"- **Pairwise F0.5:** `{best_opt['pairwise_f0_5']:.4f}` (Precision: `{best_opt['pairwise_precision']*100:.2f}%`, Recall: `{best_opt['pairwise_recall']*100:.2f}%`)\n")
        f.write(f"- **Confusion Matrix:** TP = {best_opt['tp']}, FP = {best_opt['fp']}, FN = {best_opt['fn']}\n")
        f.write(f"- **Predicted Matches:** {best_opt['predicted_matches']:,} across validation S1 entities\n")
        f.write(f"- **Zero-Match S1 Count:** {best_opt['zero_match_s1_count']} / {len(val_s1)} (preserves correct singletons/empty predictions)\n\n")
        
        f.write("## 8. Many-to-Many Collision Diagnostics\n\n")
        f.write(f"- **Total Predicted Matches:** {collision_opt['total_predictions']:,}\n")
        f.write(f"- **Unique S1 Entities with Matches:** {collision_opt['unique_s1']}\n")
        f.write(f"- **Unique Candidates Predicted:** {collision_opt['unique_candidates']}\n")
        f.write(f"- **Colliding Candidates (matched to >1 S1):** {collision_opt['colliding_candidates_count']} ({collision_opt['collision_rate']*100:.2f}%)\n")
        f.write(f"  * S2 Collisions: {collision_opt['s2_collisions']}\n")
        f.write(f"  * S3 Collisions: {collision_opt['s3_collisions']}\n\n")
        
        f.write("## 9. Representative Error Analysis\n\n")
        f.write("### False Positive Examples (Predicted Match, Ground Truth = 0):\n")
        for ex in fp_examples:
            f.write(f"- **Pair:** `{ex['pair']}` | **p:** `{ex['prob']}`\n")
            f.write(f"  - S1: `{ex['s1_name']}` | `{ex['s1_addr']}`\n")
            f.write(f"  - Cand: `{ex['cand_name']}` | `{ex['cand_addr']}`\n")
        f.write("\n### False Negative Examples (Predicted Non-Match, Ground Truth = 1):\n")
        for ex in fn_examples:
            f.write(f"- **Pair:** `{ex['pair']}` | **p:** `{ex['prob']}`\n")
            f.write(f"  - S1: `{ex['s1_name']}` | `{ex['s1_addr']}`\n")
            f.write(f"  - Cand: `{ex['cand_name']}` | `{ex['cand_addr']}`\n")
            
        f.write("\n## 10. Locked Artifacts & Verification\n\n")
        f.write(f"- Model: `{model_path}`\n")
        f.write(f"- Feature Schema: `{schema_path}`\n")
        f.write(f"- Threshold: `{thresh_path}`\n")
        f.write(f"- Metadata: `{meta_path}`\n\n")
        f.write("### Absolute Test Inference Firewall Confirmation:\n")
        f.write("- Test inference was NOT executed.\n")
        f.write("- No test records were evaluated by the classifier.\n")
        f.write("- No submission files were produced.\n")
        
    print(f"\nPhase 4B report written to {report_path}")
    print("\n" + "="*80)
    print("PHASE 4B COMPLETE: MODEL AND THRESHOLD LOCKED")
    print("="*80)
    
    return {
        'model_path': model_path,
        'schema_path': schema_path,
        'thresh_path': thresh_path,
        'selected_threshold': selected_threshold,
        'macro_f0_5': float(best_opt['macro_f0_5']),
        'macro_precision': float(best_opt['macro_precision']),
        'macro_recall': float(best_opt['macro_recall']),
        'pairwise_f0_5': float(best_opt['pairwise_f0_5']),
        'config_fingerprint': get_config_fingerprint()
    }

if __name__ == '__main__':
    run_phase4b_pipeline()
