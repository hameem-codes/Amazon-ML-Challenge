"""
PHASE 10: PER-S1 ADAPTIVE PREDICTION SELECTION EXPERIMENT

Tests:
1. Baseline @ threshold 0.99 (reproduced)
2. Experiment A: Top-K per S1 (K in [1, 2, 3, 5, 10, 20, 50, 100])
3. Experiment B: Score-gap based adaptive stopping (gaps in [0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01])
4. Experiment C: Evidence-aware adaptive selection (C1, C2, C3, C4)
5. Experiment D: Zero-match safety diagnostic across S1 groups
"""

import sys, os, time, gc, json, math
from collections import defaultdict, Counter
sys.path.insert(0, 'src')
sys.path.insert(0, '.')
import pandas as pd
import numpy as np

from normalize import normalize_business_name, normalize_country, normalize_business_address
from config import get_config_fingerprint, BLOCKING_CHUNK_SIZE
from features import build_vectorized_features, FEATURE_NAMES
from metrics import compute_macro_f05, compute_pairwise_metrics
from tfidf_model import TransductiveTFIDF
from blocking import ConfigABlockingEngine
from scratch.run_experiment_opt_e7 import generate_bounded_candidates_e7


def fit_transductive_tfidf():
    print("  Fitting Transductive Unlabeled TF-IDF on raw text (Train + Unlabeled Test slices)...", flush=True)
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
    return tfidf.get_models_dict()


def run_phase10():
    print("=" * 85, flush=True)
    print("PHASE 10: PER-S1 ADAPTIVE PREDICTION SELECTION EXPERIMENT", flush=True)
    print("=" * 85, flush=True)

    # 1. Setup validation benchmark & ground truth
    print("\n[Step 1/6] Loading validation benchmark & ground truth...", flush=True)
    s1_df = pd.read_csv('data/test/test_source1.tsv', sep='\t', dtype=str, nrows=10000, na_filter=False)
    s1_fr = s1_df[s1_df['country'].apply(normalize_country) == 'france'].head(500).copy().reset_index(drop=True)
    del s1_df
    gc.collect()
    s1_fr['norm_name'] = s1_fr['business_name'].apply(normalize_business_name)
    s1_fr['norm_address'] = s1_fr['business_address'].apply(normalize_business_address)
    s1_fr['norm_country'] = 'france'
    val_s1_ids = list(s1_fr['entity_id'].values)

    targets = pd.read_pickle('scratch/targets_france.pkl')
    engine = ConfigABlockingEngine(targets)
    target_dict = targets.set_index('entity_id').to_dict(orient='index')
    s1_dict = s1_fr.set_index('entity_id').to_dict(orient='index')
    target_lex_rank = engine.target_lex_rank

    s1_name_map = defaultdict(list)
    for idx, r in s1_fr.iterrows():
        s1_name_map[r['norm_name']].append(r['entity_id'])

    exact_matches_set = set()
    t_names = targets['norm_name'].values
    t_eids = targets['entity_id'].values
    for idx in range(len(t_names)):
        t_name = t_names[idx]
        if t_name in s1_name_map:
            t_eid = t_eids[idx]
            for s1_id in s1_name_map[t_name]:
                exact_matches_set.add((s1_id, t_eid))

    gt_by_s1 = defaultdict(set)
    for s1_id, cand_id in exact_matches_set:
        gt_by_s1[s1_id].add(cand_id)

    zero_match_s1_ids = [s1 for s1 in val_s1_ids if len(gt_by_s1[s1]) == 0]
    singleton_s1_ids = [s1 for s1 in val_s1_ids if len(gt_by_s1[s1]) == 1]
    multi_match_s1_ids = [s1 for s1 in val_s1_ids if len(gt_by_s1[s1]) > 1]

    print(f"Entities: Total={len(val_s1_ids)}, Zero-match={len(zero_match_s1_ids)}, Singleton={len(singleton_s1_ids)}, Multi-match={len(multi_match_s1_ids)}")
    print(f"Reference True Pairs: {len(exact_matches_set):,}")

    # 2. Load candidate probabilities
    cache_file = 'scratch/cap400_scored_candidates.pkl'
    assert os.path.exists(cache_file), f"Missing cache file {cache_file}"
    cand_prob_df = pd.read_pickle(cache_file)
    print(f"Loaded {len(cand_prob_df):,} scored candidate pairs from {cache_file}", flush=True)

    # Pre-index targets lexical rank for deterministic sorting
    eid_to_lex = dict(zip(engine.target_eids, engine.target_lex_rank))
    cand_prob_df['target_lex_rank'] = [eid_to_lex[eid] for eid in cand_prob_df['entity_id_cand']]

    # Filter to predictions >= 0.99
    preds_099_df = cand_prob_df[cand_prob_df['probability'] >= 0.990].copy().reset_index(drop=True)
    print(f"Total candidate pairs with probability >= 0.990: {len(preds_099_df):,}", flush=True)

    # Verify Baseline metrics reproduction
    baseline_pairs = set(zip(preds_099_df['entity_id_s1'], preds_099_df['entity_id_cand']))
    base_macro = compute_macro_f05(val_s1_ids, exact_matches_set, baseline_pairs)
    base_tp = len(baseline_pairs & exact_matches_set)
    base_fp = len(baseline_pairs - exact_matches_set)
    base_fn = len(exact_matches_set - baseline_pairs)
    base_p_prec = base_tp / (base_tp + base_fp) if (base_tp + base_fp) > 0 else 0.0
    base_p_rec = base_tp / (base_tp + base_fn) if (base_tp + base_fn) > 0 else 0.0

    print(f"\n[Baseline Verification @ 0.99]:")
    print(f"  Macro F0.5: {base_macro['macro_f0_5']:.6f} (Expected: 0.125500)")
    print(f"  Macro Precision: {base_macro['macro_precision']*100:.3f}% (Expected: 11.125%)")
    print(f"  Macro Recall: {base_macro['macro_recall']*100:.3f}% (Expected: 83.142%)")
    print(f"  TP: {base_tp} (Expected: 2195), FP: {base_fp} (Expected: 17263), FN: {base_fn} (Expected: 1996)")
    assert abs(base_macro['macro_f0_5'] - 0.125500) < 1e-5, "Baseline F0.5 mismatch!"

    # Group predictions by S1
    # For every S1: list of (cand_id, probability, target_lex_rank) sorted by (prob DESC, lex ASC)
    preds_by_s1 = defaultdict(list)
    for row in preds_099_df.itertuples():
        preds_by_s1[row.entity_id_s1].append((row.entity_id_cand, float(row.probability), int(row.target_lex_rank)))

    for s1_id in preds_by_s1:
        preds_by_s1[s1_id].sort(key=lambda x: (-x[1], x[2]))

    def evaluate_prediction_set(pred_pairs_set):
        tp = len(pred_pairs_set & exact_matches_set)
        fp = len(pred_pairs_set - exact_matches_set)
        fn = len(exact_matches_set - pred_pairs_set)
        p_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        p_rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        macro = compute_macro_f05(val_s1_ids, exact_matches_set, pred_pairs_set)

        # per-S1 counts across all 500 S1
        pred_counts = Counter(p[0] for p in pred_pairs_set)
        counts_all = [pred_counts[s1] for s1 in val_s1_ids]
        mean_p = float(np.mean(counts_all))
        median_p = float(np.median(counts_all))

        # zero-match diagnostics
        zero_match_with_preds = sum(1 for s1 in zero_match_s1_ids if pred_counts[s1] > 0)
        zero_match_correct_empty = len(zero_match_s1_ids) - zero_match_with_preds

        # Metrics for groups
        m_zero = compute_macro_f05(zero_match_s1_ids, exact_matches_set, pred_pairs_set)
        m_sing = compute_macro_f05(singleton_s1_ids, exact_matches_set, pred_pairs_set)
        m_multi = compute_macro_f05(multi_match_s1_ids, exact_matches_set, pred_pairs_set)

        return {
            "macro_precision": round(macro['macro_precision'], 6),
            "macro_recall": round(macro['macro_recall'], 6),
            "macro_f0_5": round(macro['macro_f0_5'], 6),
            "pairwise_precision": round(p_prec, 6),
            "pairwise_recall": round(p_rec, 6),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "predicted_pairs": len(pred_pairs_set),
            "mean_predictions_per_s1": round(mean_p, 2),
            "median_predictions_per_s1": round(median_p, 2),
            "zero_match_with_predictions": zero_match_with_preds,
            "zero_match_correct_empty": zero_match_correct_empty,
            "group_metrics": {
                "zero_match": {
                    "macro_f0_5": round(m_zero['macro_f0_5'], 6),
                    "macro_precision": round(m_zero['macro_precision'], 6),
                    "macro_recall": round(m_zero['macro_recall'], 6),
                    "tp": len({p for p in pred_pairs_set if p[0] in set(zero_match_s1_ids)} & exact_matches_set),
                    "fp": len({p for p in pred_pairs_set if p[0] in set(zero_match_s1_ids)} - exact_matches_set)
                },
                "singleton": {
                    "macro_f0_5": round(m_sing['macro_f0_5'], 6),
                    "macro_precision": round(m_sing['macro_precision'], 6),
                    "macro_recall": round(m_sing['macro_recall'], 6),
                    "tp": len({p for p in pred_pairs_set if p[0] in set(singleton_s1_ids)} & exact_matches_set),
                    "fp": len({p for p in pred_pairs_set if p[0] in set(singleton_s1_ids)} - exact_matches_set)
                },
                "multi_match": {
                    "macro_f0_5": round(m_multi['macro_f0_5'], 6),
                    "macro_precision": round(m_multi['macro_precision'], 6),
                    "macro_recall": round(m_multi['macro_recall'], 6),
                    "tp": len({p for p in pred_pairs_set if p[0] in set(multi_match_s1_ids)} & exact_matches_set),
                    "fp": len({p for p in pred_pairs_set if p[0] in set(multi_match_s1_ids)} - exact_matches_set)
                }
            }
        }

    # =========================================================================
    # EXPERIMENT A: TOP-K
    # =========================================================================
    print("\n[Step 2/6] Running Experiment A: Top-K Selection...", flush=True)
    k_values = [1, 2, 3, 5, 10, 20, 50, 100]
    exp_a_results = {}

    for K in k_values:
        topk_pairs = set()
        for s1_id in val_s1_ids:
            cand_list = preds_by_s1.get(s1_id, [])
            for c_id, p, lex in cand_list[:K]:
                topk_pairs.add((s1_id, c_id))

        res = evaluate_prediction_set(topk_pairs)
        exp_a_results[f"Top-K {K}"] = res
        print(f"Top-K {K:3d}: F0.5={res['macro_f0_5']:.6f} | Prec={res['macro_precision']*100:6.3f}% | Rec={res['macro_recall']*100:6.3f}% | TP={res['tp']:5d} | FP={res['fp']:5d} | Preds={res['predicted_pairs']:5d}")

    # =========================================================================
    # EXPERIMENT B: ADAPTIVE K USING SCORE GAPS
    # =========================================================================
    print("\n[Step 3/6] Running Experiment B: Score-Gap Adaptive Selection...", flush=True)
    gap_thresholds = [0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01]
    exp_b_results = {}

    for gap_thresh in gap_thresholds:
        gap_pairs = set()
        for s1_id in val_s1_ids:
            cand_list = preds_by_s1.get(s1_id, [])
            if not cand_list:
                continue

            # Keep first prediction
            gap_pairs.add((s1_id, cand_list[0][0]))

            # Continue while gap <= gap_thresh
            for idx in range(len(cand_list) - 1):
                cur_p = cand_list[idx][1]
                next_p = cand_list[idx + 1][1]
                gap = cur_p - next_p
                if gap <= gap_thresh:
                    gap_pairs.add((s1_id, cand_list[idx + 1][0]))
                else:
                    break

        res = evaluate_prediction_set(gap_pairs)
        rule_name = f"Gap <= {gap_thresh}"
        exp_b_results[rule_name] = res
        print(f"{rule_name:15s}: F0.5={res['macro_f0_5']:.6f} | Prec={res['macro_precision']*100:6.3f}% | Rec={res['macro_recall']*100:6.3f}% | TP={res['tp']:5d} | FP={res['fp']:5d} | Preds={res['predicted_pairs']:5d}")

    # =========================================================================
    # EXPERIMENT C: EVIDENCE-AWARE ADAPTIVE SELECTION
    # =========================================================================
    print("\n[Step 4/6] Running Experiment C: Evidence-Aware Adaptive Selection...", flush=True)
    # Fit TF-IDF to build features for the 19,458 predictions >= 0.99
    tfidf_models = fit_transductive_tfidf()

    # Reconstruct blocking pairs df for feature generation
    pairs_for_feat = pd.DataFrame({
        'entity_id_s1': preds_099_df['entity_id_s1'].values,
        'entity_id_cand': preds_099_df['entity_id_cand'].values,
        'source': [target_dict[eid].get('source', 'S2' if eid.startswith('S2') else 'S3') for eid in preds_099_df['entity_id_cand']],
        'blocked_country': np.ones(len(preds_099_df), dtype=np.int8),
        'blocked_name_token': np.ones(len(preds_099_df), dtype=np.int8),
        'blocked_prefix_3': np.ones(len(preds_099_df), dtype=np.int8),
        'blocked_prefix_4': np.ones(len(preds_099_df), dtype=np.int8),
        'blocked_address_token': np.ones(len(preds_099_df), dtype=np.int8),
        'num_blocking_keys': np.full(len(preds_099_df), 4, dtype=np.int8),
        'evidence_score': np.full(len(preds_099_df), 100.0, dtype=np.float32),
        'blocked_exact_name': [1 if s1_dict[s1]['norm_name'] == target_dict[c]['norm_name'] else 0 for s1, c in zip(preds_099_df['entity_id_s1'], preds_099_df['entity_id_cand'])],
        'blocked_selective_token': np.ones(len(preds_099_df), dtype=np.int8),
        'blocked_rare_token': np.zeros(len(preds_099_df), dtype=np.int8),
        'blocked_token_pair': np.zeros(len(preds_099_df), dtype=np.int8),
        'blocked_char_ngram': np.zeros(len(preds_099_df), dtype=np.int8),
        'shared_ngram_count': np.zeros(len(preds_099_df), dtype=np.int8)
    })

    print(f"  Extracting features for {len(pairs_for_feat):,} candidate predictions >= 0.99...", flush=True)
    df_feat_099 = build_vectorized_features(pairs_for_feat, s1_dict, target_dict, tfidf_models=tfidf_models)

    # Look up actual blocking keys from inverted indexes for exact accuracy
    # In Config A:
    # Key 1: country (always true)
    # Key 2: name token
    # Key 3: prefix 4
    # Key 4: prefix 3
    # Key 5: address token
    actual_b_name = []
    actual_b_p4 = []
    actual_b_p3 = []
    actual_b_addr = []
    actual_num_keys = []

    country_name_tokens = engine.country_name_token_idx['france']
    country_p4 = engine.country_prefix4_idx['france']
    country_p3 = engine.country_prefix3_idx['france']
    country_addr_tokens = engine.country_addr_token_idx['france']
    target_eid_to_idx = {eid: idx for idx, eid in enumerate(engine.target_eids)}

    for s1_id, cand_id in zip(preds_099_df['entity_id_s1'], preds_099_df['entity_id_cand']):
        s1_row = s1_dict[s1_id]
        cand_idx = target_eid_to_idx[cand_id]

        s1_n = s1_row['norm_name']
        s1_a = s1_row['norm_address']

        # name tokens
        s1_toks = [t for t in s1_n.split() if len(t) > 1 and t not in engine.stop_tokens]
        b_name = 1 if any(t in country_name_tokens and cand_idx in country_name_tokens[t] for t in s1_toks) else 0

        # p4
        p4 = s1_n[:4] if len(s1_n) >= 4 else ''
        b_p4 = 1 if p4 and p4 in country_p4 and cand_idx in country_p4[p4] else 0

        # p3
        p3 = s1_n[:3] if len(s1_n) >= 3 else ''
        b_p3 = 1 if p3 and p3 in country_p3 and cand_idx in country_p3[p3] else 0

        # addr
        s1_addr_toks = set([t for t in s1_a.split() if len(t) >= 4 and t not in engine.stop_addrs]) if s1_a else set()
        b_addr = 1 if any(t in country_addr_tokens and cand_idx in country_addr_tokens[t] for t in s1_addr_toks) else 0

        n_keys = b_name + b_p4 + b_p3 + b_addr

        actual_b_name.append(b_name)
        actual_b_p4.append(b_p4)
        actual_b_p3.append(b_p3)
        actual_b_addr.append(b_addr)
        actual_num_keys.append(n_keys)

    preds_099_df['actual_num_keys'] = actual_num_keys
    preds_099_df['name_jaro_winkler'] = df_feat_099['name_jaro_winkler'].values
    preds_099_df['name_token_jaccard'] = df_feat_099['name_token_jaccard'].values
    preds_099_df['name_exact'] = df_feat_099['name_exact'].values
    preds_099_df['address_jaro_winkler'] = df_feat_099['address_jaro_winkler'].values
    preds_099_df['address_token_jaccard'] = df_feat_099['address_token_jaccard'].values
    preds_099_df['address_exact'] = df_feat_099['address_exact'].values

    # Define Evidence Rules:
    # C1: probability >= 0.99 AND shared blocking key count >= 2
    c1_mask = preds_099_df['actual_num_keys'] >= 2

    # C2: probability >= 0.99 AND strong name evidence
    # (name_exact == 1 OR name_token_jaccard >= 0.5 OR name_jaro_winkler >= 0.90)
    c2_mask = (preds_099_df['name_exact'] == 1) | (preds_099_df['name_token_jaccard'] >= 0.5) | (preds_099_df['name_jaro_winkler'] >= 0.90)

    # C3: probability >= 0.99 AND strong address evidence
    # (address_exact == 1 OR address_token_jaccard >= 0.5 OR address_jaro_winkler >= 0.85)
    c3_mask = (preds_099_df['address_exact'] == 1) | (preds_099_df['address_token_jaccard'] >= 0.5) | (preds_099_df['address_jaro_winkler'] >= 0.85)

    # C4: probability >= 0.99 AND (strong name OR strong address evidence)
    c4_mask = c2_mask | c3_mask

    exp_c_results = {}
    for rule_name, rule_mask in [
        ("C1: Keys >= 2", c1_mask),
        ("C2: Strong Name", c2_mask),
        ("C3: Strong Addr", c3_mask),
        ("C4: Strong Name or Addr", c4_mask)
    ]:
        filtered_sub = preds_099_df[rule_mask]
        c_pairs = set(zip(filtered_sub['entity_id_s1'], filtered_sub['entity_id_cand']))
        res = evaluate_prediction_set(c_pairs)
        exp_c_results[rule_name] = res
        print(f"{rule_name:25s}: F0.5={res['macro_f0_5']:.6f} | Prec={res['macro_precision']*100:6.3f}% | Rec={res['macro_recall']*100:6.3f}% | TP={res['tp']:5d} | FP={res['fp']:5d} | Preds={res['predicted_pairs']:5d}")

    # =========================================================================
    # EXPERIMENT D: ZERO-MATCH SAFETY DIAGNOSTIC
    # =========================================================================
    print("\n[Step 5/6] Running Experiment D: Zero-Match Safety Diagnostic...", flush=True)
    # For every S1 entity, compute diagnostics from all 400 candidates:
    # Load raw candidates for all 200,000 to get max features
    # Or from preds_099_df and cand_prob_df:
    s1_max_prob = cand_prob_df.groupby('entity_id_s1')['probability'].max().to_dict()
    s1_preds_099_count = preds_099_df.groupby('entity_id_s1').size().to_dict()

    s1_max_name_jw = preds_099_df.groupby('entity_id_s1')['name_jaro_winkler'].max().to_dict()
    s1_max_addr_jacc = preds_099_df.groupby('entity_id_s1')['address_token_jaccard'].max().to_dict()
    s1_max_keys = preds_099_df.groupby('entity_id_s1')['actual_num_keys'].max().to_dict()

    exp_d_groups = {}
    for grp_label, grp_ids in [
        ("GT Zero-Match (N=94)", zero_match_s1_ids),
        ("GT Singleton (N=134)", singleton_s1_ids),
        ("GT Multi-Match (N=272)", multi_match_s1_ids)
    ]:
        probs_grp = [s1_max_prob.get(s1, 0.0) for s1 in grp_ids]
        preds_grp = [s1_preds_099_count.get(s1, 0) for s1 in grp_ids]
        name_jw_grp = [s1_max_name_jw.get(s1, 0.0) for s1 in grp_ids]
        addr_jacc_grp = [s1_max_addr_jacc.get(s1, 0.0) for s1 in grp_ids]
        keys_grp = [s1_max_keys.get(s1, 0) for s1 in grp_ids]

        def get_dist(arr):
            return {
                "mean": round(float(np.mean(arr)), 4),
                "median": round(float(np.median(arr)), 4),
                "min": round(float(np.min(arr)), 4),
                "max": round(float(np.max(arr)), 4),
                "p25": round(float(np.percentile(arr, 25)), 4),
                "p75": round(float(np.percentile(arr, 75)), 4),
                "p90": round(float(np.percentile(arr, 90)), 4)
            }

        exp_d_groups[grp_label] = {
            "max_probability": get_dist(probs_grp),
            "predictions_count_099": get_dist(preds_grp),
            "max_name_jaro_winkler": get_dist(name_jw_grp),
            "max_address_token_jaccard": get_dist(addr_jacc_grp),
            "max_shared_blocking_keys": get_dist(keys_grp)
        }
        print(f"\n{grp_label}:")
        print(f"  Max Probability: Mean={np.mean(probs_grp):.4f}, Median={np.median(probs_grp):.4f}, Min={np.min(probs_grp):.4f}, Max={np.max(probs_grp):.4f}")
        print(f"  Preds >= 0.99:   Mean={np.mean(preds_grp):.2f}, Median={np.median(preds_grp):.2f}, Min={np.min(preds_grp)}, Max={np.max(preds_grp)}")
        print(f"  Max Name JW:     Mean={np.mean(name_jw_grp):.4f}, Median={np.median(name_jw_grp):.4f}")
        print(f"  Max Addr Jacc:   Mean={np.mean(addr_jacc_grp):.4f}, Median={np.median(addr_jacc_grp):.4f}")
        print(f"  Max Keys:        Mean={np.mean(keys_grp):.2f}, Median={np.median(keys_grp):.2f}")

    # =========================================================================
    # MASTER COMPARISON TABLE
    # =========================================================================
    print("\n[Step 6/6] Generating Master Comparison Table...", flush=True)

    master_table = {}
    master_table["Baseline 0.99"] = evaluate_prediction_set(baseline_pairs)

    for k_name, res in exp_a_results.items():
        master_table[k_name] = res

    for gap_name, res in exp_b_results.items():
        master_table[gap_name] = res

    for c_name, res in exp_c_results.items():
        master_table[c_name] = res

    # Find the rule with highest Macro F0.5
    best_rule = None
    best_f05 = -1.0
    for r_name, r_data in master_table.items():
        if r_data['macro_f0_5'] > best_f05:
            best_f05 = r_data['macro_f0_5']
            best_rule = r_name

    print("\n" + "=" * 85, flush=True)
    print(f"BEST ADAPTIVE RULE: {best_rule} with Macro F0.5 = {best_f05:.6f}", flush=True)
    print(f"Baseline 0.99 Macro F0.5: {master_table['Baseline 0.99']['macro_f0_5']:.6f} (Delta: {best_f05 - master_table['Baseline 0.99']['macro_f0_5']:+.6f})", flush=True)
    print("=" * 85, flush=True)

    output_data = {
        "benchmark_summary": {
            "s1_count": 500,
            "target_count": len(targets),
            "config_fingerprint": get_config_fingerprint(),
            "reference_true_pairs": len(exact_matches_set),
            "zero_match_s1_count": len(zero_match_s1_ids),
            "singleton_s1_count": len(singleton_s1_ids),
            "multi_match_s1_count": len(multi_match_s1_ids)
        },
        "best_rule_summary": {
            "best_rule": best_rule,
            "best_macro_f0_5": best_f05,
            "baseline_macro_f0_5": master_table['Baseline 0.99']['macro_f0_5'],
            "delta_f0_5": round(best_f05 - master_table['Baseline 0.99']['macro_f0_5'], 6),
            "relative_f0_5_improvement_pct": round(((best_f05 - master_table['Baseline 0.99']['macro_f0_5']) / master_table['Baseline 0.99']['macro_f0_5']) * 100, 2),
            "metrics": master_table[best_rule]
        },
        "master_comparison_table": master_table,
        "experiment_a_top_k": exp_a_results,
        "experiment_b_score_gap": exp_b_results,
        "experiment_c_evidence_aware": exp_c_results,
        "experiment_d_zero_match_safety": exp_d_groups
    }

    out_json = 'experiments/phase10_adaptive_selection.json'
    with open(out_json, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"Saved complete experiment results to {out_json}", flush=True)

    return output_data


if __name__ == '__main__':
    run_phase10()
