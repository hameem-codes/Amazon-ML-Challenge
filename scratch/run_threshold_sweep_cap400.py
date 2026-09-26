"""
PHASE 8: THRESHOLD SWEEP FOR MACRO F0.5 @ FIXED CAP 400

Sweeps decision thresholds from 0.50 to 0.99 (50 thresholds) on the locked CAP-400 pipeline
(Config A blocking, legacy evidence ranker, 57 features, Phase 5 XGBoost)
on the 500-S1 France benchmark to find the threshold that maximizes Macro F0.5.
"""

import sys, os, time, gc, json, math
from collections import defaultdict, Counter
sys.path.insert(0, 'src')
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
import xgboost as xgb

from normalize import normalize_business_name, normalize_country, normalize_business_address
from blocking import ConfigABlockingEngine
from config import BLOCKING_CHUNK_SIZE, get_config_fingerprint
from features import build_vectorized_features, FEATURE_NAMES
from tfidf_model import TransductiveTFIDF
from metrics import compute_macro_f05, compute_pairwise_metrics
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


def run_threshold_sweep():
    print("=" * 85, flush=True)
    print("PHASE 8: THRESHOLD SWEEP FOR MACRO F0.5 @ LOCKED CAP-400", flush=True)
    print("=" * 85, flush=True)

    # 1. Sanity Checks
    print("\n[Step 1/4] Verifying Locked Artifacts & Environment...", flush=True)
    model_path = 'models/xgboost_entity_resolution_phase5_configA.json'
    assert os.path.exists(model_path), f"Missing model: {model_path}"
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    print(f"  [1/8] Phase 5 XGBoost model loaded from {model_path}", flush=True)

    schema_path = 'models/phase5_configA_feature_schema.json'
    with open(schema_path, 'r') as f:
        schema = json.load(f)
    assert len(schema['feature_names']) == 57, f"Expected 57 features, got {len(schema['feature_names'])}"
    assert schema['feature_names'] == FEATURE_NAMES, "Feature schema mismatch!"
    print(f"  [2/8] Feature schema verified: exactly 57 features", flush=True)

    fp = get_config_fingerprint()
    assert fp == "87f20ceeb84ccc6ea2d48678c7810ac5", f"Fingerprint mismatch: {fp}"
    print(f"  [3/8] Production config fingerprint verified: {fp}", flush=True)

    s1_df = pd.read_csv('data/test/test_source1.tsv', sep='\t', dtype=str, nrows=10000, na_filter=False)
    s1_fr = s1_df[s1_df['country'].apply(normalize_country) == 'france'].head(500).copy().reset_index(drop=True)
    del s1_df
    gc.collect()
    s1_fr['norm_name'] = s1_fr['business_name'].apply(normalize_business_name)
    s1_fr['norm_address'] = s1_fr['business_address'].apply(normalize_business_address)
    s1_fr['norm_country'] = 'france'
    val_s1_ids = list(s1_fr['entity_id'].values)
    assert len(val_s1_ids) == 500, f"Expected 500 S1 entities, got {len(val_s1_ids)}"
    print(f"  [4/8] Validation set verified: 500 France S1 entities", flush=True)

    targets = pd.read_pickle('scratch/targets_france.pkl')
    n_targets = len(targets)
    print(f"  [5/8] Target universe loaded: {n_targets:,} France records (S2 + S3)", flush=True)

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
    print(f"  [6/8] Ground truth reference pairs verified: {len(exact_matches_set):,} true pairs", flush=True)
    assert len(exact_matches_set) == 4191, f"Expected 4191 true pairs, got {len(exact_matches_set)}"

    s1_with_gt = {p[0] for p in exact_matches_set}
    gt_zero_match_s1_count = len(set(val_s1_ids) - s1_with_gt)
    print(f"  [7/8] Ground truth distribution: {len(s1_with_gt)} matched S1, {gt_zero_match_s1_count} zero-match S1", flush=True)

    s1_dict = s1_fr.set_index('entity_id').to_dict(orient='index')
    engine = ConfigABlockingEngine(targets)
    print(f"  [8/8] ConfigABlockingEngine initialized cleanly", flush=True)

    # 2. Get baseline CAP-400 candidates and model probabilities
    cache_file = 'scratch/cap400_scored_candidates.pkl'
    if os.path.exists(cache_file):
        print(f"\n[Step 2/4] Loading cached CAP-400 candidate probabilities from {cache_file}...", flush=True)
        cand_prob_df = pd.read_pickle(cache_file)
        print(f"  Loaded {len(cand_prob_df):,} scored candidate pairs", flush=True)
    else:
        print("\n[Step 2/4] Generating Baseline CAP-400 candidates & scoring with XGBoost...", flush=True)
        tfidf_models = fit_transductive_tfidf()

        addr_cap = 50000
        country_addr_tokens = engine.country_addr_token_idx['france']
        high_tokens = {k for k, v in country_addr_tokens.items() if len(v) > addr_cap}
        high_df_postings_sets = {k: set(country_addr_tokens[k]) for k in high_tokens}

        t0_gen = time.time()
        capped_df, metrics = generate_bounded_candidates_e7(
            engine,
            s1_fr,
            max_candidates_per_s1=400,
            chunk_size=BLOCKING_CHUNK_SIZE,
            addr_df_cap=addr_cap,
            high_df_postings_sets=high_df_postings_sets,
            use_fallback=False
        )
        t_gen = time.time() - t0_gen
        print(f"  Generated {len(capped_df):,} candidates in {t_gen:.2f}s", flush=True)

        needed_target_eids = set(capped_df['entity_id_cand'].unique())
        target_slice = targets[targets['entity_id'].isin(needed_target_eids)]
        target_dict = target_slice.set_index('entity_id').to_dict(orient='index')
        del target_slice
        gc.collect()

        t0_feat = time.time()
        feat_chunks = []
        batch_sz = 25000
        for b_idx in range(0, len(capped_df), batch_sz):
            chunk_pair_df = capped_df.iloc[b_idx:b_idx + batch_sz]
            chunk_feat = build_vectorized_features(chunk_pair_df, s1_dict, target_dict, tfidf_models=tfidf_models)
            feat_chunks.append(chunk_feat[FEATURE_NAMES])
            del chunk_pair_df, chunk_feat
            gc.collect()

        df_features = pd.concat(feat_chunks, ignore_index=True)
        del feat_chunks, target_dict
        gc.collect()
        t_feat = time.time() - t0_feat
        print(f"  Extracted 57 features in {t_feat:.2f}s", flush=True)

        t0_model = time.time()
        probs = model.predict_proba(df_features[FEATURE_NAMES].values)[:, 1]
        t_model = time.time() - t0_model
        print(f"  Scored {len(probs):,} pairs with XGBoost in {t_model:.2f}s", flush=True)

        cand_prob_df = pd.DataFrame({
            'entity_id_s1': capped_df['entity_id_s1'].values,
            'entity_id_cand': capped_df['entity_id_cand'].values,
            'probability': probs.astype(np.float32)
        })
        pd.to_pickle(cand_prob_df, cache_file)
        print(f"  Cached scored candidate probabilities to {cache_file}", flush=True)
        del capped_df, df_features
        gc.collect()

    # 3. Probability Distribution Summary / Histogram
    all_probs = cand_prob_df['probability'].values
    print("\n[Step 3/4] Probability Distribution Summary (200,000 candidate pairs):", flush=True)
    pctiles = [0, 10, 25, 50, 75, 90, 95, 99, 99.5, 99.9, 100]
    pctile_vals = np.percentile(all_probs, pctiles)
    for p, val in zip(pctiles, pctile_vals):
        print(f"  p{p:4.1f}%: {val:.6f}", flush=True)

    prob_hist = {}
    bin_edges = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
    hist_counts, _ = np.histogram(all_probs, bins=bin_edges)
    for i in range(len(hist_counts)):
        label = f"[{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f})" if i < len(hist_counts)-1 else f"[{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f}]"
        prob_hist[label] = int(hist_counts[i])
        print(f"  Bin {label}: {hist_counts[i]:,} candidates ({hist_counts[i]/len(all_probs)*100:.2f}%)", flush=True)

    # 4. Threshold Sweep (0.50 to 0.99, exactly 50 thresholds)
    print("\n[Step 4/4] Executing Threshold Sweep across 50 Thresholds (0.50 -> 0.99)...", flush=True)
    thresholds = [round(0.50 + i * 0.01, 2) for i in range(50)]
    assert len(thresholds) == 50, f"Expected 50 thresholds, got {len(thresholds)}"

    s1_arr = cand_prob_df['entity_id_s1'].values
    cand_arr = cand_prob_df['entity_id_cand'].values
    probs_arr = cand_prob_df['probability'].values

    # Pre-index true pairs by S1 for fast evaluation
    s1_set_all = set(val_s1_ids)

    sweep_results = []
    best_threshold = None
    best_f05 = -1.0

    ref_threshold = 0.91
    ref_result = None

    for thr in thresholds:
        mask = (probs_arr >= thr)
        n_pred = int(np.sum(mask))

        if n_pred > 0:
            pred_s1 = s1_arr[mask]
            pred_cand = cand_arr[mask]
            pred_pairs = set(zip(pred_s1, pred_cand))
            s1_counts = Counter(pred_s1)
        else:
            pred_pairs = set()
            s1_counts = Counter()

        # Compute Macro F0.5
        macro_res = compute_macro_f05(val_s1_ids, exact_matches_set, pred_pairs)
        macro_f05 = macro_res['macro_f0_5']
        macro_prec = macro_res['macro_precision']
        macro_rec = macro_res['macro_recall']
        pred_matched_s1 = macro_res['one_or_more_match_s1_count']
        pred_zero_match_s1 = macro_res['zero_match_s1_count']

        # Pairwise metrics
        tp = len(pred_pairs & exact_matches_set)
        fp = len(pred_pairs - exact_matches_set)
        fn = len(exact_matches_set - pred_pairs)
        p_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        p_rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # Prediction-count-per-S1 distribution across ALL 500 S1 entities
        per_s1_preds = [s1_counts[s1_id] for s1_id in val_s1_ids]
        mean_p = float(np.mean(per_s1_preds))
        median_p = float(np.median(per_s1_preds))
        min_p = int(np.min(per_s1_preds))
        max_p = int(np.max(per_s1_preds))
        p25_p = float(np.percentile(per_s1_preds, 25))
        p75_p = float(np.percentile(per_s1_preds, 75))
        p90_p = float(np.percentile(per_s1_preds, 90))
        p95_p = float(np.percentile(per_s1_preds, 95))
        p99_p = float(np.percentile(per_s1_preds, 99))

        item = {
            "threshold": thr,
            "macro_f0_5": round(macro_f05, 6),
            "macro_precision": round(macro_prec, 6),
            "macro_recall": round(macro_rec, 6),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "pairwise_precision": round(p_prec, 6),
            "pairwise_recall": round(p_rec, 6),
            "total_predicted_matches": n_pred,
            "total_correct_matches": tp,
            "total_false_matches": fp,
            "total_missed_true_matches": fn,
            "predicted_matched_s1_count": pred_matched_s1,
            "predicted_zero_match_s1_count": pred_zero_match_s1,
            "ground_truth_zero_match_s1_count": gt_zero_match_s1_count,
            "predictions_per_s1": {
                "mean": round(mean_p, 2),
                "median": round(median_p, 2),
                "min": min_p,
                "max": max_p,
                "p25": round(p25_p, 2),
                "p75": round(p75_p, 2),
                "p90": round(p90_p, 2),
                "p95": round(p95_p, 2),
                "p99": round(p99_p, 2)
            }
        }
        sweep_results.append(item)

        if macro_f05 > best_f05:
            best_f05 = macro_f05
            best_threshold = thr

        if thr == ref_threshold:
            ref_result = item

        print(f"Thr: {thr:.2f} | F0.5: {macro_f05:.6f} | Prec: {macro_prec*100:6.3f}% | Rec: {macro_rec*100:6.3f}% | TP: {tp:5d} | FP: {fp:5d} | FN: {fn:5d} | Matches S1: {pred_matched_s1:3d} | Zero S1: {pred_zero_match_s1:3d}", flush=True)

    # Check for ties with best threshold
    tied_thresholds = [item for item in sweep_results if item['macro_f0_5'] == best_f05]
    best_item = [item for item in sweep_results if item['threshold'] == best_threshold][0]

    print("\n" + "=" * 85, flush=True)
    print(f"THRESHOLD SWEEP SUMMARY: Best Threshold = {best_threshold:.2f} (Macro F0.5 = {best_f05:.6f})", flush=True)
    print(f"Baseline Reference (0.910): Macro F0.5 = {ref_result['macro_f0_5']:.6f}", flush=True)
    print("=" * 85, flush=True)

    delta_best_vs_ref = {
        "delta_macro_f0_5": round(best_item['macro_f0_5'] - ref_result['macro_f0_5'], 6),
        "delta_macro_precision": round(best_item['macro_precision'] - ref_result['macro_precision'], 6),
        "delta_macro_recall": round(best_item['macro_recall'] - ref_result['macro_recall'], 6),
        "delta_tp": best_item['tp'] - ref_result['tp'],
        "delta_fp": best_item['fp'] - ref_result['fp'],
        "delta_fn": best_item['fn'] - ref_result['fn'],
        "delta_zero_match_s1": best_item['predicted_zero_match_s1_count'] - ref_result['predicted_zero_match_s1_count'],
        "relative_f0_5_change_pct": round(((best_item['macro_f0_5'] - ref_result['macro_f0_5']) / ref_result['macro_f0_5']) * 100, 2)
    }

    results_data = {
        "benchmark_summary": {
            "s1_count": 500,
            "target_count": n_targets,
            "config_fingerprint": fp,
            "locked_model": model_path,
            "reference_threshold": ref_threshold,
            "reference_true_pairs": len(exact_matches_set),
            "ground_truth_zero_match_s1_count": gt_zero_match_s1_count,
            "candidate_cap": 400
        },
        "best_threshold_selection": {
            "best_threshold": best_threshold,
            "macro_f0_5": best_item['macro_f0_5'],
            "macro_precision": best_item['macro_precision'],
            "macro_recall": best_item['macro_recall'],
            "tp": best_item['tp'],
            "fp": best_item['fp'],
            "fn": best_item['fn'],
            "predicted_matched_s1_count": best_item['predicted_matched_s1_count'],
            "predicted_zero_match_s1_count": best_item['predicted_zero_match_s1_count'],
            "tied_thresholds_count": len(tied_thresholds),
            "tied_thresholds_list": [t['threshold'] for t in tied_thresholds]
        },
        "comparison_against_ref_0910": {
            "reference_threshold": 0.91,
            "ref_macro_f0_5": ref_result['macro_f0_5'],
            "ref_macro_precision": ref_result['macro_precision'],
            "ref_macro_recall": ref_result['macro_recall'],
            "ref_tp": ref_result['tp'],
            "ref_fp": ref_result['fp'],
            "ref_fn": ref_result['fn'],
            "delta": delta_best_vs_ref
        },
        "probability_distribution": {
            "percentiles": {f"p{p}": round(float(v), 6) for p, v in zip(pctiles, pctile_vals)},
            "histogram": prob_hist
        },
        "threshold_grid": sweep_results
    }

    out_json = 'experiments/threshold_sweep_cap400_results.json'
    with open(out_json, 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"Saved results to {out_json}", flush=True)

    return results_data


if __name__ == '__main__':
    run_threshold_sweep()
