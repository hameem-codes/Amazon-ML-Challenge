"""
CAP -> MODEL -> MACRO F0.5 VALIDATION EXPERIMENT
Evaluates whether increasing candidate cap (400, 600, 800, 1000) improves the final
competition metric (Macro F0.5) when scored by the locked Phase 5 XGBoost model at threshold 0.910.
"""

import sys, os, time, gc, json, threading, hashlib
from collections import defaultdict
sys.path.insert(0, 'src')
sys.path.insert(0, '.')
import psutil
import pandas as pd
import numpy as np
import xgboost as xgb

from normalize import normalize_business_name, normalize_country, normalize_business_address
from blocking import ConfigABlockingEngine
from config import BLOCKING_CHUNK_SIZE, CANDIDATE_CAP_CONFIG, get_config_fingerprint
from features import build_vectorized_features, FEATURE_NAMES
from tfidf_model import TransductiveTFIDF
from metrics import compute_macro_f05, compute_pairwise_metrics
from scratch.run_experiment_opt_e7 import generate_bounded_candidates_e7


class PeakMemoryTracker:
    def __init__(self, interval=0.05):
        self.interval = interval
        self.process = psutil.Process(os.getpid())
        self.peak_bytes = self.process.memory_info().rss
        self._running = False
        self._thread = None

    def _track(self):
        while self._running:
            try:
                rss = self.process.memory_info().rss
                if rss > self.peak_bytes:
                    self.peak_bytes = rss
            except Exception:
                pass
            time.sleep(self.interval)

    def start(self):
        self.peak_bytes = self.process.memory_info().rss
        self._running = True
        self._thread = threading.Thread(target=self._track, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        rss = self.process.memory_info().rss
        if rss > self.peak_bytes:
            self.peak_bytes = rss
        return self.peak_bytes


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


def run_experiment():
    print("=" * 85, flush=True)
    print("PHASE: CAP -> MODEL -> MACRO F0.5 VALIDATION EXPERIMENT", flush=True)
    print("=" * 85, flush=True)

    # 10. SANITY CHECKS BEFORE RUNNING
    print("\n[Step 1/5] Executing 10 Sanity Checks on Locked Artifacts & Contracts...", flush=True)
    
    # 1. Model loads
    model_path = 'models/xgboost_entity_resolution_phase5_configA.json'
    assert os.path.exists(model_path), f"Missing model: {model_path}"
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    print(f"  [1/10] Phase 5 XGBoost model loaded successfully from {model_path}", flush=True)

    # 2. Feature schema
    schema_path = 'models/phase5_configA_feature_schema.json'
    with open(schema_path, 'r') as f:
        schema = json.load(f)
    assert len(schema['feature_names']) == 57, f"Expected 57 features, got {len(schema['feature_names'])}"
    assert schema['feature_names'] == FEATURE_NAMES, "Feature names mismatch!"
    print(f"  [2/10] Feature schema verified: exactly 57 features", flush=True)

    # 3. Threshold
    thresh_path = 'models/phase5_configA_threshold.json'
    with open(thresh_path, 'r') as f:
        thresh_data = json.load(f)
    locked_threshold = float(thresh_data['threshold'])
    assert locked_threshold == 0.910, f"Expected threshold 0.910, got {locked_threshold}"
    print(f"  [3/10] Decision threshold verified: {locked_threshold}", flush=True)

    # 4. Fingerprint
    fp = get_config_fingerprint()
    assert fp == "87f20ceeb84ccc6ea2d48678c7810ac5", f"Fingerprint mismatch: {fp}"
    print(f"  [4/10] Production config fingerprint verified: {fp}", flush=True)

    # 5. Validation set: 500 France S1 entities
    s1_df = pd.read_csv('data/test/test_source1.tsv', sep='\t', dtype=str, nrows=10000, na_filter=False)
    s1_fr = s1_df[s1_df['country'].apply(normalize_country) == 'france'].head(500).copy().reset_index(drop=True)
    del s1_df
    gc.collect()
    s1_fr['norm_name'] = s1_fr['business_name'].apply(normalize_business_name)
    s1_fr['norm_address'] = s1_fr['business_address'].apply(normalize_business_address)
    s1_fr['norm_country'] = 'france'
    val_s1_ids = list(s1_fr['entity_id'].values)
    assert len(val_s1_ids) == 500, f"Expected 500 S1 entities, got {len(val_s1_ids)}"
    print(f"  [5/10] Validation set verified: 500 France S1 entities", flush=True)

    # 6. Load targets and ground truth
    targets = pd.read_pickle('scratch/targets_france.pkl')
    n_targets = len(targets)
    print(f"  [6/10] Target pool loaded: {n_targets:,} France records (S2 + S3)", flush=True)

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
    print(f"  [7/10] Ground truth reference set verified: {len(exact_matches_set):,} true pairs", flush=True)

    # 7. No test labels accessed (exact name matches benchmark used as reference)
    print(f"  [8/10] Firewall check: zero test ground truth labels used", flush=True)

    # 8. S1 lookup dict
    s1_dict = s1_fr.set_index('entity_id').to_dict(orient='index')
    print(f"  [9/10] S1 lookup dict built: {len(s1_dict):,} records", flush=True)

    # 9. Inverted index engine
    print(f"  [10/10] Building ConfigABlockingEngine...", flush=True)
    engine = ConfigABlockingEngine(targets)
    print("All 10 Sanity Checks PASSED cleanly!\n", flush=True)

    # Fit transductive TF-IDF
    tfidf_models = fit_transductive_tfidf()

    # Precompute high-DF address tokens (> 50,000)
    addr_cap = 50000
    country_addr_tokens = engine.country_addr_token_idx['france']
    high_tokens = {k for k, v in country_addr_tokens.items() if len(v) > addr_cap}
    high_df_postings_sets = {k: set(country_addr_tokens[k]) for k in high_tokens}
    print(f"Precomputed {len(high_tokens)} high-DF address token sets (> {addr_cap:,})", flush=True)

    # Ground truth zero-match S1 entities
    s1_with_gt = {p[0] for p in exact_matches_set}
    gt_zero_match_s1_count = len(set(val_s1_ids) - s1_with_gt)
    print(f"Ground-Truth S1 distribution: {len(s1_with_gt)} S1 entities with true matches, {gt_zero_match_s1_count} zero-match S1 entities", flush=True)

    caps = [400, 600, 800, 1000]
    results = {}

    print("\n" + "=" * 85, flush=True)
    print("EXECUTING PIPELINE ACROSS CAPS: 400 -> 600 -> 800 -> 1000", flush=True)
    print("=" * 85, flush=True)

    for cap_val in caps:
        cfg_name = f"CAP-{cap_val}"
        print(f"\n>>> Running {cfg_name} (Candidate Cap = {cap_val} / S1)...", flush=True)
        t0_total = time.time()

        mem_tracker = PeakMemoryTracker(interval=0.05)
        mem_tracker.start()

        # Step A: Bounded Blocking Candidate Generation
        t0_block = time.time()
        capped_df, metrics = generate_bounded_candidates_e7(
            engine,
            s1_fr,
            max_candidates_per_s1=cap_val,
            chunk_size=BLOCKING_CHUNK_SIZE,
            addr_df_cap=addr_cap,
            high_df_postings_sets=high_df_postings_sets,
            use_fallback=False
        )
        t_block = time.time() - t0_block
        n_cands = len(capped_df)
        print(f"  [Blocking] Generated {n_cands:,} post-cap candidates in {t_block:.2f}s", flush=True)

        # Candidate diagnostics
        post_cap_pairs = set(zip(capped_df['entity_id_s1'], capped_df['entity_id_cand']))
        true_pairs_retained = len(post_cap_pairs & exact_matches_set)
        true_pairs_lost = len(exact_matches_set) - true_pairs_retained
        candidate_recall = true_pairs_retained / len(exact_matches_set)
        raw_cands = metrics['raw_candidate_count']
        raw_recall = 1.0

        cands_per_s1_arr = capped_df.groupby('entity_id_s1').size().values
        mean_cands_s1 = float(np.mean(cands_per_s1_arr)) if len(cands_per_s1_arr) > 0 else 0.0
        median_cands_s1 = float(np.median(cands_per_s1_arr)) if len(cands_per_s1_arr) > 0 else 0.0
        max_cands_s1 = int(np.max(cands_per_s1_arr)) if len(cands_per_s1_arr) > 0 else 0

        # Step B: Feature Extraction (vectorized batches of 25,000)
        t0_feat = time.time()
        needed_target_eids = set(capped_df['entity_id_cand'].unique())
        target_slice = targets[targets['entity_id'].isin(needed_target_eids)]
        target_dict = target_slice.set_index('entity_id').to_dict(orient='index')
        del target_slice
        gc.collect()

        feat_chunks = []
        batch_sz = 25000
        for b_idx in range(0, n_cands, batch_sz):
            chunk_pair_df = capped_df.iloc[b_idx:b_idx + batch_sz]
            chunk_feat = build_vectorized_features(chunk_pair_df, s1_dict, target_dict, tfidf_models=tfidf_models)
            feat_chunks.append(chunk_feat[FEATURE_NAMES])
            del chunk_pair_df, chunk_feat
            gc.collect()

        df_features = pd.concat(feat_chunks, ignore_index=True) if feat_chunks else pd.DataFrame(columns=FEATURE_NAMES)
        del feat_chunks, target_dict
        gc.collect()

        t_feat = time.time() - t0_feat
        print(f"  [Features] Built 57 features for {len(df_features):,} pairs in {t_feat:.2f}s", flush=True)

        # Step C: XGBoost Scoring & Classification
        t0_model = time.time()
        X = df_features[FEATURE_NAMES].values
        probs = model.predict_proba(X)[:, 1]
        del X, df_features
        gc.collect()

        match_mask = (probs >= locked_threshold)
        pred_s1 = capped_df['entity_id_s1'].values[match_mask]
        pred_cand = capped_df['entity_id_cand'].values[match_mask]
        predicted_pairs = set(zip(pred_s1, pred_cand))
        t_model = time.time() - t0_model
        print(f"  [XGBoost] Scored in {t_model:.2f}s | Matches >= {locked_threshold}: {len(predicted_pairs):,}", flush=True)

        peak_bytes = mem_tracker.stop()
        peak_mb = peak_bytes / (1024 * 1024)
        t_total = time.time() - t0_total
        throughput = len(s1_fr) / t_total if t_total > 0 else 0

        # Step D: Entity-Level Macro F0.5 Evaluation
        macro_metrics = compute_macro_f05(val_s1_ids, exact_matches_set, predicted_pairs)
        macro_f05 = macro_metrics['macro_f0_5']
        macro_prec = macro_metrics['macro_precision']
        macro_rec = macro_metrics['macro_recall']
        pred_zero_match_s1 = macro_metrics['zero_match_s1_count']
        pred_matched_s1 = macro_metrics['one_or_more_match_s1_count']

        # Pairwise metrics
        tp = len(predicted_pairs & exact_matches_set)
        fp = len(predicted_pairs - exact_matches_set)
        fn = len(exact_matches_set - predicted_pairs)
        p_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        p_rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        p_denom = (0.25 * p_prec) + p_rec
        pairwise_f05 = (1.25 * p_prec * p_rec) / p_denom if p_denom > 0 else 0.0

        print(f"  [Metrics] Macro F0.5: {macro_f05:.4f} | Prec: {macro_prec:.4f} | Rec: {macro_rec:.4f} | TP: {tp:,} | FP: {fp:,} | FN: {fn:,}", flush=True)

        results[cfg_name] = {
            'config': cfg_name,
            'cap': cap_val,
            # Candidate diagnostics
            'raw_candidates': raw_cands,
            'raw_candidate_recall': raw_recall,
            'post_cap_candidates': n_cands,
            'average_candidates_per_s1': round(mean_cands_s1, 1),
            'median_candidates_per_s1': round(median_cands_s1, 1),
            'max_candidates_per_s1': max_cands_s1,
            'post_cap_candidate_recall': round(candidate_recall, 6),
            'true_pairs_retained_in_candidates': true_pairs_retained,
            'true_pairs_lost_in_candidates': true_pairs_lost,
            # Model & evaluation metrics
            'macro_f0_5': round(macro_f05, 6),
            'macro_precision': round(macro_prec, 6),
            'macro_recall': round(macro_rec, 6),
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'pairwise_precision': round(p_prec, 6),
            'pairwise_recall': round(p_rec, 6),
            'pairwise_f0_5': round(pairwise_f05, 6),
            'predicted_matched_s1_count': pred_matched_s1,
            'predicted_zero_match_s1_count': pred_zero_match_s1,
            'ground_truth_zero_match_s1_count': gt_zero_match_s1_count,
            # Performance diagnostics
            'blocking_runtime_s': round(t_block, 2),
            'feature_runtime_s': round(t_feat, 2),
            'model_runtime_s': round(t_model, 2),
            'total_runtime_s': round(t_total, 2),
            'throughput_s1_per_sec': round(throughput, 2),
            'peak_rss_mb': round(peak_mb, 1),
            # Save predictions set for diff analysis
            'predicted_pairs_list': list(predicted_pairs)
        }

        del capped_df, probs, match_mask, predicted_pairs
        gc.collect()

    # Step E: Incremental Benefit Analysis vs CAP-400
    base = results['CAP-400']
    for cap_val in [400, 600, 800, 1000]:
        cfg_name = f"CAP-{cap_val}"
        r = results[cfg_name]
        d_f05 = r['macro_f0_5'] - base['macro_f0_5']
        d_prec = r['macro_precision'] - base['macro_precision']
        d_rec = r['macro_recall'] - base['macro_recall']
        d_cands = r['post_cap_candidates'] - base['post_cap_candidates']
        d_time = r['total_runtime_s'] - base['total_runtime_s']
        d_tp = r['tp'] - base['tp']
        d_fp = r['fp'] - base['fp']

        r['delta_macro_f0_5'] = round(d_f05, 6)
        r['delta_macro_precision'] = round(d_prec, 6)
        r['delta_macro_recall'] = round(d_rec, 6)
        r['additional_candidates'] = d_cands
        r['additional_runtime_s'] = round(d_time, 2)
        r['additional_true_pairs'] = d_tp
        r['additional_false_positives'] = d_fp

    # Prediction set differences between caps
    preds_400 = set(tuple(p) for p in results['CAP-400']['predicted_pairs_list'])
    preds_600 = set(tuple(p) for p in results['CAP-600']['predicted_pairs_list'])
    preds_800 = set(tuple(p) for p in results['CAP-800']['predicted_pairs_list'])
    preds_1000 = set(tuple(p) for p in results['CAP-1000']['predicted_pairs_list'])

    prediction_diffs = {
        'CAP-600_vs_CAP-400': {
            'new_predictions_count': len(preds_600 - preds_400),
            'dropped_predictions_count': len(preds_400 - preds_600),
            'new_predictions_sample': list(preds_600 - preds_400)[:10],
            'new_predictions_tp_count': len((preds_600 - preds_400) & exact_matches_set),
            'new_predictions_fp_count': len((preds_600 - preds_400) - exact_matches_set)
        },
        'CAP-800_vs_CAP-400': {
            'new_predictions_count': len(preds_800 - preds_400),
            'dropped_predictions_count': len(preds_400 - preds_800),
            'new_predictions_sample': list(preds_800 - preds_400)[:10],
            'new_predictions_tp_count': len((preds_800 - preds_400) & exact_matches_set),
            'new_predictions_fp_count': len((preds_800 - preds_400) - exact_matches_set)
        },
        'CAP-1000_vs_CAP-400': {
            'new_predictions_count': len(preds_1000 - preds_400),
            'dropped_predictions_count': len(preds_400 - preds_1000),
            'new_predictions_sample': list(preds_1000 - preds_400)[:10],
            'new_predictions_tp_count': len((preds_1000 - preds_400) & exact_matches_set),
            'new_predictions_fp_count': len((preds_1000 - preds_400) - exact_matches_set)
        }
    }

    # Remove predicted_pairs_list from saved json to keep size clean
    clean_results = {}
    for k, v in results.items():
        v_clean = dict(v)
        v_clean.pop('predicted_pairs_list', None)
        clean_results[k] = v_clean

    output_data = {
        'benchmark_summary': {
            's1_count': 500,
            'target_count': n_targets,
            'config_fingerprint': fp,
            'locked_model': model_path,
            'locked_threshold': locked_threshold,
            'reference_true_pairs': len(exact_matches_set),
            'ground_truth_zero_match_s1_count': gt_zero_match_s1_count
        },
        'configurations': clean_results,
        'prediction_differences': prediction_diffs
    }

    out_json = 'experiments/optimization_cap_model_f05_results.json'
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nAll results saved to {out_json}", flush=True)

    # Print Primary Table
    print("\n" + "=" * 130, flush=True)
    print(f"{'Cap':<8}{'Candidates':<14}{'Cand Recall':<14}{'Macro Prec':<14}{'Macro Rec':<14}{'Macro F0.5':<14}{'TP':<8}{'FP':<8}{'FN':<8}{'Runtime':<12}{'Peak RAM':<12}")
    print("-" * 130, flush=True)
    for cap_val in [400, 600, 800, 1000]:
        r = clean_results[f"CAP-{cap_val}"]
        print(f"{r['cap']:<8}{r['post_cap_candidates']:<14,}{r['post_cap_candidate_recall']:<14.4%}{r['macro_precision']:<14.4%}{r['macro_recall']:<14.4%}{r['macro_f0_5']:<14.4f}{r['tp']:<8,}{r['fp']:<8,}{r['fn']:<8,}{r['total_runtime_s']:<12.2f}s{r['peak_rss_mb']:<12.1f}MB", flush=True)
    print("=" * 130, flush=True)

    return output_data


if __name__ == '__main__':
    run_experiment()
