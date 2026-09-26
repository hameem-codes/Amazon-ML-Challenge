"""
PHASE 12A: TOP-K + CONFIDENCE + SCORE-MARGIN DIAGNOSTIC

Evaluates a stricter post-model decision layer on the 500-France validation benchmark:
1. Top-K only (K in [1, 2, 3, 5], p >= 0.99)
2. Top-K + Absolute Confidence (K in [1, 2, 3, 5], tau_abs in [0.90, 0.92, 0.94, 0.96, 0.98, 0.99])
3. Top-K + Confidence + Margin (K in [1, 2, 3, 5], tau_abs in [0.90, 0.92, 0.94, 0.96, 0.98, 0.99], tau_gap in [0.00, 0.0001, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10])
4. Master comparison table vs Phase 10 baseline
5. Subgroup analysis (Zero-match, Singleton, Multi-match) for top 10 configurations
6. Selection analysis against precision/recall, multi-match retention, and safety criteria
"""

import sys, os, time, gc, json
from collections import defaultdict, Counter
sys.path.insert(0, 'src')
sys.path.insert(0, '.')
import pandas as pd
import numpy as np

from normalize import normalize_business_name, normalize_country, normalize_business_address
from config import get_config_fingerprint
from metrics import compute_macro_f05
from blocking import ConfigABlockingEngine


def run_phase12a():
    print("=" * 85, flush=True)
    print("PHASE 12A: TOP-K + CONFIDENCE + SCORE-MARGIN DIAGNOSTIC", flush=True)
    print("=" * 85, flush=True)

    fp = get_config_fingerprint()
    print(f"Production Fingerprint: {fp}", flush=True)
    assert fp == "87f20ceeb84ccc6ea2d48678c7810ac5", f"Fingerprint mismatch: {fp}"

    # 1. Setup validation benchmark & ground truth
    print("\n[Step 1/5] Loading validation benchmark & ground truth...", flush=True)
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

    gt_pairs_singleton = len({p for p in exact_matches_set if p[0] in set(singleton_s1_ids)})
    gt_pairs_multi = len({p for p in exact_matches_set if p[0] in set(multi_match_s1_ids)})

    print(f"Entities: Total={len(val_s1_ids)}, Zero-match={len(zero_match_s1_ids)}, Singleton={len(singleton_s1_ids)}, Multi-match={len(multi_match_s1_ids)}")
    print(f"Reference True Pairs: Total={len(exact_matches_set):,}, Singleton={gt_pairs_singleton}, Multi-match={gt_pairs_multi}")

    # 2. Load candidate probabilities
    print("\n[Step 2/5] Loading scored candidate pool (CAP-400)...", flush=True)
    cache_file = 'scratch/cap400_scored_candidates.pkl'
    assert os.path.exists(cache_file), f"Missing cache: {cache_file}"
    cand_prob_df = pd.read_pickle(cache_file)

    eid_to_lex = dict(zip(engine.target_eids, engine.target_lex_rank))
    cand_prob_df['target_lex_rank'] = [eid_to_lex[eid] for eid in cand_prob_df['entity_id_cand']]

    # Pre-organize candidates per S1 sorted by (-prob, lex_rank)
    cands_by_s1 = defaultdict(list)
    for row in cand_prob_df.itertuples():
        cands_by_s1[row.entity_id_s1].append((row.entity_id_cand, float(row.probability), int(row.target_lex_rank)))

    for s1 in cands_by_s1:
        cands_by_s1[s1].sort(key=lambda x: (-x[1], x[2]))

    # Phase 10 Baseline Verification
    print("\n[Step 3/5] Verifying Phase 10 Baseline (Gap <= 0.0001, p >= 0.99)...", flush=True)
    base_pairs = set()
    for s1_id in val_s1_ids:
        c_list = [c for c in cands_by_s1.get(s1_id, []) if c[1] >= 0.990]
        if not c_list:
            continue
        base_pairs.add((s1_id, c_list[0][0]))
        for idx in range(len(c_list) - 1):
            if (c_list[idx][1] - c_list[idx + 1][1]) <= 0.0001:
                base_pairs.add((s1_id, c_list[idx + 1][0]))
            else:
                break

    base_macro = compute_macro_f05(val_s1_ids, exact_matches_set, base_pairs)
    base_tp = len(base_pairs & exact_matches_set)
    base_fp = len(base_pairs - exact_matches_set)
    base_fn = len(exact_matches_set - base_pairs)
    base_p_prec = base_tp / (base_tp + base_fp) if (base_tp + base_fp) > 0 else 0.0
    base_p_rec = base_tp / (base_tp + base_fn) if (base_tp + base_fn) > 0 else 0.0

    base_singleton_tp = len({p for p in base_pairs if p[0] in set(singleton_s1_ids)} & exact_matches_set)
    base_multi_tp = len({p for p in base_pairs if p[0] in set(multi_match_s1_ids)} & exact_matches_set)

    print(f"Baseline F0.5: {base_macro['macro_f0_5']:.6f} | Prec: {base_macro['macro_precision']*100:.3f}% | Rec: {base_macro['macro_recall']*100:.3f}%")
    print(f"TP: {base_tp}, FP: {base_fp}, FN: {base_fn} | Sing TP: {base_singleton_tp}/{gt_pairs_singleton} | Multi TP: {base_multi_tp}/{gt_pairs_multi}")
    assert abs(base_macro['macro_f0_5'] - 0.317727) < 1e-4, f"Baseline mismatch: {base_macro['macro_f0_5']}"

    # Evaluation function for any prediction set
    def evaluate_predictions(rule_label, k_val, tau_abs_val, tau_gap_val, pred_pairs):
        tp = len(pred_pairs & exact_matches_set)
        fp = len(pred_pairs - exact_matches_set)
        fn = len(exact_matches_set - pred_pairs)
        p_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        p_rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        macro = compute_macro_f05(val_s1_ids, exact_matches_set, pred_pairs)

        pred_counts = Counter(p[0] for p in pred_pairs)
        counts_all = [pred_counts[s1] for s1 in val_s1_ids]
        mean_p = float(np.mean(counts_all))
        median_p = float(np.median(counts_all))

        zero_with_preds = sum(1 for s1 in zero_match_s1_ids if pred_counts[s1] > 0)
        zero_correct_empty = len(zero_match_s1_ids) - zero_with_preds

        sing_tp = len({p for p in pred_pairs if p[0] in set(singleton_s1_ids)} & exact_matches_set)
        multi_tp = len({p for p in pred_pairs if p[0] in set(multi_match_s1_ids)} & exact_matches_set)

        sing_ret_vs_base = (sing_tp / base_singleton_tp * 100) if base_singleton_tp > 0 else 0.0
        multi_ret_vs_base = (multi_tp / base_multi_tp * 100) if base_multi_tp > 0 else 0.0

        return {
            "rule": rule_label,
            "K": k_val,
            "tau_abs": tau_abs_val,
            "tau_gap": tau_gap_val,
            "macro_precision": round(macro['macro_precision'], 6),
            "macro_recall": round(macro['macro_recall'], 6),
            "macro_f0_5": round(macro['macro_f0_5'], 6),
            "pairwise_precision": round(p_prec, 6),
            "pairwise_recall": round(p_rec, 6),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "predicted_pairs": len(pred_pairs),
            "mean_predictions_per_s1": round(mean_p, 2),
            "median_predictions_per_s1": round(median_p, 2),
            "zero_match_receiving_predictions": zero_with_preds,
            "zero_match_correct_empty": zero_correct_empty,
            "singleton_tp": sing_tp,
            "singleton_tp_retention_pct": round(sing_ret_vs_base, 2),
            "multi_match_tp": multi_tp,
            "multi_match_tp_retention_pct": round(multi_ret_vs_base, 2),
        }

    # =========================================================================
    # STEP 1: TOP-K ONLY (p >= 0.99)
    # =========================================================================
    print("\n[Step 1] Running Step 1: Top-K Only (p >= 0.99)...", flush=True)
    step1_results = []
    for K in [1, 2, 3, 5]:
        pairs = set()
        for s1_id in val_s1_ids:
            cands = [c for c in cands_by_s1.get(s1_id, []) if c[1] >= 0.990]
            for c in cands[:K]:
                pairs.add((s1_id, c[0]))
        res = evaluate_predictions(f"Top-{K} (p>=0.99)", K, 0.99, 0.0, pairs)
        step1_results.append(res)
        print(f"  Top-{K:1d} (p>=0.99)     : F0.5={res['macro_f0_5']:.6f} | Prec={res['macro_precision']*100:6.3f}% | Rec={res['macro_recall']*100:6.3f}% | TP={res['tp']:5d} | FP={res['fp']:5d} | Preds={res['predicted_pairs']:5d} | SingRet={res['singleton_tp_retention_pct']:5.1f}% | MultiRet={res['multi_match_tp_retention_pct']:5.1f}%")

    # =========================================================================
    # STEP 2: TOP-K + ABSOLUTE CONFIDENCE
    # =========================================================================
    print("\n[Step 2] Running Step 2: Top-K + Absolute Confidence...", flush=True)
    k_list = [1, 2, 3, 5]
    tau_abs_list = [0.90, 0.92, 0.94, 0.96, 0.98, 0.99]
    step2_results = []

    for K in k_list:
        for tau_abs in tau_abs_list:
            pairs = set()
            for s1_id in val_s1_ids:
                cands = [c for c in cands_by_s1.get(s1_id, []) if c[1] >= tau_abs]
                for c in cands[:K]:
                    pairs.add((s1_id, c[0]))
            res = evaluate_predictions(f"Top-{K} (p>={tau_abs})", K, tau_abs, 0.0, pairs)
            step2_results.append(res)

    print(f"  Completed {len(step2_results)} Step 2 configurations.")

    # =========================================================================
    # STEP 3: TOP-K + CONFIDENCE + MARGIN
    # =========================================================================
    print("\n[Step 3] Running Step 3: Top-K + Confidence + Margin...", flush=True)
    tau_gap_list = [0.00, 0.0001, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10]
    all_grid_results = []

    # Pre-extract per-S1 top-1, top-2, and gap
    s1_p1 = {}
    s1_p2 = {}
    s1_gap = {}
    for s1_id in val_s1_ids:
        cands = cands_by_s1.get(s1_id, [])
        p1 = cands[0][1] if len(cands) > 0 else 0.0
        p2 = cands[1][1] if len(cands) > 1 else 0.0
        s1_p1[s1_id] = p1
        s1_p2[s1_id] = p2
        s1_gap[s1_id] = p1 - p2

    for K in k_list:
        for tau_abs in tau_abs_list:
            for tau_gap in tau_gap_list:
                pairs = set()
                for s1_id in val_s1_ids:
                    p1 = s1_p1[s1_id]
                    gap = s1_gap[s1_id]
                    # Gate condition: p1 >= tau_abs AND gap >= tau_gap
                    if p1 >= tau_abs and gap >= tau_gap:
                        # Keep at most K candidates that satisfy p >= tau_abs
                        cands = [c for c in cands_by_s1.get(s1_id, []) if c[1] >= tau_abs]
                        for c in cands[:K]:
                            pairs.add((s1_id, c[0]))

                rule_name = f"K={K}_abs={tau_abs:.2f}_gap={tau_gap}"
                res = evaluate_predictions(rule_name, K, tau_abs, tau_gap, pairs)
                all_grid_results.append(res)

    print(f"  Completed {len(all_grid_results)} total grid configurations.")

    # =========================================================================
    # STEP 4: MASTER TABLE SORTED BY F0.5
    # =========================================================================
    master_df = pd.DataFrame(all_grid_results)
    sorted_df = master_df.sort_values(by="macro_f0_5", ascending=False).reset_index(drop=True)

    print("\nTop 15 Configurations by Macro F0.5:")
    cols_display = ["rule", "K", "tau_abs", "tau_gap", "macro_f0_5", "macro_precision", "macro_recall", "tp", "fp", "predicted_pairs", "zero_match_receiving_predictions", "singleton_tp_retention_pct", "multi_match_tp_retention_pct"]
    for idx, r in sorted_df.head(15).iterrows():
        print(f"  [{idx+1:2d}] {r['rule']:26s} | F0.5={r['macro_f0_5']:.6f} | Prec={r['macro_precision']*100:6.3f}% | Rec={r['macro_recall']*100:6.3f}% | TP={r['tp']:5d} | FP={r['fp']:5d} | ZeroPreds={r['zero_match_receiving_predictions']:2d} | SingRet={r['singleton_tp_retention_pct']:5.1f}% | MultiRet={r['multi_match_tp_retention_pct']:5.1f}%")

    # =========================================================================
    # STEP 5: SUBGROUP ANALYSIS FOR TOP 10 CONFIGURATIONS
    # =========================================================================
    print("\n[Step 5] Detailed Subgroup Analysis for Top 10 Configurations...", flush=True)
    top10_records = []

    for rank, row in sorted_df.head(10).iterrows():
        K = int(row['K'])
        tau_abs = float(row['tau_abs'])
        tau_gap = float(row['tau_gap'])
        rule_name = row['rule']

        # Reconstruct predictions
        pred_pairs = set()
        for s1_id in val_s1_ids:
            p1 = s1_p1[s1_id]
            gap = s1_gap[s1_id]
            if p1 >= tau_abs and gap >= tau_gap:
                cands = [c for c in cands_by_s1.get(s1_id, []) if c[1] >= tau_abs]
                for c in cands[:K]:
                    pred_pairs.add((s1_id, c[0]))

        pred_counts = Counter(p[0] for p in pred_pairs)

        subgroups = {}
        for grp_label, grp_ids, base_grp_tp, total_gt in [
            ("zero_match", zero_match_s1_ids, 0, 0),
            ("singleton", singleton_s1_ids, base_singleton_tp, gt_pairs_singleton),
            ("multi_match", multi_match_s1_ids, base_multi_tp, gt_pairs_multi),
        ]:
            grp_set = set(grp_ids)
            grp_preds = {p for p in pred_pairs if p[0] in grp_set}
            grp_gt = {p for p in exact_matches_set if p[0] in grp_set}

            grp_tp = len(grp_preds & grp_gt)
            grp_fp = len(grp_preds - grp_gt)
            grp_fn = len(grp_gt - grp_preds)

            counts = [pred_counts[s1] for s1 in grp_ids]
            mean_m = float(np.mean(counts))
            median_m = float(np.median(counts))

            tp_ret_vs_base = (grp_tp / base_grp_tp * 100) if base_grp_tp > 0 else (100.0 if grp_tp == 0 else 0.0)

            subgroups[grp_label] = {
                "entity_count": len(grp_ids),
                "tp": grp_tp,
                "fp": grp_fp,
                "fn": grp_fn,
                "mean_predicted_matches": round(mean_m, 2),
                "median_predicted_matches": round(median_m, 2),
                "tp_retention_vs_base_pct": round(tp_ret_vs_base, 2),
            }

        rec = {
            "rank": rank + 1,
            "rule": rule_name,
            "K": K,
            "tau_abs": tau_abs,
            "tau_gap": tau_gap,
            "overall_metrics": row.to_dict(),
            "subgroups": subgroups,
        }
        top10_records.append(rec)

    # Save output JSON
    output_data = {
        "benchmark": {
            "validation_universe": "500-S1 France Benchmark",
            "total_s1": len(val_s1_ids),
            "zero_match_s1": len(zero_match_s1_ids),
            "singleton_s1": len(singleton_s1_ids),
            "multi_match_s1": len(multi_match_s1_ids),
            "total_ground_truth_pairs": len(exact_matches_set),
            "baseline_macro_f05": base_macro['macro_f0_5'],
            "baseline_tp": base_tp,
            "baseline_fp": base_fp,
            "baseline_fn": base_fn,
            "baseline_singleton_tp": base_singleton_tp,
            "baseline_multi_match_tp": base_multi_tp,
        },
        "step1_topk_only": step1_results,
        "step2_topk_confidence": step2_results,
        "all_grid_configurations_count": len(all_grid_results),
        "top_10_configurations": top10_records,
        "master_table": all_grid_results,
    }

    out_json = "experiments/phase12a_topk_confidence_margin.json"
    with open(out_json, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved JSON results to {out_json}")

    return output_data, sorted_df


if __name__ == "__main__":
    run_phase12a()
