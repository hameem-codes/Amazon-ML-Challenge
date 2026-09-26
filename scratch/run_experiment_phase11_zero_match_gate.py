"""
PHASE 11: ZERO-MATCH GATE DIAGNOSTIC EXPERIMENT

Evaluates whether an inference-time zero-match gate can suppress false matches
for genuine zero-match entities without destroying legitimate singleton/multi-match matches.

Locks:
- Config A blocking
- Legacy evidence ranking
- Candidate cap 400
- Phase 5 XGBoost model
- 57-feature schema
- Production code (UNTOUCHED)
- 500-France validation benchmark
"""

import sys, os, time, gc, json, math
from collections import defaultdict, Counter
sys.path.insert(0, 'src')
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
from rapidfuzz.distance import JaroWinkler

from normalize import normalize_business_name, normalize_country, normalize_business_address
from config import get_config_fingerprint
from metrics import compute_macro_f05
from blocking import ConfigABlockingEngine


def token_jaccard(s1, s2):
    t1 = set(s1.split())
    t2 = set(s2.split())
    if not t1 or not t2:
        return 0.0
    return len(t1 & t2) / len(t1 | t2)


def run_phase11():
    print("=" * 85, flush=True)
    print("PHASE 11: ZERO-MATCH GATE DIAGNOSTIC EXPERIMENT", flush=True)
    print("=" * 85, flush=True)

    fp = get_config_fingerprint()
    print(f"Production Fingerprint: {fp}", flush=True)
    assert fp == "87f20ceeb84ccc6ea2d48678c7810ac5", f"Fingerprint mismatch: {fp}"

    # 1. Load validation benchmark & ground truth
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

    # 2. Load candidate pairs & probabilities
    print("\n[Step 2/5] Loading scored candidate pool (CAP-400)...", flush=True)
    cache_file = 'scratch/cap400_scored_candidates.pkl'
    assert os.path.exists(cache_file), f"Missing cache: {cache_file}"
    cand_prob_df = pd.read_pickle(cache_file)

    cands_e6 = pd.read_csv('scratch/candidates_e6_0.tsv.gz', sep='\t', compression='gzip')
    assert len(cand_prob_df) == len(cands_e6), "Length mismatch between candidates and scores"

    # Pre-index target lexical rank
    eid_to_lex = dict(zip(engine.target_eids, engine.target_lex_rank))
    cand_prob_df['target_lex_rank'] = [eid_to_lex[eid] for eid in cand_prob_df['entity_id_cand']]
    cand_prob_df['num_blocking_keys'] = cands_e6['num_blocking_keys'].values
    cand_prob_df['evidence_score'] = cands_e6['evidence_score'].values

    # Compute text similarities for all 200k pairs
    print("  Computing text similarities for all 200,000 candidate pairs...", flush=True)
    s1_names = [s1_dict[s1]['norm_name'] for s1 in cand_prob_df['entity_id_s1']]
    s1_addrs = [s1_dict[s1]['norm_address'] for s1 in cand_prob_df['entity_id_s1']]
    c_names = [target_dict[c]['norm_name'] for c in cand_prob_df['entity_id_cand']]
    c_addrs = [target_dict[c]['norm_address'] for c in cand_prob_df['entity_id_cand']]

    name_jw = [float(JaroWinkler.similarity(s, t)) for s, t in zip(s1_names, c_names)]
    addr_jw = [float(JaroWinkler.similarity(s, t)) for s, t in zip(s1_addrs, c_addrs)]
    name_jaccard = [float(token_jaccard(s, t)) for s, t in zip(s1_names, c_names)]
    addr_jaccard = [float(token_jaccard(s, t)) for s, t in zip(s1_addrs, c_addrs)]
    name_exact = [1 if s == t else 0 for s, t in zip(s1_names, c_names)]
    addr_exact = [1 if s == t else 0 for s, t in zip(s1_addrs, c_addrs)]

    cand_prob_df['name_jw'] = name_jw
    cand_prob_df['addr_jw'] = addr_jw
    cand_prob_df['name_jaccard'] = name_jaccard
    cand_prob_df['addr_jaccard'] = addr_jaccard
    cand_prob_df['name_exact'] = name_exact
    cand_prob_df['addr_exact'] = addr_exact
    cand_prob_df['country_match'] = 1.0  # all France

    # Strong evidence flags
    cand_prob_df['strong_name'] = (
        (cand_prob_df['name_exact'] == 1) |
        (cand_prob_df['name_jaccard'] >= 0.5) |
        (cand_prob_df['name_jw'] >= 0.90)
    ).astype(int)

    cand_prob_df['strong_address'] = (
        (cand_prob_df['addr_exact'] == 1) |
        (cand_prob_df['addr_jaccard'] >= 0.5) |
        (cand_prob_df['addr_jw'] >= 0.85)
    ).astype(int)

    # 3. Build per-S1 Candidate & Prediction structures
    print("\n[Step 3/5] Building per-S1 entity diagnostics...", flush=True)

    # All candidates per S1 sorted by (-prob, lex_rank)
    cands_by_s1 = defaultdict(list)
    for row in cand_prob_df.itertuples():
        cands_by_s1[row.entity_id_s1].append(row)

    for s1 in cands_by_s1:
        cands_by_s1[s1].sort(key=lambda r: (-r.probability, r.target_lex_rank))

    # Phase 10 Gap-0.0001 baseline predictions
    # Filter to >= 0.99 first, then gap stopping <= 0.0001
    baseline_gap_preds_by_s1 = defaultdict(list)
    for s1_id in val_s1_ids:
        cands = cands_by_s1.get(s1_id, [])
        preds_099 = [c for c in cands if c.probability >= 0.990]
        if not preds_099:
            continue
        # Add top candidate
        selected = [preds_099[0]]
        for idx in range(len(preds_099) - 1):
            cur_p = preds_099[idx].probability
            next_p = preds_099[idx + 1].probability
            if (cur_p - next_p) <= 0.0001:
                selected.append(preds_099[idx + 1])
            else:
                break
        baseline_gap_preds_by_s1[s1_id] = selected

    # Verify Baseline Reproduction
    base_pairs = set()
    for s1_id, c_list in baseline_gap_preds_by_s1.items():
        for c in c_list:
            base_pairs.add((s1_id, c.entity_id_cand))

    base_macro = compute_macro_f05(val_s1_ids, exact_matches_set, base_pairs)
    base_tp = len(base_pairs & exact_matches_set)
    base_fp = len(base_pairs - exact_matches_set)
    base_fn = len(exact_matches_set - base_pairs)
    base_p_prec = base_tp / (base_tp + base_fp) if (base_tp + base_fp) > 0 else 0.0
    base_p_rec = base_tp / (base_tp + base_fn) if (base_tp + base_fn) > 0 else 0.0

    print(f"Phase 10 Gap-0.0001 Baseline Verification:")
    print(f"  Macro F0.5:     {base_macro['macro_f0_5']:.6f} (Expected: 0.317727)")
    print(f"  Macro Precision:{base_macro['macro_precision']*100:.3f}% (Expected: 32.602%)")
    print(f"  Macro Recall:   {base_macro['macro_recall']*100:.3f}% (Expected: 70.270%)")
    print(f"  TP: {base_tp} (Expected: 1497), FP: {base_fp} (Expected: 3995), FN: {base_fn} (Expected: 2694)")
    assert abs(base_macro['macro_f0_5'] - 0.317727) < 1e-4, f"Baseline mismatch: {base_macro['macro_f0_5']}"

    # Group baseline TPs
    base_singleton_tp = len({p for p in base_pairs if p[0] in set(singleton_s1_ids)} & exact_matches_set)
    base_multi_tp = len({p for p in base_pairs if p[0] in set(multi_match_s1_ids)} & exact_matches_set)
    print(f"  Baseline TPs: Singleton={base_singleton_tp}/{gt_pairs_singleton}, Multi={base_multi_tp}/{gt_pairs_multi}")

    # =========================================================================
    # STEP 1: ZERO-MATCH DIAGNOSTIC TABLE
    # =========================================================================
    print("\n[Step 1 Diagnostic Table: Feature Distributions by GT Group]...", flush=True)

    s1_diag_records = []
    for s1_id in val_s1_ids:
        cands = cands_by_s1.get(s1_id, [])
        top_cand = cands[0] if cands else None
        sec_cand = cands[1] if len(cands) > 1 else None

        cand_count = len(cands)
        max_prob = float(top_cand.probability) if top_cand else 0.0
        sec_prob = float(sec_cand.probability) if sec_cand else 0.0
        gap = max_prob - sec_prob

        n_099 = sum(1 for c in cands if c.probability >= 0.990)
        n_gap = len(baseline_gap_preds_by_s1.get(s1_id, []))

        max_name_sim = max(c.name_jw for c in cands) if cands else 0.0
        max_addr_sim = max(c.addr_jw for c in cands) if cands else 0.0
        max_country = 1.0
        max_keys = max(c.num_blocking_keys for c in cands) if cands else 0
        max_evidence = max(c.evidence_score for c in cands) if cands else 0.0

        rec = {
            "s1_id": s1_id,
            "gt_type": "zero_match" if s1_id in set(zero_match_s1_ids) else ("singleton" if s1_id in set(singleton_s1_ids) else "multi_match"),
            "candidate_count": cand_count,
            "max_model_probability": max_prob,
            "second_highest_probability": sec_prob,
            "top_probability_gap": gap,
            "number_predictions_at_099": n_099,
            "number_predictions_after_gap_0001": n_gap,
            "max_name_sim": max_name_sim,
            "max_addr_sim": max_addr_sim,
            "max_country": max_country,
            "max_shared_keys": max_keys,
            "max_blocking_evidence": max_evidence,
            "top_cand_id": top_cand.entity_id_cand if top_cand else None,
            "top_cand_name_jw": float(top_cand.name_jw) if top_cand else 0.0,
            "top_cand_name_jaccard": float(top_cand.name_jaccard) if top_cand else 0.0,
            "top_cand_name_exact": int(top_cand.name_exact) if top_cand else 0,
            "top_cand_addr_jw": float(top_cand.addr_jw) if top_cand else 0.0,
            "top_cand_addr_jaccard": float(top_cand.addr_jaccard) if top_cand else 0.0,
            "top_cand_addr_exact": int(top_cand.addr_exact) if top_cand else 0,
            "top_cand_country": 1.0,
            "top_cand_shared_keys": int(top_cand.num_blocking_keys) if top_cand else 0,
            "top_cand_blocking_evidence": float(top_cand.evidence_score) if top_cand else 0.0,
            "top_cand_probability": max_prob,
            "top_cand_strong_name": int(top_cand.strong_name) if top_cand else 0,
            "top_cand_strong_address": int(top_cand.strong_address) if top_cand else 0,
        }
        s1_diag_records.append(rec)

    diag_df = pd.DataFrame(s1_diag_records)

    # Compute distribution statistics per GT group
    dist_stats = {}
    for grp in ["zero_match", "singleton", "multi_match"]:
        grp_df = diag_df[diag_df['gt_type'] == grp]
        dist_stats[grp] = {
            "entity_count": len(grp_df),
            "max_prob": {
                "mean": round(float(grp_df['max_model_probability'].mean()), 6),
                "std": round(float(grp_df['max_model_probability'].std()), 6),
                "min": round(float(grp_df['max_model_probability'].min()), 6),
                "median": round(float(grp_df['max_model_probability'].median()), 6),
                "max": round(float(grp_df['max_model_probability'].max()), 6),
            },
            "second_prob": {
                "mean": round(float(grp_df['second_highest_probability'].mean()), 6),
                "median": round(float(grp_df['second_highest_probability'].median()), 6),
            },
            "top_prob_gap": {
                "mean": round(float(grp_df['top_probability_gap'].mean()), 6),
                "median": round(float(grp_df['top_probability_gap'].median()), 6),
                "min": round(float(grp_df['top_probability_gap'].min()), 6),
                "max": round(float(grp_df['top_probability_gap'].max()), 6),
            },
            "predictions_at_099": {
                "mean": round(float(grp_df['number_predictions_at_099'].mean()), 2),
                "median": round(float(grp_df['number_predictions_at_099'].median()), 2),
            },
            "predictions_after_gap_0001": {
                "mean": round(float(grp_df['number_predictions_after_gap_0001'].mean()), 2),
                "median": round(float(grp_df['number_predictions_after_gap_0001'].median()), 2),
            },
            "max_name_sim": {
                "mean": round(float(grp_df['max_name_sim'].mean()), 4),
                "median": round(float(grp_df['max_name_sim'].median()), 4),
            },
            "max_addr_sim": {
                "mean": round(float(grp_df['max_addr_sim'].mean()), 4),
                "median": round(float(grp_df['max_addr_sim'].median()), 4),
            },
            "max_blocking_evidence": {
                "mean": round(float(grp_df['max_blocking_evidence'].mean()), 1),
                "median": round(float(grp_df['max_blocking_evidence'].median()), 1),
            },
            "top_cand_name_jw": {
                "mean": round(float(grp_df['top_cand_name_jw'].mean()), 4),
                "median": round(float(grp_df['top_cand_name_jw'].median()), 4),
                "min": round(float(grp_df['top_cand_name_jw'].min()), 4),
            },
            "top_cand_addr_jw": {
                "mean": round(float(grp_df['top_cand_addr_jw'].mean()), 4),
                "median": round(float(grp_df['top_cand_addr_jw'].median()), 4),
                "min": round(float(grp_df['top_cand_addr_jw'].min()), 4),
            },
            "top_cand_shared_keys": {
                "mean": round(float(grp_df['top_cand_shared_keys'].mean()), 2),
                "median": round(float(grp_df['top_cand_shared_keys'].median()), 2),
            },
            "top_cand_blocking_evidence": {
                "mean": round(float(grp_df['top_cand_blocking_evidence'].mean()), 1),
                "median": round(float(grp_df['top_cand_blocking_evidence'].median()), 1),
            },
            "top_cand_strong_name_rate": round(float(grp_df['top_cand_strong_name'].mean()), 4),
            "top_cand_strong_address_rate": round(float(grp_df['top_cand_strong_address'].mean()), 4),
            "top_cand_either_strong_rate": round(float(((grp_df['top_cand_strong_name'] == 1) | (grp_df['top_cand_strong_address'] == 1)).mean()), 4),
            "top_cand_neither_strong_rate": round(float(((grp_df['top_cand_strong_name'] == 0) & (grp_df['top_cand_strong_address'] == 0)).mean()), 4),
        }

    print("Distribution Summary:")
    for grp, st in dist_stats.items():
        print(f"  [{grp.upper()} (N={st['entity_count']})]: max_prob mean={st['max_prob']['mean']:.6f}, min={st['max_prob']['min']:.6f} | top_name_jw mean={st['top_cand_name_jw']['mean']:.4f} | top_addr_jw mean={st['top_cand_addr_jw']['mean']:.4f} | neither_strong={st['top_cand_neither_strong_rate']*100:.1f}%")

    # =========================================================================
    # STEP 2 & 3: EVALUATION FUNCTION FOR GATES
    # =========================================================================
    def evaluate_gate(gate_name, gate_fn, base_pred_map):
        """
        gate_fn(s1_diag_row) returns True if entity should be SUPPRESSED (predictions set to empty).
        """
        filtered_pairs = set()
        suppressed_s1 = set()

        for s1_id in val_s1_ids:
            row = diag_df[diag_df['s1_id'] == s1_id].iloc[0]
            if gate_fn(row):
                suppressed_s1.add(s1_id)
            else:
                for c in base_pred_map.get(s1_id, []):
                    filtered_pairs.add((s1_id, c.entity_id_cand))

        macro = compute_macro_f05(val_s1_ids, exact_matches_set, filtered_pairs)
        tp = len(filtered_pairs & exact_matches_set)
        fp = len(filtered_pairs - exact_matches_set)
        fn = len(exact_matches_set - filtered_pairs)
        p_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        p_rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        pred_counts = Counter(p[0] for p in filtered_pairs)
        counts_all = [pred_counts[s1] for s1 in val_s1_ids]
        mean_p = float(np.mean(counts_all))
        median_p = float(np.median(counts_all))

        zero_with_preds = sum(1 for s1 in zero_match_s1_ids if pred_counts[s1] > 0)
        zero_correct_empty = len(zero_match_s1_ids) - zero_with_preds

        # TP retention
        singleton_tp = len({p for p in filtered_pairs if p[0] in set(singleton_s1_ids)} & exact_matches_set)
        multi_tp = len({p for p in filtered_pairs if p[0] in set(multi_match_s1_ids)} & exact_matches_set)

        sing_retention_vs_base = (singleton_tp / base_singleton_tp * 100) if base_singleton_tp > 0 else 0.0
        sing_retention_vs_gt = (singleton_tp / gt_pairs_singleton * 100) if gt_pairs_singleton > 0 else 0.0
        multi_retention_vs_base = (multi_tp / base_multi_tp * 100) if base_multi_tp > 0 else 0.0
        multi_retention_vs_gt = (multi_tp / gt_pairs_multi * 100) if gt_pairs_multi > 0 else 0.0

        # TP lost vs baseline
        sing_tp_lost = base_singleton_tp - singleton_tp
        multi_tp_lost = base_multi_tp - multi_tp
        total_tp_lost = base_tp - tp

        return {
            "gate_name": gate_name,
            "macro_precision": round(macro['macro_precision'], 6),
            "macro_recall": round(macro['macro_recall'], 6),
            "macro_f0_5": round(macro['macro_f0_5'], 6),
            "pairwise_precision": round(p_prec, 6),
            "pairwise_recall": round(p_rec, 6),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "predicted_pairs": len(filtered_pairs),
            "mean_predictions_per_s1": round(mean_p, 2),
            "median_predictions_per_s1": round(median_p, 2),
            "zero_match_with_predictions": zero_with_preds,
            "zero_match_correct_empty": zero_correct_empty,
            "singleton_tp_retained": singleton_tp,
            "singleton_tp_lost": sing_tp_lost,
            "singleton_tp_retention_pct_vs_base": round(sing_retention_vs_base, 2),
            "singleton_tp_retention_pct_vs_gt": round(sing_retention_vs_gt, 2),
            "multi_match_tp_retained": multi_tp,
            "multi_match_tp_lost": multi_tp_lost,
            "multi_match_tp_retention_pct_vs_base": round(multi_retention_vs_base, 2),
            "multi_match_tp_retention_pct_vs_gt": round(multi_retention_vs_gt, 2),
            "total_tp_lost": total_tp_lost,
            "suppressed_s1_count": len(suppressed_s1),
            "suppressed_by_group": {
                "zero_match": sum(1 for s in suppressed_s1 if s in set(zero_match_s1_ids)),
                "singleton": sum(1 for s in suppressed_s1 if s in set(singleton_s1_ids)),
                "multi_match": sum(1 for s in suppressed_s1 if s in set(multi_match_s1_ids)),
            }
        }

    # Define Gates
    gates_definitions = {
        "G1 (< 0.999)": lambda r: r['max_model_probability'] < 0.999,
        "G2 (< 0.9995)": lambda r: r['max_model_probability'] < 0.9995,
        "G3 (< 0.9998)": lambda r: r['max_model_probability'] < 0.9998,
        "G4 (no strong name)": lambda r: r['top_cand_strong_name'] == 0,
        "G5 (no strong address)": lambda r: r['top_cand_strong_address'] == 0,
        "G6 (neither strong name nor strong address)": lambda r: (r['top_cand_strong_name'] == 0) and (r['top_cand_strong_address'] == 0),
    }

    # Also test combined gate if useful
    # Combined: G3 + G6 (probability < 0.9998 OR neither strong)
    combined_gate_g3_g6 = lambda r: (r['max_model_probability'] < 0.9998) or ((r['top_cand_strong_name'] == 0) and (r['top_cand_strong_address'] == 0))

    # Evaluate Baseline (No Gate)
    baseline_metrics = evaluate_gate("Baseline (Gap <= 0.0001, No Gate)", lambda r: False, baseline_gap_preds_by_s1)

    print("\n[Step 2/5] Evaluating Simple Gates on Gap-0.0001 Baseline...", flush=True)
    step2_results = {}
    for gname, gfn in gates_definitions.items():
        res = evaluate_gate(gname, gfn, baseline_gap_preds_by_s1)
        step2_results[gname] = res
        print(f"  {gname:45s}: F0.5={res['macro_f0_5']:.6f} | Prec={res['macro_precision']*100:6.3f}% | Rec={res['macro_recall']*100:6.3f}% | TP={res['tp']:5d} (lost {res['total_tp_lost']:3d}) | FP={res['fp']:5d} | ZeroEmpty={res['zero_match_correct_empty']:2d}/94 | SingRet={res['singleton_tp_retention_pct_vs_base']:5.1f}% | MultiRet={res['multi_match_tp_retention_pct_vs_base']:5.1f}%")

    print("\n[Step 3/5] Evaluating Combined Decision Rule (Gap-0.0001 + Gate)...", flush=True)
    # Test combined gate
    combined_res = evaluate_gate("Combined (Gap <= 0.0001 + G3 + G6)", combined_gate_g3_g6, baseline_gap_preds_by_s1)
    print(f"  Combined (Gap <= 0.0001 + G3 + G6)         : F0.5={combined_res['macro_f0_5']:.6f} | Prec={combined_res['macro_precision']*100:6.3f}% | Rec={combined_res['macro_recall']*100:6.3f}% | TP={combined_res['tp']:5d} (lost {combined_res['total_tp_lost']:3d}) | FP={combined_res['fp']:5d} | ZeroEmpty={combined_res['zero_match_correct_empty']:2d}/94 | SingRet={combined_res['singleton_tp_retention_pct_vs_base']:5.1f}% | MultiRet={combined_res['multi_match_tp_retention_pct_vs_base']:5.1f}%")

    # Also evaluate Gates Standalone on Raw >= 0.99 Predictions (for comprehensive reporting)
    print("\n[Diagnostic: Evaluating Gates Standalone on Raw >= 0.99 (without Gap stopping)]...", flush=True)
    raw_099_preds_by_s1 = defaultdict(list)
    for s1_id in val_s1_ids:
        raw_099_preds_by_s1[s1_id] = [c for c in cands_by_s1.get(s1_id, []) if c.probability >= 0.990]

    raw_base_metrics = evaluate_gate("Raw >= 0.99 Baseline", lambda r: False, raw_099_preds_by_s1)
    step2_standalone_results = {}
    for gname, gfn in gates_definitions.items():
        res = evaluate_gate(f"Raw >= 0.99 + {gname}", gfn, raw_099_preds_by_s1)
        step2_standalone_results[gname] = res

    # =========================================================================
    # STEP 4: ZERO-MATCH SAFETY & TOP 20 HARDEST ZERO-MATCH ENTITIES
    # =========================================================================
    print("\n[Step 4/5] Detailed Zero-Match Inspection...", flush=True)

    # Pick the best gate among G1-G6 by F0.5 or safety
    # Find gate with highest F0.5
    best_gate_name = max(step2_results.keys(), key=lambda k: step2_results[k]['macro_f0_5'])
    best_gate_res = step2_results[best_gate_name]
    best_gate_fn = gates_definitions[best_gate_name]
    print(f"  Highest F0.5 Gate among G1-G6: {best_gate_name} (Macro F0.5: {best_gate_res['macro_f0_5']:.6f})")

    # Zero match details
    zero_df = diag_df[diag_df['gt_type'] == 'zero_match'].copy().reset_index(drop=True)
    zero_suppressed = [best_gate_fn(row) for _, row in zero_df.iterrows()]
    zero_df['suppressed_by_best_gate'] = zero_suppressed

    print(f"  GT Zero-Match Entities: {len(zero_df)}")
    print(f"  Remaining empty under {best_gate_name}: {sum(zero_suppressed)} / {len(zero_df)}")
    print(f"  Still receiving predictions: {len(zero_df) - sum(zero_suppressed)} / {len(zero_df)}")

    # Sort zero-match entities by hardest (highest max_probability, then highest evidence score)
    hard_zero_df = zero_df.sort_values(by=['max_model_probability', 'top_cand_blocking_evidence'], ascending=[False, False]).head(20)

    hardest_20_list = []
    for idx, r in hard_zero_df.iterrows():
        s1_row = s1_dict[r['s1_id']]
        top_cand_row = target_dict[r['top_cand_id']]

        item = {
            "s1_id": r['s1_id'],
            "s1_name": s1_row['norm_name'],
            "s1_address": s1_row['norm_address'],
            "top_target_id": r['top_cand_id'],
            "top_target_name": top_cand_row['norm_name'],
            "top_target_address": top_cand_row['norm_address'],
            "max_probability": round(float(r['max_model_probability']), 6),
            "name_similarity_jw": round(float(r['top_cand_name_jw']), 4),
            "name_token_jaccard": round(float(r['top_cand_name_jaccard']), 4),
            "address_similarity_jw": round(float(r['top_cand_addr_jw']), 4),
            "address_token_jaccard": round(float(r['top_cand_addr_jaccard']), 4),
            "country": "france",
            "blocking_evidence": float(r['top_cand_blocking_evidence']),
            "shared_keys": int(r['top_cand_shared_keys']),
            "strong_name": int(r['top_cand_strong_name']),
            "strong_address": int(r['top_cand_strong_address']),
            "suppressed_by_gate": bool(r['suppressed_by_best_gate']),
            "predictions_count_baseline": int(r['number_predictions_after_gap_0001']),
        }
        hardest_20_list.append(item)

    print("\nTop 5 Hardest Zero-Match Entities:")
    for h in hardest_20_list[:5]:
        print(f"  S1: {h['s1_id']} | Target: {h['top_target_id']} | Prob: {h['max_probability']:.6f} | NameJW: {h['name_similarity_jw']:.4f} | AddrJW: {h['address_similarity_jw']:.4f} | Suppressed: {h['suppressed_by_gate']}")

    # =========================================================================
    # STEP 5: LEGITIMATE MATCH SAFETY
    # =========================================================================
    print("\n[Step 5/5] Legitimate Match Safety Audit...", flush=True)
    # Check TP loss for each gate
    for gname, gres in step2_results.items():
        sing_loss = gres['singleton_tp_lost']
        multi_loss = gres['multi_match_tp_lost']
        tot_loss = gres['total_tp_lost']
        print(f"  {gname:45s}: Singleton TP Retention = {gres['singleton_tp_retention_pct_vs_base']:5.1f}% (Lost {sing_loss:2d}), Multi TP Retention = {gres['multi_match_tp_retention_pct_vs_base']:5.1f}% (Lost {multi_loss:2d}), Total TP Lost = {tot_loss:3d}")

    # =========================================================================
    # SAVE JSON RESULTS
    # =========================================================================
    output_json = {
        "benchmark": {
            "validation_universe": "500-S1 France Benchmark",
            "targets_count": len(targets),
            "total_s1": len(val_s1_ids),
            "zero_match_s1": len(zero_match_s1_ids),
            "singleton_s1": len(singleton_s1_ids),
            "multi_match_s1": len(multi_match_s1_ids),
            "total_ground_truth_pairs": len(exact_matches_set),
            "singleton_ground_truth_pairs": gt_pairs_singleton,
            "multi_match_ground_truth_pairs": gt_pairs_multi,
            "production_fingerprint": fp,
        },
        "step1_distribution_statistics": dist_stats,
        "baseline_gap_0001": baseline_metrics,
        "step2_simple_gates": step2_results,
        "step3_combined_gate": combined_res,
        "step2_standalone_gates_on_raw_099": step2_standalone_results,
        "step4_hardest_20_zero_match": hardest_20_list,
        "best_gate": {
            "name": best_gate_name,
            "metrics": best_gate_res
        }
    }

    os.makedirs('experiments', exist_ok=True)
    out_path = 'experiments/phase11_zero_match_gate.json'
    with open(out_path, 'w') as f:
        json.dump(output_json, f, indent=2)
    print(f"\nSaved results to {out_path}", flush=True)

    return output_json


if __name__ == '__main__':
    run_phase11()
