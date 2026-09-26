"""
PHASE — PRE-RANKING ABLATION EXPERIMENT

Evaluates whether classical ranking functions can improve Recall@400
over the legacy evidence ranking on the existing Config A raw candidate pool.

Benchmark:
- Same 500 S1 France stress benchmark
- Same France target pool (1,434,993 targets)
- Reference ground truth: 4,191 true pairs
- Same Config A raw candidate pool
- Candidate cap: 400 per S1
- Deterministic tie-breaking:
    1. ranking score DESC
    2. blocking_channel_count DESC
    3. target entity ID ASC

Downstream Validation:
- Phase 5 XGBoost model (models/xgboost_entity_resolution_phase5_configA.json)
- 57 locked features
- p >= 0.99
- gap <= 0.0001 stopping rule
"""

import sys, os, time, gc, json, psutil
from collections import defaultdict, Counter
sys.path.insert(0, 'src')
sys.path.insert(0, '.')

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import pandas as pd
import numpy as np
import xgboost as xgb
from rapidfuzz.distance import JaroWinkler
from sklearn.feature_extraction.text import TfidfVectorizer

from normalize import normalize_business_name, normalize_country, normalize_business_address
from blocking import ConfigABlockingEngine
from config import get_config_fingerprint
from features import build_vectorized_features, FEATURE_NAMES
from metrics import compute_macro_f05
from tfidf_model import TransductiveTFIDF


def token_jaccard(s1, s2):
    t1 = set(s1.split())
    t2 = set(s2.split())
    if not t1 or not t2:
        return 0.0
    return len(t1 & t2) / len(t1 | t2)


def length_similarity(s1, s2):
    l1, l2 = len(s1), len(s2)
    if l1 == 0 or l2 == 0:
        return 0.0
    return 1.0 - abs(l1 - l2) / max(l1, l2)


def fit_transductive_tfidf():
    print("  Fitting Transductive Unlabeled TF-IDF for 57 features...", flush=True)
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


def run_ablation():
    print("=" * 85)
    print("PHASE: PRE-RANKING ABLATION EXPERIMENT")
    print("=" * 85)

    fp = get_config_fingerprint()
    print(f"Production Fingerprint: {fp}")
    assert fp == "87f20ceeb84ccc6ea2d48678c7810ac5", f"Fingerprint mismatch: {fp}"

    proc = psutil.Process(os.getpid())
    peak_ram = proc.memory_info().rss

    # 1. Load targets
    print("\n[Step 1/6] Loading 1,434,993 France targets and ConfigABlockingEngine...")
    t0 = time.time()
    targets = pd.read_pickle('scratch/targets_france.pkl')
    engine = ConfigABlockingEngine(targets)
    target_names = engine.target_names
    target_addrs = engine.target_addrs
    target_eids = engine.target_eids
    target_lex = engine.target_lex_rank
    target_dict = targets.set_index('entity_id').to_dict(orient='index')
    print(f"Targets & Engine ready in {time.time()-t0:.2f}s")

    # 2. Fit TF-IDF models for character 3-grams
    print("\n[Step 2/6] Fitting sparse Character 3-gram TF-IDF vectorizers...")
    t0 = time.time()
    vec_name = TfidfVectorizer(analyzer="char", ngram_range=(3, 3), min_df=2)
    X_target_name = vec_name.fit_transform(pd.Series(target_names).fillna(''))
    print(f"  Name TF-IDF fitted in {time.time()-t0:.2f}s, shape: {X_target_name.shape}")

    t0 = time.time()
    vec_addr = TfidfVectorizer(analyzer="char", ngram_range=(3, 3), min_df=2)
    X_target_addr = vec_addr.fit_transform(pd.Series(target_addrs).fillna(''))
    print(f"  Address TF-IDF fitted in {time.time()-t0:.2f}s, shape: {X_target_addr.shape}")

    # 3. Load 500 France S1 & Ground Truth
    print("\n[Step 3/6] Loading 500-France validation benchmark...")
    s1_df = pd.read_csv('data/test/test_source1.tsv', sep='\t', dtype=str, nrows=10000, na_filter=False)
    s1_fr = s1_df[s1_df['country'].apply(normalize_country) == 'france'].head(500).copy().reset_index(drop=True)
    del s1_df
    gc.collect()
    s1_fr['norm_name'] = s1_fr['business_name'].apply(normalize_business_name)
    s1_fr['norm_address'] = s1_fr['business_address'].apply(normalize_business_address)
    s1_fr['norm_country'] = 'france'
    s1_dict = s1_fr.set_index('entity_id').to_dict(orient='index')
    val_s1_ids = list(s1_fr['entity_id'].values)

    s1_name_map = defaultdict(list)
    for idx, r in s1_fr.iterrows():
        s1_name_map[r['norm_name']].append(r['entity_id'])

    exact_matches_set = set()
    t_names = engine.target_names
    t_eids = engine.target_eids
    for idx in range(len(t_names)):
        t_name = t_names[idx]
        if t_name in s1_name_map:
            t_eid = t_eids[idx]
            for s1_id in s1_name_map[t_name]:
                exact_matches_set.add((s1_id, t_eid))

    TOTAL_GT_PAIRS = len(exact_matches_set)
    print(f"Validation dataset: {len(s1_fr)} S1 entities, {TOTAL_GT_PAIRS:,} reference true pairs")
    assert TOTAL_GT_PAIRS == 4191, f"Expected 4,191 true pairs, got {TOTAL_GT_PAIRS}"

    # Identify ground-truth groups
    gt_by_s1 = defaultdict(set)
    for s1_id, cand_id in exact_matches_set:
        gt_by_s1[s1_id].add(cand_id)
    zero_match_s1_ids = [s1 for s1 in val_s1_ids if len(gt_by_s1[s1]) == 0]
    singleton_s1_ids = [s1 for s1 in val_s1_ids if len(gt_by_s1[s1]) == 1]
    multi_match_s1_ids = [s1 for s1 in val_s1_ids if len(gt_by_s1[s1]) > 1]

    # Pre-transform S1 TF-IDF
    s1_name_tfidf = vec_name.transform(s1_fr['norm_name'])
    s1_addr_tfidf = vec_addr.transform(s1_fr['norm_address'])

    # 4. Multi-Stage Ranking Ablation
    print("\n[Step 4/6] Executing 10-Stage Ranking Ablation on identical raw candidate pool...")

    stage_pairs = {i: set() for i in range(10)}
    stage_pairs_k300 = {i: set() for i in range(10)}
    stage_pairs_k500 = {i: set() for i in range(10)}
    stage_runtimes = {i: 0.0 for i in range(10)}
    stage_cap_hits = {i: 0 for i in range(10)}
    stage_cands_per_s1 = {i: [] for i in range(10)}
    total_raw_candidates = 0

    BIT_NAME = 1 << 1
    BIT_P3 = 1 << 2
    BIT_P4 = 1 << 3
    BIT_ADDR = 1 << 4

    country_name_tokens = engine.country_name_token_idx['france']
    country_p3 = engine.country_prefix3_idx['france']
    country_p4 = engine.country_prefix4_idx['france']
    country_addr_tokens = engine.country_addr_token_idx['france']

    t_start_ablation = time.time()

    # Numerical scale inspection variables
    scale_samples = []

    for s1_idx in range(len(s1_fr)):
        s1_row = s1_fr.iloc[s1_idx]
        s1_id = s1_row['entity_id']
        name = s1_row['norm_name']
        addr = s1_row['norm_address']

        s1_tokens = [t for t in name.split() if len(t) > 1 and t not in engine.stop_tokens]
        s1_p3 = name[:3] if len(name) >= 3 else ''
        s1_p4 = name[:4] if len(name) >= 4 else ''
        s1_addr_tokens = set([t for t in addr.split() if len(t) >= 4 and t not in engine.stop_addrs]) if addr else set()

        cand_flags = {}
        cand_scores = {}

        # Channel 2: Country + Name Token (selective)
        for t in s1_tokens:
            if t in country_name_tokens:
                for idx in country_name_tokens[t]:
                    cand_flags[idx] = cand_flags.get(idx, 0) | BIT_NAME
                    cand_scores[idx] = cand_scores.get(idx, 0) + 70

        # Channel 3: Country + Prefix 4
        if s1_p4 and s1_p4 in country_p4:
            for idx in country_p4[s1_p4]:
                cand_flags[idx] = cand_flags.get(idx, 0) | BIT_P4
                cand_scores[idx] = cand_scores.get(idx, 0) + 50

        # Channel 4: Country + Prefix 3
        if s1_p3 and s1_p3 in country_p3:
            for idx in country_p3[s1_p3]:
                cand_flags[idx] = cand_flags.get(idx, 0) | BIT_P3
                cand_scores[idx] = cand_scores.get(idx, 0) + 30

        # Channel 5: Country + Address Token
        for atok in s1_addr_tokens:
            if atok in country_addr_tokens:
                for idx in country_addr_tokens[atok]:
                    f = cand_flags.get(idx, 0)
                    if not (f & BIT_ADDR):
                        cand_flags[idx] = f | BIT_ADDR
                        cand_scores[idx] = cand_scores.get(idx, 0) + 60

        n_cands = len(cand_flags)
        total_raw_candidates += n_cands

        if n_cands == 0:
            for st in range(10):
                stage_cands_per_s1[st].append(0)
            continue

        cand_indices_all = np.array(list(cand_flags.keys()), dtype=np.int32)
        legacy_sc_all = np.array([float(cand_scores[idx]) for idx in cand_indices_all], dtype=np.float32)

        # Mathematical pruning of candidates that can never reach Top-500:
        # Maximum possible sum of all additions in Stage 1-9 is 12.0.
        # Any candidate with legacy_sc < S_500 - 12.0 cannot possibly beat the 500th candidate.
        if n_cands > 500:
            s_500_threshold = np.partition(legacy_sc_all, -500)[-500] - 12.0
            eligible_mask = legacy_sc_all >= s_500_threshold
            cand_indices = cand_indices_all[eligible_mask]
            legacy_sc = legacy_sc_all[eligible_mask]
        else:
            cand_indices = cand_indices_all
            legacy_sc = legacy_sc_all

        n_eligible = len(cand_indices)
        target_eids_cand = [target_eids[idx] for idx in cand_indices]
        target_lex_cand = [target_lex[idx] for idx in cand_indices]

        # Blocking channel count
        b_name = np.array([1 if (cand_flags[idx] & BIT_NAME) else 0 for idx in cand_indices], dtype=np.int8)
        b_p3 = np.array([1 if (cand_flags[idx] & BIT_P3) else 0 for idx in cand_indices], dtype=np.int8)
        b_p4 = np.array([1 if (cand_flags[idx] & BIT_P4) else 0 for idx in cand_indices], dtype=np.int8)
        b_addr = np.array([1 if (cand_flags[idx] & BIT_ADDR) else 0 for idx in cand_indices], dtype=np.int8)
        b_count = b_name + b_p3 + b_p4 + b_addr

        prefix_bon = np.where((b_p3 == 1) | (b_p4 == 1), 1.0, 0.0).astype(np.float32)

        c_names = [target_names[idx] for idx in cand_indices]
        c_addrs = [target_addrs[idx] for idx in cand_indices]

        name_jacc = np.array([token_jaccard(name, cn) for cn in c_names], dtype=np.float32)
        name_jw = np.array([float(JaroWinkler.similarity(name, cn)) for cn in c_names], dtype=np.float32)

        s1_vec_n = s1_name_tfidf[s1_idx]
        cand_mat_n = X_target_name[cand_indices]
        name_tfidf_sc = np.asarray(cand_mat_n.dot(s1_vec_n.T).todense()).ravel().astype(np.float32)

        addr_jacc = np.array([token_jaccard(addr, ca) for ca in c_addrs], dtype=np.float32)

        s1_vec_a = s1_addr_tfidf[s1_idx]
        cand_mat_a = X_target_addr[cand_indices]
        addr_tfidf_sc = np.asarray(cand_mat_a.dot(s1_vec_a.T).todense()).ravel().astype(np.float32)

        country_sc = np.ones(n_eligible, dtype=np.float32)
        len_sim = np.array([length_similarity(name, cn) for cn in c_names], dtype=np.float32)

        if len(scale_samples) < 5:
            scale_samples.append({
                "legacy_min": float(np.min(legacy_sc)),
                "legacy_max": float(np.max(legacy_sc)),
                "legacy_mean": float(np.mean(legacy_sc)),
                "similarity_sum_max": float(np.max(name_jacc + name_jw + name_tfidf_sc + addr_jacc + addr_tfidf_sc + country_sc + b_count + prefix_bon + len_sim))
            })

        # Define stage score arrays
        stage_scores = {}
        stage_scores[0] = legacy_sc
        stage_scores[1] = stage_scores[0] + name_jacc
        stage_scores[2] = stage_scores[1] + name_jw
        stage_scores[3] = stage_scores[2] + name_tfidf_sc
        stage_scores[4] = stage_scores[3] + addr_jacc
        stage_scores[5] = stage_scores[4] + addr_tfidf_sc
        stage_scores[6] = stage_scores[5] + country_sc
        stage_scores[7] = stage_scores[6] + b_count.astype(np.float32)
        stage_scores[8] = stage_scores[7] + prefix_bon
        stage_scores[9] = stage_scores[8] + len_sim

        for st in range(10):
            t_st0 = time.time()
            sc = stage_scores[st]

            if n_cands <= 400:
                selected_eids = target_eids_cand
                selected_300 = target_eids_cand[:300]
                selected_500 = target_eids_cand
            else:
                stage_cap_hits[st] += 1
                # Deterministic tie-breaking:
                # 1. score DESC (-sc)
                # 2. b_count DESC (-b_count)
                # 3. target_lex ASC (target_lex_cand)
                sort_keys = (np.array(target_lex_cand, dtype=np.int32), -b_count, -sc)
                order = np.lexsort(sort_keys)

                selected_eids = [target_eids_cand[i] for i in order[:400]]
                selected_300 = [target_eids_cand[i] for i in order[:min(300, len(order))]]
                selected_500 = [target_eids_cand[i] for i in order[:min(500, len(order))]]

            stage_cands_per_s1[st].append(len(selected_eids))
            for ceid in selected_eids:
                stage_pairs[st].add((s1_id, ceid))
            for ceid in selected_300:
                stage_pairs_k300[st].add((s1_id, ceid))
            for ceid in selected_500:
                stage_pairs_k500[st].add((s1_id, ceid))

            stage_runtimes[st] += (time.time() - t_st0)

        cur_ram = proc.memory_info().rss
        if cur_ram > peak_ram:
            peak_ram = cur_ram

        if (s1_idx + 1) % 100 == 0 or (s1_idx + 1) == len(s1_fr):
            print(f"  Processed {s1_idx+1}/{len(s1_fr)} S1 entities ({time.time()-t_start_ablation:.1f}s)...", flush=True)

    total_ablation_time = time.time() - t_start_ablation
    print(f"\nAblation completed in {total_ablation_time:.2f}s | Peak RAM: {peak_ram / 1024**2:.1f} MB")

    # Compute metrics for every stage
    stage_signals = [
        "Legacy",
        "+ Name Jaccard",
        "+ Name JW",
        "+ Name TF-IDF",
        "+ Address Jaccard",
        "+ Address TF-IDF",
        "+ Country",
        "+ Blocking Count",
        "+ Prefix",
        "+ Length"
    ]

    results_table = []
    base_tp = None

    for st in range(10):
        pairs = stage_pairs[st]
        tp = len(pairs & exact_matches_set)
        lost = TOTAL_GT_PAIRS - tp
        recall_400 = (tp / TOTAL_GT_PAIRS) * 100

        tp_300 = len(stage_pairs_k300[st] & exact_matches_set)
        tp_500 = len(stage_pairs_k500[st] & exact_matches_set)
        rec_300 = (tp_300 / TOTAL_GT_PAIRS) * 100
        rec_500 = (tp_500 / TOTAL_GT_PAIRS) * 100

        if st == 0:
            base_tp = tp
            delta_prev = "—"
            delta_base = "—"
        else:
            prev_tp = results_table[st - 1]['true_retained']
            d_p = tp - prev_tp
            d_b = tp - base_tp
            delta_prev = f"{d_p:+d}"
            delta_base = f"{d_b:+d}"

        row_metric = {
            "stage": st,
            "signals": stage_signals[st],
            "recall_400": round(recall_400, 4),
            "recall_300": round(rec_300, 4),
            "recall_500": round(rec_500, 4),
            "true_retained": tp,
            "true_lost": lost,
            "delta_vs_previous": delta_prev,
            "delta_vs_baseline": delta_base,
            "runtime_s": round(stage_runtimes[st], 2),
            "peak_ram_mb": round(peak_ram / 1024**2, 1),
            "candidates_per_s1": round(float(np.mean(stage_cands_per_s1[st])), 2),
            "cap_hits": stage_cap_hits[st],
        }
        results_table.append(row_metric)

    # Print Mandatory Output Table
    print("\n" + "=" * 95)
    print("MANDATORY PRE-RANKING ABLATION OUTPUT TABLE")
    print("=" * 95)
    print(f"| {'Stage':>5} | {'Signals':<20} | {'Recall@400':>10} | {'True Retained':>13} | {'Delta vs Prev':>14} | {'Delta vs Base':>14} | {'Runtime':>8} | {'Peak RAM':>10} |")
    print(f"|------:|:--------------------|-----------:|--------------:|--------------:|--------------:|--------:|---------:|")
    for r in results_table:
        print(f"| {r['stage']:5d} | {r['signals']:<20} | {r['recall_400']:9.4f}% | {r['true_retained']:13d} | {r['delta_vs_previous']:>14} | {r['delta_vs_baseline']:>14} | {r['runtime_s']:7.2f}s | {r['peak_ram_mb']:8.1f} MB |")
    print("=" * 95)

    # Verify Stage 0
    st0_tp = results_table[0]['true_retained']
    st0_rec = results_table[0]['recall_400']
    print(f"\nStage 0 Verification:")
    print(f"  Retained True Pairs: {st0_tp} (Expected: ~3,470)")
    print(f"  Recall@400:          {st0_rec:.4f}% (Expected: ~82.7965%)")
    assert abs(st0_rec - 82.7965) < 0.1, f"Stage 0 baseline mismatch: {st0_rec:.4f}% vs 82.7965%"

    # Identify best stage
    best_stage_row = max(results_table, key=lambda x: x['recall_400'])
    best_stage = best_stage_row['stage']
    print(f"\nBest Ranking Stage: Stage {best_stage} ({best_stage_row['signals']}) with Recall@400 = {best_stage_row['recall_400']:.4f}% ({best_stage_row['true_retained']} pairs)")
    print(f"  Recall@300: {best_stage_row['recall_300']:.4f}%")
    print(f"  Recall@400: {best_stage_row['recall_400']:.4f}%")
    print(f"  Recall@500: {best_stage_row['recall_500']:.4f}%")

    # Downstream Validation of the Best Stage vs Baseline
    print(f"\n[Step 5/6] Running Downstream XGBoost Validation ONLY for Stage {best_stage}...")
    model = xgb.XGBClassifier()
    model.load_model('models/xgboost_entity_resolution_phase5_configA.json')
    tfidf_models = fit_transductive_tfidf()

    best_pairs_list = list(stage_pairs[best_stage])
    df_pairs_best = pd.DataFrame({
        'entity_id_s1': [p[0] for p in best_pairs_list],
        'entity_id_cand': [p[1] for p in best_pairs_list],
        'source': ['S2' if p[1].startswith('S2') else 'S3' for p in best_pairs_list],
        'blocked_country': np.ones(len(best_pairs_list), dtype=np.int8),
        'blocked_name_token': np.ones(len(best_pairs_list), dtype=np.int8),
        'blocked_prefix_3': np.ones(len(best_pairs_list), dtype=np.int8),
        'blocked_prefix_4': np.ones(len(best_pairs_list), dtype=np.int8),
        'blocked_address_token': np.ones(len(best_pairs_list), dtype=np.int8),
        'num_blocking_keys': np.full(len(best_pairs_list), 4, dtype=np.int8),
        'evidence_score': np.full(len(best_pairs_list), 100.0, dtype=np.float32),
        'blocked_exact_name': [1 if s1_dict[p[0]]['norm_name'] == target_dict[p[1]]['norm_name'] else 0 for p in best_pairs_list],
        'blocked_selective_token': np.ones(len(best_pairs_list), dtype=np.int8),
        'blocked_rare_token': np.zeros(len(best_pairs_list), dtype=np.int8),
        'blocked_token_pair': np.zeros(len(best_pairs_list), dtype=np.int8),
        'blocked_char_ngram': np.zeros(len(best_pairs_list), dtype=np.int8),
        'shared_ngram_count': np.zeros(len(best_pairs_list), dtype=np.int8)
    })

    print(f"  Extracting 57 locked features for {len(df_pairs_best):,} candidates...", flush=True)
    df_features_best = build_vectorized_features(df_pairs_best, s1_dict, target_dict, tfidf_models=tfidf_models)

    # Check for NaN / Inf
    assert df_features_best[FEATURE_NAMES].isna().sum().sum() == 0, "NaN in features!"
    assert np.isinf(df_features_best[FEATURE_NAMES].values).sum() == 0, "Inf in features!"

    print("  Predicting model probabilities...", flush=True)
    probs = model.predict_proba(df_features_best[FEATURE_NAMES].values)[:, 1]
    df_pairs_best['probability'] = probs
    eid_to_lex = dict(zip(engine.target_eids, engine.target_lex_rank))
    df_pairs_best['target_lex_rank'] = [eid_to_lex[eid] for eid in df_pairs_best['entity_id_cand']]

    # Filter to probability >= 0.99
    preds_099 = df_pairs_best[df_pairs_best['probability'] >= 0.990].copy()
    preds_by_s1 = defaultdict(list)
    for row in preds_099.itertuples():
        preds_by_s1[row.entity_id_s1].append((row.entity_id_cand, float(row.probability), int(row.target_lex_rank)))

    for s1 in preds_by_s1:
        preds_by_s1[s1].sort(key=lambda x: (-x[1], x[2]))

    # Apply gap <= 0.0001 stopping rule
    final_pred_pairs = set()
    for s1_id in val_s1_ids:
        c_list = preds_by_s1.get(s1_id, [])
        if not c_list:
            continue
        final_pred_pairs.add((s1_id, c_list[0][0]))
        for idx in range(len(c_list) - 1):
            if (c_list[idx][1] - c_list[idx + 1][1]) <= 0.0001:
                final_pred_pairs.add((s1_id, c_list[idx + 1][0]))
            else:
                break

    # Compute downstream metrics
    macro = compute_macro_f05(val_s1_ids, exact_matches_set, final_pred_pairs)
    tp = len(final_pred_pairs & exact_matches_set)
    fp = len(final_pred_pairs - exact_matches_set)
    fn = len(exact_matches_set - final_pred_pairs)
    p_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    p_rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    sing_tp = len({p for p in final_pred_pairs if p[0] in set(singleton_s1_ids)} & exact_matches_set)
    multi_tp = len({p for p in final_pred_pairs if p[0] in set(multi_match_s1_ids)} & exact_matches_set)
    pred_counts = Counter(p[0] for p in final_pred_pairs)
    zero_preds = sum(1 for s in zero_match_s1_ids if pred_counts[s] > 0)

    downstream_metrics = {
        "macro_f0_5": round(macro['macro_f0_5'], 6),
        "macro_precision": round(macro['macro_precision'], 6),
        "macro_recall": round(macro['macro_recall'], 6),
        "pairwise_precision": round(p_prec, 6),
        "pairwise_recall": round(p_rec, 6),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "predicted_pairs": len(final_pred_pairs),
        "singleton_tp": sing_tp,
        "multi_match_tp": multi_tp,
        "zero_match_with_predictions": zero_preds,
    }

    print("\nDownstream Pipeline Validation Results:")
    print(f"  Macro F0.5:      {downstream_metrics['macro_f0_5']:.6f} (Baseline: 0.317727)")
    print(f"  Macro Precision: {downstream_metrics['macro_precision']*100:.3f}% (Baseline: 32.602%)")
    print(f"  Macro Recall:    {downstream_metrics['macro_recall']*100:.3f}% (Baseline: 70.270%)")
    print(f"  TP: {tp} (Baseline: 1497), FP: {fp} (Baseline: 3995), FN: {fn} (Baseline: 2694)")
    print(f"  Singleton TP: {sing_tp} (Baseline: 111), Multi-Match TP: {multi_tp} (Baseline: 1386)")

    # 6. Save JSON results
    out_json = {
        "benchmark": {
            "total_s1": len(s1_fr),
            "total_targets": len(targets),
            "reference_true_pairs": TOTAL_GT_PAIRS,
            "raw_candidates_total": total_raw_candidates,
            "production_fingerprint": fp
        },
        "scale_inspection": scale_samples,
        "stage_results": results_table,
        "best_stage": best_stage_row,
        "baseline_metrics": {
            "macro_f0_5": 0.317727,
            "macro_precision": 0.32602,
            "macro_recall": 0.70270,
            "tp": 1497,
            "fp": 3995,
            "fn": 2694,
            "singleton_tp": 111,
            "multi_match_tp": 1386
        },
        "downstream_metrics_best_stage": downstream_metrics
    }

    os.makedirs('experiments', exist_ok=True)
    json_path = 'experiments/preranking_ablation_results.json'
    with open(json_path, 'w') as f:
        json.dump(out_json, f, indent=2)
    print(f"\nSaved results to {json_path}")

    return out_json


if __name__ == '__main__':
    run_ablation()
