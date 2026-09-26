"""
PHASE 7: COMPOSITE PRE-RANKER @ FIXED CAP 400 VALIDATION EXPERIMENT

Evaluates whether a multi-component composite pre-ranker at fixed cap 400
improves the downstream final model metric (Macro F0.5) over the legacy evidence ranker,
using the locked Phase 5 XGBoost model and threshold 0.910 on the 500-S1 France benchmark.
"""

import sys, os, time, gc, json, threading, math, heapq
from collections import defaultdict
sys.path.insert(0, 'src')
sys.path.insert(0, '.')
import psutil
import pandas as pd
import numpy as np
import xgboost as xgb

from normalize import normalize_business_name, normalize_country, normalize_business_address
from blocking import ConfigABlockingEngine
from config import BLOCKING_CHUNK_SIZE, get_config_fingerprint
from features import build_vectorized_features, FEATURE_NAMES
from tfidf_model import TransductiveTFIDF
from metrics import compute_macro_f05, compute_pairwise_metrics

BIT_COUNTRY = 1 << 0
BIT_NAME = 1 << 1
BIT_P3 = 1 << 2
BIT_P4 = 1 << 3
BIT_ADDR = 1 << 4


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


def generate_composite_preranked_chunk(
    engine,
    df_s1_chunk,
    s1_orig_indices,
    s1_lex_ranks,
    max_candidates_per_s1=400,
    addr_df_cap=50000,
    high_df_postings_sets=None
):
    s1_names = df_s1_chunk['norm_name'].values if 'norm_name' in df_s1_chunk.columns else np.array([''] * len(df_s1_chunk))
    s1_addrs = df_s1_chunk['norm_address'].values if 'norm_address' in df_s1_chunk.columns else np.array([''] * len(df_s1_chunk))
    s1_countries = df_s1_chunk['norm_country'].values if 'norm_country' in df_s1_chunk.columns else np.array([''] * len(df_s1_chunk))

    out_s1_idx = []
    out_s1_lex = []
    out_cand_idx = []
    out_cand_lex = []
    out_mask = []
    out_composite_score = []
    out_exact = []

    raw_counts_chunk = {}
    names_target = engine.target_names
    target_addrs = engine.target_addrs
    target_lex = engine.target_lex_rank
    log_N = math.log(max(engine.n_target, 2))

    country_name_tokens_all = engine.country_name_token_idx
    country_p3_all = engine.country_prefix3_idx
    country_p4_all = engine.country_prefix4_idx
    country_addr_tokens_all = engine.country_addr_token_idx

    for s1_idx_in_chunk in range(len(df_s1_chunk)):
        s1_orig = s1_orig_indices[s1_idx_in_chunk]
        s1_lex = s1_lex_ranks[s1_idx_in_chunk]

        name = str(s1_names[s1_idx_in_chunk])
        addr = str(s1_addrs[s1_idx_in_chunk])
        country = str(s1_countries[s1_idx_in_chunk])

        s1_tokens = [t for t in name.split() if len(t) > 1 and t not in engine.stop_tokens]
        s1_p3 = name[:3] if len(name) >= 3 else ''
        s1_p4 = name[:4] if len(name) >= 4 else ''
        s1_addr_tokens = set([t for t in addr.split() if len(t) >= 4 and t not in engine.stop_addrs]) if addr else set()

        cand_flags = {}
        cand_name_hits = {}
        cand_addr_hits = {}
        cand_rarity = {}

        country_name_tokens = country_name_tokens_all[country]
        country_p3 = country_p3_all[country]
        country_p4 = country_p4_all[country]
        country_addr_tokens = country_addr_tokens_all[country]

        # Channel 2: Country + Name Token (selective)
        for t in s1_tokens:
            if t in country_name_tokens:
                postings = country_name_tokens[t]
                df_t = len(postings)
                rarity_t = max(0.0, 1.0 - math.log(max(df_t, 1)) / log_N)
                for idx in postings:
                    cand_flags[idx] = cand_flags.get(idx, 0) | BIT_NAME
                    cand_name_hits[idx] = cand_name_hits.get(idx, 0) + 1
                    if rarity_t > cand_rarity.get(idx, 0.0):
                        cand_rarity[idx] = rarity_t

        # Channel 3: Country + Prefix 4
        if s1_p4 and s1_p4 in country_p4:
            postings = country_p4[s1_p4]
            df_p4 = len(postings)
            rarity_p4 = max(0.0, 1.0 - math.log(max(df_p4, 1)) / log_N)
            for idx in postings:
                cand_flags[idx] = cand_flags.get(idx, 0) | BIT_P4
                if rarity_p4 > cand_rarity.get(idx, 0.0):
                    cand_rarity[idx] = rarity_p4

        # Channel 4: Country + Prefix 3
        if s1_p3 and s1_p3 in country_p3:
            postings = country_p3[s1_p3]
            df_p3 = len(postings)
            rarity_p3 = max(0.0, 1.0 - math.log(max(df_p3, 1)) / log_N)
            for idx in postings:
                cand_flags[idx] = cand_flags.get(idx, 0) | BIT_P3
                if rarity_p3 > cand_rarity.get(idx, 0.0):
                    cand_rarity[idx] = rarity_p3

        # Channel 5: Country + Address Token
        low_df_tokens = [t for t in s1_addr_tokens if t in country_addr_tokens and len(country_addr_tokens[t]) <= addr_df_cap]
        for atok in low_df_tokens:
            postings = country_addr_tokens[atok]
            df_a = len(postings)
            rarity_a = max(0.0, 1.0 - math.log(max(df_a, 1)) / log_N)
            for idx in postings:
                cand_flags[idx] = cand_flags.get(idx, 0) | BIT_ADDR
                cand_addr_hits[idx] = cand_addr_hits.get(idx, 0) + 1
                if rarity_a > cand_rarity.get(idx, 0.0):
                    cand_rarity[idx] = rarity_a

        # High-DF tokens: award address evidence and rarity to candidates already generated
        if high_df_postings_sets:
            s1_high_tokens = [t for t in s1_addr_tokens if t in high_df_postings_sets]
            for ht in s1_high_tokens:
                postings_set = high_df_postings_sets[ht]
                df_ht = len(country_addr_tokens[ht])
                rarity_ht = max(0.0, 1.0 - math.log(max(df_ht, 1)) / log_N)
                for idx in list(cand_flags.keys()):
                    if idx in postings_set:
                        cand_flags[idx] = cand_flags[idx] | BIT_ADDR
                        cand_addr_hits[idx] = cand_addr_hits.get(idx, 0) + 1
                        if rarity_ht > cand_rarity.get(idx, 0.0):
                            cand_rarity[idx] = rarity_ht

        n_cands = len(cand_flags)
        raw_counts_chunk[s1_orig] = n_cands

        if n_cands == 0:
            continue

        # COMPUTE COMPOSITE SCORES FOR ALL CANDIDATES OF THIS S1
        len_name_s1 = len(name)
        len_addr_s1 = len(addr)
        n_s1_tokens = max(len(s1_tokens), 1)
        n_s1_addrs = max(len(s1_addr_tokens), 1)

        composite_scores = {}
        for idx, flags in cand_flags.items():
            c_name = str(names_target[idx])
            c_addr = str(target_addrs[idx])

            # 1. Name score [0, 1]
            if name and c_name == name:
                name_score = 1.0
            else:
                tok_ratio = min(cand_name_hits.get(idx, 0) / n_s1_tokens, 1.0)
                has_p4 = 1.0 if (flags & BIT_P4) else 0.0
                has_p3 = 1.0 if (flags & BIT_P3) else 0.0
                name_score = min(0.70 * tok_ratio + 0.20 * has_p4 + 0.10 * has_p3, 1.0)

            # 2. Address score [0, 1] (Strictly bounded, not cumulative)
            if len(s1_addr_tokens) > 0:
                address_score = min(cand_addr_hits.get(idx, 0) / n_s1_addrs, 1.0)
            else:
                address_score = 0.0

            # 3. Rarity score [0, 1]
            rarity_score = cand_rarity.get(idx, 0.0)

            # 4. Country score [0, 1]
            country_score = 1.0

            # 5. Structural score [0, 1]
            len_c_name = len(c_name)
            name_len_ratio = min(len_name_s1, len_c_name) / max(len_name_s1, len_c_name, 1)
            if len_addr_s1 > 0 and len(c_addr) > 0:
                addr_len_ratio = min(len_addr_s1, len(c_addr)) / max(len_addr_s1, len(c_addr), 1)
                structural_score = 0.6 * name_len_ratio + 0.4 * addr_len_ratio
            else:
                structural_score = name_len_ratio

            # Composite pre-ranker formula:
            comp_score = (
                0.50 * name_score
                + 0.20 * address_score
                + 0.15 * rarity_score
                + 0.10 * country_score
                + 0.05 * structural_score
            )
            composite_scores[idx] = comp_score

        # Top-K selection with deterministic ordering:
        # 1. composite_score DESC (-comp_score)
        # 2. target_lex_rank ASC (target_lex[idx])
        K = max_candidates_per_s1
        if n_cands <= K:
            selected_indices = sorted(cand_flags.keys(), key=lambda i: (-composite_scores[i], target_lex[i]))
        else:
            selected_indices = heapq.nsmallest(K, cand_flags.keys(), key=lambda i: (-composite_scores[i], target_lex[i]))

        for idx in selected_indices:
            out_s1_idx.append(s1_orig)
            out_s1_lex.append(s1_lex)
            out_cand_idx.append(idx)
            out_cand_lex.append(target_lex[idx])
            out_mask.append(cand_flags[idx] | BIT_COUNTRY)
            out_composite_score.append(composite_scores[idx])
            out_exact.append(1 if names_target[idx] == name and name != '' else 0)

        del cand_flags, cand_name_hits, cand_addr_hits, cand_rarity, composite_scores

    N_out = len(out_s1_idx)
    if N_out == 0:
        chunk_df = pd.DataFrame(columns=[
            's1_orig_idx', 's1_lex_rank', 'cand_orig_idx', 'cand_lex_rank',
            'bitmask', 'composite_score', 'blocked_exact_name'
        ])
    else:
        chunk_df = pd.DataFrame({
            's1_orig_idx': np.array(out_s1_idx, dtype=np.int32),
            's1_lex_rank': np.array(out_s1_lex, dtype=np.int32),
            'cand_orig_idx': np.array(out_cand_idx, dtype=np.int32),
            'cand_lex_rank': np.array(out_cand_lex, dtype=np.int32),
            'bitmask': np.array(out_mask, dtype=np.int8),
            'composite_score': np.array(out_composite_score, dtype=np.float32),
            'blocked_exact_name': np.array(out_exact, dtype=np.int8)
        })

    return chunk_df, raw_counts_chunk


def generate_composite_preranked_candidates(
    engine,
    df_s1,
    max_candidates_per_s1=400,
    chunk_size=BLOCKING_CHUNK_SIZE,
    addr_df_cap=50000,
    high_df_postings_sets=None
):
    n_s1 = len(df_s1)
    s1_ids_all = df_s1['entity_id'].values

    # Deterministic integer lexical rank for S1
    order_s1 = np.argsort(s1_ids_all)
    s1_lex_ranks = np.empty(n_s1, dtype=np.int32)
    s1_lex_ranks[order_s1] = np.arange(n_s1, dtype=np.int32)

    chunks = []
    total_raw_candidates = 0
    raw_counts_per_s1 = {}

    for start_idx in range(0, n_s1, chunk_size):
        end_idx = min(start_idx + chunk_size, n_s1)
        sub_df = df_s1.iloc[start_idx:end_idx]
        orig_indices = np.arange(start_idx, end_idx, dtype=np.int32)
        chunk_lex_ranks = s1_lex_ranks[start_idx:end_idx]

        chunk_res, chunk_raw_counts = generate_composite_preranked_chunk(
            engine,
            sub_df,
            orig_indices,
            chunk_lex_ranks,
            max_candidates_per_s1=max_candidates_per_s1,
            addr_df_cap=addr_df_cap,
            high_df_postings_sets=high_df_postings_sets
        )

        total_raw_candidates += sum(chunk_raw_counts.values())
        raw_counts_per_s1.update(chunk_raw_counts)

        if len(chunk_res) > 0:
            chunks.append(chunk_res)
        del chunk_res
        gc.collect()

    if chunks:
        capped_df = pd.concat(chunks, ignore_index=True)
        # Sort output deterministically: (s1_lex_rank ASC, cand_lex_rank ASC)
        capped_df.sort_values(by=['s1_lex_rank', 'cand_lex_rank'], ascending=[True, True], inplace=True)
        capped_df.reset_index(drop=True, inplace=True)

        N = len(capped_df)
        masks = capped_df['bitmask'].values
        b_ctry = (masks & BIT_COUNTRY).astype(np.int8)
        b_name = ((masks & BIT_NAME) >> 1).astype(np.int8)
        b_p3 = ((masks & BIT_P3) >> 2).astype(np.int8)
        b_p4 = ((masks & BIT_P4) >> 3).astype(np.int8)
        b_addr = ((masks & BIT_ADDR) >> 4).astype(np.int8)
        num_keys = (b_name + b_p3 + b_p4 + b_addr).astype(np.int8)

        cand_orig = capped_df['cand_orig_idx'].values
        final_df = pd.DataFrame({
            'entity_id_s1': s1_ids_all[capped_df['s1_orig_idx'].values],
            'entity_id_cand': engine.target_eids[cand_orig],
            'source': engine.target_sources[cand_orig],
            'blocked_country': b_ctry,
            'blocked_name_token': b_name,
            'blocked_prefix_3': b_p3,
            'blocked_prefix_4': b_p4,
            'blocked_address_token': b_addr,
            'num_blocking_keys': num_keys,
            'evidence_score': capped_df['composite_score'].values * 100.0,
            'composite_score': capped_df['composite_score'].values,
            'blocked_exact_name': capped_df['blocked_exact_name'].values,
            'blocked_selective_token': b_name,
            'blocked_rare_token': np.zeros(N, dtype=np.int8),
            'blocked_token_pair': np.zeros(N, dtype=np.int8),
            'blocked_char_ngram': np.zeros(N, dtype=np.int8),
            'shared_ngram_count': np.zeros(N, dtype=np.int8)
        })
    else:
        final_df = engine.generate_candidates_for_chunk(df_s1.iloc[:0])

    counts_arr = np.array(list(raw_counts_per_s1.values())) if raw_counts_per_s1 else np.array([0])
    metrics = {
        'raw_candidate_count': int(total_raw_candidates),
        'post_cap_candidate_count': len(final_df),
        'candidates_per_s1': float(np.mean(counts_arr)),
        'median_candidates_per_s1': float(np.median(counts_arr)),
        'max_candidates_per_s1': int(np.max(counts_arr)),
        'min_candidates_per_s1': int(np.min(counts_arr)),
    }
    return final_df, metrics


def run_experiment():
    print("=" * 85, flush=True)
    print("PHASE 7: COMPOSITE PRE-RANKER @ FIXED CAP 400 EXPERIMENT", flush=True)
    print("=" * 85, flush=True)

    # Sanity checks
    print("\n[Step 1/5] Verifying Locked Artifacts & Contracts...", flush=True)

    model_path = 'models/xgboost_entity_resolution_phase5_configA.json'
    assert os.path.exists(model_path), f"Missing model: {model_path}"
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    print(f"  [1/9] Phase 5 XGBoost model loaded from {model_path}", flush=True)

    schema_path = 'models/phase5_configA_feature_schema.json'
    with open(schema_path, 'r') as f:
        schema = json.load(f)
    assert len(schema['feature_names']) == 57, f"Expected 57 features, got {len(schema['feature_names'])}"
    assert schema['feature_names'] == FEATURE_NAMES, "Feature names mismatch!"
    print(f"  [2/9] Feature schema verified: exactly 57 features", flush=True)

    thresh_path = 'models/phase5_configA_threshold.json'
    with open(thresh_path, 'r') as f:
        thresh_data = json.load(f)
    locked_threshold = float(thresh_data['threshold'])
    assert locked_threshold == 0.910, f"Expected threshold 0.910, got {locked_threshold}"
    print(f"  [3/9] Decision threshold verified: {locked_threshold}", flush=True)

    fp = get_config_fingerprint()
    assert fp == "87f20ceeb84ccc6ea2d48678c7810ac5", f"Fingerprint mismatch: {fp}"
    print(f"  [4/9] Production config fingerprint verified: {fp}", flush=True)

    s1_df = pd.read_csv('data/test/test_source1.tsv', sep='\t', dtype=str, nrows=10000, na_filter=False)
    s1_fr = s1_df[s1_df['country'].apply(normalize_country) == 'france'].head(500).copy().reset_index(drop=True)
    del s1_df
    gc.collect()
    s1_fr['norm_name'] = s1_fr['business_name'].apply(normalize_business_name)
    s1_fr['norm_address'] = s1_fr['business_address'].apply(normalize_business_address)
    s1_fr['norm_country'] = 'france'
    val_s1_ids = list(s1_fr['entity_id'].values)
    assert len(val_s1_ids) == 500, f"Expected 500 S1 entities, got {len(val_s1_ids)}"
    print(f"  [5/9] Validation set verified: 500 France S1 entities", flush=True)

    targets = pd.read_pickle('scratch/targets_france.pkl')
    n_targets = len(targets)
    print(f"  [6/9] Target pool loaded: {n_targets:,} France records (S2 + S3)", flush=True)

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
    print(f"  [7/9] Ground truth reference set verified: {len(exact_matches_set):,} true pairs", flush=True)
    assert len(exact_matches_set) == 4191, f"Expected 4191 true pairs, got {len(exact_matches_set)}"

    s1_dict = s1_fr.set_index('entity_id').to_dict(orient='index')
    print(f"  [8/9] S1 lookup dict built: {len(s1_dict):,} records", flush=True)

    print(f"  [9/9] Building ConfigABlockingEngine...", flush=True)
    engine = ConfigABlockingEngine(targets)
    print("All Sanity Checks PASSED cleanly!\n", flush=True)

    # Fit TF-IDF
    tfidf_models = fit_transductive_tfidf()

    # Precompute high-DF address tokens (> 50,000)
    addr_cap = 50000
    country_addr_tokens = engine.country_addr_token_idx['france']
    high_tokens = {k for k, v in country_addr_tokens.items() if len(v) > addr_cap}
    high_df_postings_sets = {k: set(country_addr_tokens[k]) for k in high_tokens}
    print(f"Precomputed {len(high_tokens)} high-DF address token sets (> {addr_cap:,})", flush=True)

    s1_with_gt = {p[0] for p in exact_matches_set}
    gt_zero_match_s1_count = len(set(val_s1_ids) - s1_with_gt)
    print(f"Ground-Truth S1 distribution: {len(s1_with_gt)} S1 entities with true matches, {gt_zero_match_s1_count} zero-match S1 entities", flush=True)

    # BASELINE VALUES (from CAP-400 experiment)
    baseline = {
        "config": "CAP-400 (Legacy Evidence Ranker)",
        "cap": 400,
        "raw_candidates": 95603816,
        "raw_candidate_recall": 1.0,
        "post_cap_candidates": 200000,
        "post_cap_candidate_recall": 0.827965,
        "true_pairs_retained_in_candidates": 3470,
        "true_pairs_lost_in_candidates": 721,
        "macro_f0_5": 0.077903,
        "macro_precision": 0.067123,
        "macro_recall": 0.863008,
        "tp": 2601,
        "fp": 32930,
        "fn": 1590,
        "pairwise_precision": 0.073204,
        "pairwise_recall": 0.620616,
        "pairwise_f0_5": 0.088884,
        "predicted_matched_s1_count": 500,
        "predicted_zero_match_s1_count": 0,
        "total_runtime_s": 320.49,
        "peak_rss_mb": 1252.5
    }

    # Load baseline predictions if available, or compute on the fly
    baseline_predictions_file = 'scratch/baseline_cap400_predictions.pkl'
    if os.path.exists(baseline_predictions_file):
        baseline_predicted_pairs = pd.read_pickle(baseline_predictions_file)
        print(f"Loaded {len(baseline_predicted_pairs):,} baseline predictions from cache", flush=True)
    else:
        print("\n>>> Generating Baseline CAP-400 predictions for exact set diff...", flush=True)
        from scratch.run_experiment_opt_e7 import generate_bounded_candidates_e7
        base_df, _ = generate_bounded_candidates_e7(
            engine,
            s1_fr,
            max_candidates_per_s1=400,
            chunk_size=BLOCKING_CHUNK_SIZE,
            addr_df_cap=addr_cap,
            high_df_postings_sets=high_df_postings_sets,
            use_fallback=False
        )
        base_targets = targets[targets['entity_id'].isin(set(base_df['entity_id_cand'].unique()))].set_index('entity_id').to_dict(orient='index')
        base_feats = []
        for b_idx in range(0, len(base_df), 25000):
            cf = build_vectorized_features(base_df.iloc[b_idx:b_idx+25000], s1_dict, base_targets, tfidf_models=tfidf_models)
            base_feats.append(cf[FEATURE_NAMES])
        base_df_feat = pd.concat(base_feats, ignore_index=True)
        base_probs = model.predict_proba(base_df_feat.values)[:, 1]
        base_match = (base_probs >= locked_threshold)
        baseline_predicted_pairs = set(zip(base_df['entity_id_s1'].values[base_match], base_df['entity_id_cand'].values[base_match]))
        pd.to_pickle(baseline_predicted_pairs, baseline_predictions_file)
        del base_df, base_targets, base_feats, base_df_feat, base_probs, base_match
        gc.collect()
        print(f"Generated and cached {len(baseline_predicted_pairs):,} baseline predictions", flush=True)

    print("\n" + "=" * 85, flush=True)
    print("RUNNING COMPOSITE PRE-RANKER @ CAP-400", flush=True)
    print("=" * 85, flush=True)

    t0_total = time.time()
    mem_tracker = PeakMemoryTracker(interval=0.05)
    mem_tracker.start()

    # Step A: Blocking with Composite Pre-Ranker
    t0_block = time.time()
    capped_df, metrics = generate_composite_preranked_candidates(
        engine,
        s1_fr,
        max_candidates_per_s1=400,
        chunk_size=BLOCKING_CHUNK_SIZE,
        addr_df_cap=addr_cap,
        high_df_postings_sets=high_df_postings_sets
    )
    t_block = time.time() - t0_block
    n_cands = len(capped_df)
    print(f"  [Blocking + Pre-Ranker] Generated {n_cands:,} capped candidates in {t_block:.2f}s", flush=True)

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

    print(f"  [Candidate Stats] Raw: {raw_cands:,} | Post-cap: {n_cands:,} | Recall: {candidate_recall * 100:.2f}% | Retained: {true_pairs_retained:,} | Lost: {true_pairs_lost:,}", flush=True)

    # Step B: Feature Extraction
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

    # Step C: Model Scoring
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

    # Step D: Metric Computation
    macro_metrics = compute_macro_f05(val_s1_ids, exact_matches_set, predicted_pairs)
    macro_f05 = macro_metrics['macro_f0_5']
    macro_prec = macro_metrics['macro_precision']
    macro_rec = macro_metrics['macro_recall']
    pred_zero_match_s1 = macro_metrics['zero_match_s1_count']
    pred_matched_s1 = macro_metrics['one_or_more_match_s1_count']

    tp = len(predicted_pairs & exact_matches_set)
    fp = len(predicted_pairs - exact_matches_set)
    fn = len(exact_matches_set - predicted_pairs)
    p_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    p_rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    p_denom = (0.25 * p_prec) + p_rec
    pairwise_f05 = (1.25 * p_prec * p_rec) / p_denom if p_denom > 0 else 0.0

    print(f"\n[RESULTS] Macro F0.5 = {macro_f05:.6f} | Prec = {macro_prec * 100:.4f}% | Rec = {macro_rec * 100:.4f}% | TP = {tp} | FP = {fp} | FN = {fn}", flush=True)

    # Compare with baseline
    delta_f05 = macro_f05 - baseline['macro_f0_5']
    delta_prec = macro_prec - baseline['macro_precision']
    delta_rec = macro_rec - baseline['macro_recall']
    delta_tp = tp - baseline['tp']
    delta_fp = fp - baseline['fp']
    delta_fn = fn - baseline['fn']

    # Exact differences vs baseline predictions if we generate baseline pairs
    # Let's save composite predicted pairs to scratch
    pd.to_pickle(predicted_pairs, 'scratch/composite_cap400_predictions.pkl')

    # Load or generate baseline predictions for exact set difference
    # If baseline predictions file does not exist, let's generate it quickly or load from cap experiment
    new_predictions_sample = []
    dropped_predictions_sample = []
    new_preds_count = 0
    dropped_preds_count = 0
    new_tp_count = 0
    new_fp_count = 0
    dropped_tp_count = 0
    dropped_fp_count = 0

    if baseline_predicted_pairs is not None:
        new_preds = predicted_pairs - baseline_predicted_pairs
        dropped_preds = baseline_predicted_pairs - predicted_pairs
        new_preds_count = len(new_preds)
        dropped_preds_count = len(dropped_preds)
        new_tp_count = len(new_preds & exact_matches_set)
        new_fp_count = len(new_preds - exact_matches_set)
        dropped_tp_count = len(dropped_preds & exact_matches_set)
        dropped_fp_count = len(dropped_preds - exact_matches_set)
        new_predictions_sample = list(list(p) for p in sorted(list(new_preds))[:15])
        dropped_predictions_sample = list(list(p) for p in sorted(list(dropped_preds))[:15])

    results_data = {
        "benchmark_summary": {
            "s1_count": 500,
            "target_count": n_targets,
            "config_fingerprint": fp,
            "locked_model": model_path,
            "locked_threshold": locked_threshold,
            "reference_true_pairs": len(exact_matches_set),
            "ground_truth_zero_match_s1_count": gt_zero_match_s1_count
        },
        "baseline_cap400": baseline,
        "composite_preranker_cap400": {
            "config": "COMPOSITE-PRERANKER-CAP400",
            "cap": 400,
            "raw_candidates": raw_cands,
            "raw_candidate_recall": raw_recall,
            "post_ranker_candidates": n_cands,
            "post_ranker_candidate_recall": round(candidate_recall, 6),
            "average_candidates_per_s1": round(mean_cands_s1, 2),
            "median_candidates_per_s1": round(median_cands_s1, 2),
            "max_candidates_per_s1": max_cands_s1,
            "true_pairs_retained": true_pairs_retained,
            "true_pairs_lost": true_pairs_lost,
            "macro_f0_5": round(macro_f05, 6),
            "macro_precision": round(macro_prec, 6),
            "macro_recall": round(macro_rec, 6),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "pairwise_precision": round(p_prec, 6),
            "pairwise_recall": round(p_rec, 6),
            "pairwise_f0_5": round(pairwise_f05, 6),
            "predicted_matched_s1_count": pred_matched_s1,
            "predicted_zero_match_s1_count": pred_zero_match_s1,
            "ground_truth_zero_match_s1_count": gt_zero_match_s1_count,
            "blocking_runtime_s": round(t_block, 2),
            "feature_runtime_s": round(t_feat, 2),
            "model_runtime_s": round(t_model, 2),
            "total_runtime_s": round(t_total, 2),
            "throughput_s1_per_sec": round(throughput, 2),
            "peak_rss_mb": round(peak_mb, 1),
            "delta_macro_f0_5": round(delta_f05, 6),
            "delta_macro_precision": round(delta_prec, 6),
            "delta_macro_recall": round(delta_rec, 6),
            "delta_tp": delta_tp,
            "delta_fp": delta_fp,
            "delta_fn": delta_fn
        },
        "prediction_differences": {
            "new_predictions_count": new_preds_count,
            "new_predictions_tp_count": new_tp_count,
            "new_predictions_fp_count": new_fp_count,
            "dropped_predictions_count": dropped_preds_count,
            "dropped_predictions_tp_count": dropped_tp_count,
            "dropped_predictions_fp_count": dropped_fp_count,
            "new_predictions_sample": new_predictions_sample,
            "dropped_predictions_sample": dropped_predictions_sample
        }
    }

    out_json = 'experiments/composite_preranker_cap400_results.json'
    with open(out_json, 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"\nSaved results to {out_json}", flush=True)

    return results_data


if __name__ == '__main__':
    run_experiment()
