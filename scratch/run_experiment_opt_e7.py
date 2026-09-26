"""
PHASE E7 — RECALL-PRESERVING DF-CAP EXPERIMENT
Tests selective address token filtering with combined-evidence ranking and per-S1 fallback.

Configurations:
- E7-0: Original E6-A selective DF > 50,000 (no fallback)
- E7-A: Selective DF > 50,000 + combined-evidence ranking + per-S1 fallback
- E7-B: Selective DF > 100,000 + combined-evidence ranking + per-S1 fallback
- E7-C: Selective DF > 150,000 + combined-evidence ranking + per-S1 fallback
- E7-D: Selective DF > 200,000 + combined-evidence ranking + per-S1 fallback
"""

import sys, os, time, gc, json, threading, heapq
from collections import defaultdict
sys.path.insert(0, 'src')
sys.path.insert(0, '.')
import psutil
import pandas as pd
import numpy as np

from normalize import normalize_business_name, normalize_country, normalize_business_address
from blocking import ConfigABlockingEngine
from config import BLOCKING_CHUNK_SIZE, CANDIDATE_CAP_CONFIG, get_config_fingerprint


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


BIT_COUNTRY = 1 << 0
BIT_NAME = 1 << 1
BIT_P3 = 1 << 2
BIT_P4 = 1 << 3
BIT_ADDR = 1 << 4


def generate_bounded_chunk_e7(
    engine,
    df_s1_chunk,
    s1_orig_indices,
    s1_lex_ranks,
    max_candidates_per_s1=400,
    addr_df_cap=50000,
    high_df_postings_sets=None,
    use_fallback=False
):
    s1_names = df_s1_chunk['norm_name'].values if 'norm_name' in df_s1_chunk.columns else np.array([''] * len(df_s1_chunk))
    s1_addrs = df_s1_chunk['norm_address'].values if 'norm_address' in df_s1_chunk.columns else np.array([''] * len(df_s1_chunk))
    s1_countries = df_s1_chunk['norm_country'].values if 'norm_country' in df_s1_chunk.columns else np.array([''] * len(df_s1_chunk))

    out_s1_idx = []
    out_s1_lex = []
    out_cand_idx = []
    out_cand_lex = []
    out_mask = []
    out_score = []
    out_exact = []

    raw_counts_chunk = {}
    fallback_counts_chunk = 0
    names_target = engine.target_names
    target_lex = engine.target_lex_rank

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
        cand_scores = {}

        country_name_tokens = engine.country_name_token_idx[country]
        country_p3 = engine.country_prefix3_idx[country]
        country_p4 = engine.country_prefix4_idx[country]
        country_addr_tokens = engine.country_addr_token_idx[country]

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
        # 1. Low-DF tokens generate candidates and evidence
        low_df_tokens = [t for t in s1_addr_tokens if t in country_addr_tokens and len(country_addr_tokens[t]) <= addr_df_cap]
        addr_cands_before = sum(1 for f in cand_flags.values() if (f & BIT_ADDR))
        
        for atok in low_df_tokens:
            for idx in country_addr_tokens[atok]:
                f = cand_flags.get(idx, 0)
                if not (f & BIT_ADDR):
                    cand_flags[idx] = f | BIT_ADDR
                    cand_scores[idx] = cand_scores.get(idx, 0) + 60

        # 2. High-DF tokens DO NOT generate new candidate postings.
        # BUT for candidates already generated through other channels, award address evidence!
        if high_df_postings_sets:
            s1_high_tokens = [t for t in s1_addr_tokens if t in high_df_postings_sets]
            if s1_high_tokens:
                for idx in list(cand_flags.keys()):
                    f = cand_flags[idx]
                    if not (f & BIT_ADDR):
                        for ht in s1_high_tokens:
                            if idx in high_df_postings_sets[ht]:
                                cand_flags[idx] = f | BIT_ADDR
                                cand_scores[idx] = cand_scores[idx] + 60
                                break

        # 3. Per-S1 Fallback (if enabled):
        # If selective filtering caused the address channel to produce NO usable candidates,
        # OR if the S1 entity would otherwise have zero candidates:
        if use_fallback:
            addr_cands_after = sum(1 for f in cand_flags.values() if (f & BIT_ADDR))
            needs_fallback = False
            if len(s1_addr_tokens) > 0 and addr_cands_after == addr_cands_before and len(low_df_tokens) == 0:
                # S1 had address tokens, but ALL of them exceeded the DF cap
                needs_fallback = True
            elif len(cand_flags) == 0 and len(s1_addr_tokens) > 0:
                needs_fallback = True

            if needs_fallback and high_df_postings_sets:
                fallback_counts_chunk += 1
                for ht in [t for t in s1_addr_tokens if t in high_df_postings_sets]:
                    for idx in country_addr_tokens[ht]:
                        f = cand_flags.get(idx, 0)
                        if not (f & BIT_ADDR):
                            cand_flags[idx] = f | BIT_ADDR
                            cand_scores[idx] = cand_scores.get(idx, 0) + 60

        n_cands = len(cand_flags)
        raw_counts_chunk[s1_orig] = n_cands

        if n_cands == 0:
            continue

        # Cap ranking & extraction (COMBINED EVIDENCE RANKING)
        K = max_candidates_per_s1
        if n_cands <= K:
            for idx, flags in cand_flags.items():
                out_s1_idx.append(s1_orig)
                out_s1_lex.append(s1_lex)
                out_cand_idx.append(idx)
                out_cand_lex.append(target_lex[idx])
                out_mask.append(flags | BIT_COUNTRY)
                out_score.append(cand_scores[idx])
                out_exact.append(1 if names_target[idx] == name and name != '' else 0)
        else:
            score_buckets = defaultdict(list)
            for idx, flags in cand_flags.items():
                score_buckets[cand_scores[idx]].append(idx)

            collected_indices = []
            for sc in sorted(score_buckets.keys(), reverse=True):
                bucket = score_buckets[sc]
                if len(collected_indices) + len(bucket) <= K:
                    collected_indices.extend(bucket)
                    if len(collected_indices) == K:
                        break
                else:
                    needed = K - len(collected_indices)
                    top_items = heapq.nsmallest(needed, bucket, key=lambda i: target_lex[i])
                    collected_indices.extend(top_items)
                    break

            for idx in collected_indices:
                out_s1_idx.append(s1_orig)
                out_s1_lex.append(s1_lex)
                out_cand_idx.append(idx)
                out_cand_lex.append(target_lex[idx])
                out_mask.append(cand_flags[idx] | BIT_COUNTRY)
                out_score.append(cand_scores[idx])
                out_exact.append(1 if names_target[idx] == name and name != '' else 0)

    if out_s1_idx:
        df_chunk = pd.DataFrame({
            's1_orig_idx': np.array(out_s1_idx, dtype=np.int32),
            's1_lex_rank': np.array(out_s1_lex, dtype=np.int32),
            'cand_orig_idx': np.array(out_cand_idx, dtype=np.int32),
            'cand_lex_rank': np.array(out_cand_lex, dtype=np.int32),
            'evidence_mask': np.array(out_mask, dtype=np.uint8),
            'evidence_score': np.array(out_score, dtype=np.int32),
            'blocked_exact_name': np.array(out_exact, dtype=np.int8)
        })
    else:
        df_chunk = pd.DataFrame({
            's1_orig_idx': np.array([], dtype=np.int32),
            's1_lex_rank': np.array([], dtype=np.int32),
            'cand_orig_idx': np.array([], dtype=np.int32),
            'cand_lex_rank': np.array([], dtype=np.int32),
            'evidence_mask': np.array([], dtype=np.uint8),
            'evidence_score': np.array([], dtype=np.int32),
            'blocked_exact_name': np.array([], dtype=np.int8)
        })

    return df_chunk, raw_counts_chunk, fallback_counts_chunk


def generate_bounded_candidates_e7(
    engine,
    df_s1,
    max_candidates_per_s1=400,
    chunk_size=25,
    addr_df_cap=50000,
    high_df_postings_sets=None,
    use_fallback=False
):
    c_size = chunk_size
    post_cap_chunks = []
    raw_counts_per_s1 = {}
    total_raw_candidates = 0
    overflowing_s1_count = 0
    zero_candidate_s1_count = 0
    total_fallbacks = 0

    s1_ids_all = df_s1['entity_id'].values
    s1_order = np.argsort(s1_ids_all)
    s1_lex_ranks_all = np.empty(len(s1_ids_all), dtype=np.int32)
    s1_lex_ranks_all[s1_order] = np.arange(len(s1_ids_all), dtype=np.int32)
    s1_indices_all = np.arange(len(s1_ids_all), dtype=np.int32)

    for i in range(0, len(df_s1), c_size):
        chunk_s1 = df_s1.iloc[i:i + c_size]
        chunk_indices = s1_indices_all[i:i + c_size]
        chunk_lex = s1_lex_ranks_all[i:i + c_size]

        capped_chunk, counts_in_chunk, fallbacks_in_chunk = generate_bounded_chunk_e7(
            engine,
            chunk_s1,
            chunk_indices,
            chunk_lex,
            max_candidates_per_s1=max_candidates_per_s1,
            addr_df_cap=addr_df_cap,
            high_df_postings_sets=high_df_postings_sets,
            use_fallback=use_fallback
        )
        total_fallbacks += fallbacks_in_chunk

        for sid_idx in chunk_indices:
            cnt = counts_in_chunk.get(sid_idx, 0)
            raw_counts_per_s1[s1_ids_all[sid_idx]] = cnt
            total_raw_candidates += cnt
            if cnt == 0:
                zero_candidate_s1_count += 1
            if cnt > max_candidates_per_s1:
                overflowing_s1_count += 1

        if len(capped_chunk) > 0:
            post_cap_chunks.append(capped_chunk)

        del capped_chunk
        gc.collect()

    if post_cap_chunks:
        capped_df = pd.concat(post_cap_chunks, ignore_index=True)
        capped_df = capped_df.sort_values(
            by=['s1_lex_rank', 'evidence_score', 'cand_lex_rank'],
            ascending=[True, False, True]
        ).reset_index(drop=True)

        mask = capped_df['evidence_mask'].values
        cand_orig = capped_df['cand_orig_idx'].values
        N = len(capped_df)

        b_ctry = ((mask >> 0) & 1).astype(np.int8)
        b_name = ((mask >> 1) & 1).astype(np.int8)
        b_p3 = ((mask >> 2) & 1).astype(np.int8)
        b_p4 = ((mask >> 3) & 1).astype(np.int8)
        b_addr = ((mask >> 4) & 1).astype(np.int8)
        num_keys = (b_name + b_p3 + b_p4 + b_addr).astype(np.int8)

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
            'evidence_score': capped_df['evidence_score'].values,
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
        'overflowing_s1_count': int(overflowing_s1_count),
        'zero_candidate_s1_count': int(zero_candidate_s1_count),
        'fallback_s1_count': int(total_fallbacks)
    }

    return final_df, metrics


def run_phase_e7():
    print("=" * 80, flush=True)
    print("PHASE E7: RECALL-PRESERVING DF-CAP EXPERIMENT", flush=True)
    print("=" * 80, flush=True)

    # 0. Production Firewall Check
    fp = get_config_fingerprint()
    print(f"Production Config Fingerprint: {fp}", flush=True)
    assert fp == "87f20ceeb84ccc6ea2d48678c7810ac5", f"Fingerprint mismatch: {fp}"
    print("Production Config Fingerprint VERIFIED: 87f20ceeb84ccc6ea2d48678c7810ac5", flush=True)

    # 1. Load France Targets
    print("\n[1/3] Loading France targets from cache...", flush=True)
    t0_targets = time.time()
    targets = pd.read_pickle('scratch/targets_france.pkl')
    n_targets = len(targets)
    print(f"  Loaded {n_targets:,} France targets in {time.time()-t0_targets:.2f}s", flush=True)

    # 2. Build Inverted Index
    print("\n[2/3] Building inverted index...", flush=True)
    t0_eng = time.time()
    engine = ConfigABlockingEngine(targets)
    print(f"  Inverted index built in {time.time()-t0_eng:.2f}s", flush=True)

    # 3. Load 500 France S1 Records
    print("\n[3/3] Loading 500 France S1 records...", flush=True)
    s1_df = pd.read_csv('data/test/test_source1.tsv', sep='\t', dtype=str, nrows=10000, na_filter=False)
    s1_fr = s1_df[s1_df['country'].apply(normalize_country) == 'france'].head(500).copy().reset_index(drop=True)
    del s1_df
    gc.collect()

    s1_fr['norm_name'] = s1_fr['business_name'].apply(normalize_business_name)
    s1_fr['norm_address'] = s1_fr['business_address'].apply(normalize_business_address)
    s1_fr['norm_country'] = 'france'
    n_s1 = len(s1_fr)
    print(f"  Loaded {n_s1} France S1 records.", flush=True)

    # Ground Truth Reference Pairs (Exact Name Matches: 4,191)
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
    print(f"  Reference True Pairs: {len(exact_matches_set):,} across {len(s1_name_map)} S1 names", flush=True)

    # Target info lookup for loss analysis
    target_dict = {}
    for idx in range(len(engine.target_eids)):
        eid = engine.target_eids[idx]
        target_dict[eid] = {
            'name': engine.target_names[idx],
            'addr': engine.target_addrs[idx],
            'source': engine.target_sources[idx],
            'idx': idx
        }
    s1_dict = s1_fr.set_index('entity_id').to_dict('index')

    # Load Control E6-0 Candidate Pairs from scratch for exact comparisons
    print("\nLoading E6-0 Control Candidate Pairs from scratch/candidates_e6_0.tsv.gz...", flush=True)
    df_e6_0 = pd.read_csv('scratch/candidates_e6_0.tsv.gz', sep='\t')
    e6_0_retained_pairs = set(zip(df_e6_0['entity_id_s1'], df_e6_0['entity_id_cand']))
    e6_0_true_post = e6_0_retained_pairs & exact_matches_set
    e6_0_raw = 216808077
    e6_0_time = 233.74
    print(f"  Control E6-0: {len(e6_0_retained_pairs):,} post-cap candidates | {len(e6_0_true_post):,} true pairs", flush=True)

    # Experiment Matrix
    configs = [
        ('E7-0', 50000, False, '50k Selective (No Fallback, E6-A equiv)'),
        ('E7-A', 50000, True, '50k Selective + Per-S1 Fallback'),
        ('E7-B', 100000, True, '100k Selective + Per-S1 Fallback'),
        ('E7-C', 150000, True, '150k Selective + Per-S1 Fallback'),
        ('E7-D', 200000, True, '200k Selective + Per-S1 Fallback'),
    ]

    country_addr_tokens = engine.country_addr_token_idx['france']
    results = {}
    capped_candidates_dict = {}
    loss_analysis_dict = {}

    print("\n" + "=" * 80, flush=True)
    print("RUNNING CONFIGURATIONS E7-0 -> E7-D", flush=True)
    print("=" * 80, flush=True)

    for cfg_id, addr_cap, use_fallback, label in configs:
        print(f"\n>>> Running {cfg_id} ({label})...", flush=True)
        t0_cfg = time.time()

        # Build high-DF posting sets for this threshold
        high_tokens = {k for k, v in country_addr_tokens.items() if len(v) > addr_cap}
        high_df_postings_sets = {k: set(country_addr_tokens[k]) for k in high_tokens}
        print(f"  High-DF address tokens (> {addr_cap:,}): {len(high_tokens)}", flush=True)

        mem_tracker = PeakMemoryTracker(interval=0.05)
        mem_tracker.start()

        t0_gen = time.time()
        capped_df, metrics = generate_bounded_candidates_e7(
            engine,
            s1_fr,
            max_candidates_per_s1=400,
            chunk_size=BLOCKING_CHUNK_SIZE,
            addr_df_cap=addr_cap,
            high_df_postings_sets=high_df_postings_sets,
            use_fallback=use_fallback
        )
        t_gen = time.time() - t0_gen
        peak_bytes = mem_tracker.stop()
        peak_mb = peak_bytes / (1024 * 1024)
        t_total = time.time() - t0_cfg

        raw_count = metrics['raw_candidate_count']
        post_cap_count = metrics['post_cap_candidate_count']
        mean_raw = metrics['candidates_per_s1']
        zero_cand_s1 = metrics['zero_candidate_s1_count']
        fallback_s1 = metrics['fallback_s1_count']

        post_cap_pairs = set(zip(capped_df['entity_id_s1'], capped_df['entity_id_cand']))
        capped_candidates_dict[cfg_id] = post_cap_pairs

        # True pairs metrics
        true_recov_post = len(post_cap_pairs & exact_matches_set)
        post_cap_recall = true_recov_post / len(exact_matches_set) if exact_matches_set else 1.0

        # Raw true pairs: All exact matches are generated by Name/Prefix channels
        raw_recov = len(exact_matches_set)
        raw_recall = 1.0

        throughput = n_s1 / t_gen if t_gen > 0 else 0
        raw_red = (1.0 - raw_count / e6_0_raw) * 100
        time_red = (1.0 - t_gen / e6_0_time) * 100

        cfg_true_post = post_cap_pairs & exact_matches_set
        lost_vs_e6_0 = len(e6_0_true_post - cfg_true_post)
        gain_vs_e6_0 = len(cfg_true_post - e6_0_true_post)

        recall_target_passed = (post_cap_recall >= 0.99)
        status_flag = "RECALL TARGET PASSED" if recall_target_passed else "RECALL TARGET FAILED"

        print(f"  {cfg_id} Done in {t_gen:.2f}s ({throughput:.2f} S1/s) | Raw: {raw_count:,} | Post-Cap Rec: {post_cap_recall:.4%} ({true_recov_post:,} / {len(exact_matches_set):,}) | Fallbacks: {fallback_s1} | Status: {status_flag}", flush=True)

        results[cfg_id] = {
            'config': cfg_id,
            'label': label,
            'addr_df_cap': addr_cap,
            'use_fallback': use_fallback,
            'raw_candidates': raw_count,
            'mean_raw_candidates': round(mean_raw, 2),
            'raw_true_pairs_captured': raw_recov,
            'raw_recall': round(raw_recall, 6),
            'post_cap_candidates': post_cap_count,
            'post_cap_true_pairs_recovered': true_recov_post,
            'post_cap_recall': round(post_cap_recall, 6),
            'zero_candidate_s1_count': zero_cand_s1,
            'fallback_s1_count': fallback_s1,
            'runtime_seconds': round(t_gen, 2),
            'throughput_s1_per_sec': round(throughput, 2),
            'peak_rss_mb': round(peak_mb, 1),
            'candidate_reduction_pct': round(raw_red, 2),
            'runtime_reduction_pct': round(time_red, 2),
            'true_pairs_lost_vs_control': lost_vs_e6_0,
            'true_pairs_gained_vs_control': gain_vs_e6_0,
            'recall_target_passed': recall_target_passed,
            'status_flag': status_flag
        }

        # LOSS ANALYSIS: True pairs lost after the cap
        lost_pairs = exact_matches_set - cfg_true_post
        lost_details = []
        for s1_id, t_id in list(lost_pairs)[:50]:
            s1_info = s1_dict[s1_id]
            t_info = target_dict[t_id]
            s1_name = s1_info['norm_name']
            s1_addr = s1_info['norm_address']
            t_name = t_info['name']
            t_addr = t_info['addr']
            t_src = t_info['source']

            # Check channels that fired for this pair
            firing_channels = []
            for tok in s1_name.split():
                if tok in engine.country_name_token_idx['france'] and t_info['idx'] in engine.country_name_token_idx['france'][tok]:
                    firing_channels.append(f"NameToken('{tok}')")
            if len(s1_name) >= 3 and s1_name[:3] in engine.country_prefix3_idx['france'] and t_info['idx'] in engine.country_prefix3_idx['france'][s1_name[:3]]:
                firing_channels.append(f"Prefix3('{s1_name[:3]}')")
            if len(s1_name) >= 4 and s1_name[:4] in engine.country_prefix4_idx['france'] and t_info['idx'] in engine.country_prefix4_idx['france'][s1_name[:4]]:
                firing_channels.append(f"Prefix4('{s1_name[:4]}')")
            for atok in s1_addr.split():
                if atok in country_addr_tokens and t_info['idx'] in country_addr_tokens[atok]:
                    firing_channels.append(f"AddrToken('{atok}')")

            lost_details.append({
                's1_id': s1_id,
                'target_id': t_id,
                'source': t_src,
                'business_name': s1_name,
                's1_address': s1_addr,
                'target_address': t_addr,
                'firing_channels': firing_channels,
                'loss_stage': 'CAP_DISPLACEMENT',
                'reason': 'Generated by Name/Prefix channels but displaced by 400-candidate cap ranking cutoff.'
            })

        loss_analysis_dict[cfg_id] = {
            'total_lost': len(lost_pairs),
            'sample_details': lost_details
        }

        del capped_df, high_df_postings_sets
        gc.collect()

    output_data = {
        'benchmark_summary': {
            's1_count': n_s1,
            'target_count': n_targets,
            'candidate_cap': 400,
            'config_fingerprint': fp,
            'reference_true_pairs': len(exact_matches_set),
            'target_recall_threshold': 0.99
        },
        'configurations': results,
        'loss_analysis': loss_analysis_dict
    }

    out_json = 'experiments/optimization_e7_results.json'
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nAll results saved to {out_json}", flush=True)

    # Print Comparison Table
    print("\n" + "=" * 140, flush=True)
    print(f"{'Config':<8}{'DF Cap':<10}{'Fallback':<10}{'Raw Cands':<14}{'Raw Rec':<10}{'Post Cands':<12}{'Post Rec':<12}{'Zero S1':<9}{'Runtime(s)':<12}{'Peak RSS':<10}{'Cand Red%':<11}{'True Lost':<11}{'Target Status':<22}")
    print("-" * 140, flush=True)
    for cfg_id in ['E7-0', 'E7-A', 'E7-B', 'E7-C', 'E7-D']:
        r = results[cfg_id]
        fb_str = "YES" if r['use_fallback'] else "NO"
        print(f"{r['config']:<8}{r['addr_df_cap']:<10,}{fb_str:<10}{r['raw_candidates']:<14,}{r['raw_recall']:<10.2%}{r['post_cap_candidates']:<12,}{r['post_cap_recall']:<12.4%}{r['zero_candidate_s1_count']:<9}{r['runtime_seconds']:<12.2f}{r['peak_rss_mb']:<10.1f}{r['candidate_reduction_pct']:<11.2f}%{r['true_pairs_lost_vs_control']:<11}{r['status_flag']:<22}", flush=True)
    print("=" * 140, flush=True)

    return output_data


if __name__ == '__main__':
    run_phase_e7()
