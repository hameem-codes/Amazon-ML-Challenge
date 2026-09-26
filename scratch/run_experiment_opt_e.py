"""
OPTIMIZATION E — ISOLATED BLOCKING FREQUENCY-CAP EXPERIMENT
Side-by-side, isolated evaluation of Document-Frequency (DF) filtering on Config A blocking.
Tests:
- E0: Baseline Config A (No cap)
- E1: DF Cap 5,000
- E2: DF Cap 10,000
- E3: DF Cap 20,000
- E4: DF Cap 50,000

Measures:
- Candidate Volume (raw, mean, median, max, post-cap, mean post-cap)
- Performance (candidate generation runtime, peak RSS, speedup/reduction vs E0)
- Recall on True Pairs (raw recall, post-cap recall, zero-candidate S1 count)
- Channel Diagnostics (Country, Name Token, Prefix-3, Prefix-4, Address Token)
- Critical Recall Analysis (inspection of every lost pair relative to E0)
"""

import sys, os, time, gc, json, copy, threading
from collections import defaultdict
sys.path.insert(0, 'src')
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


def build_df_capped_engine(base_engine, df_cap=None):
    """
    Creates an isolated clone of base_engine with DF filtering applied to:
    - Name Token (DF > df_cap -> ignored)
    - Prefix-3 (DF > df_cap -> ignored)
    - Prefix-4 (DF > df_cap -> ignored)
    - Address Token (DF > df_cap -> ignored)
    Country partition is unchanged.
    """
    if df_cap is None:
        return base_engine

    cloned = copy.copy(base_engine)
    
    cloned.country_name_token_idx = defaultdict(lambda: defaultdict(list))
    for c, idx in base_engine.country_name_token_idx.items():
        cloned.country_name_token_idx[c] = {k: v for k, v in idx.items() if len(v) <= df_cap}
        
    cloned.country_prefix3_idx = defaultdict(lambda: defaultdict(list))
    for c, idx in base_engine.country_prefix3_idx.items():
        cloned.country_prefix3_idx[c] = {k: v for k, v in idx.items() if len(v) <= df_cap}
        
    cloned.country_prefix4_idx = defaultdict(lambda: defaultdict(list))
    for c, idx in base_engine.country_prefix4_idx.items():
        cloned.country_prefix4_idx[c] = {k: v for k, v in idx.items() if len(v) <= df_cap}
        
    cloned.country_addr_token_idx = defaultdict(lambda: defaultdict(list))
    for c, idx in base_engine.country_addr_token_idx.items():
        cloned.country_addr_token_idx[c] = {k: v for k, v in idx.items() if len(v) <= df_cap}
        
    return cloned


def analyze_channels(base_engine, s1_df, true_target_indices_by_s1, df_cap=None):
    """
    Measures per-channel candidate contribution and recall:
    - Country
    - Name Token
    - Prefix-3
    - Prefix-4
    - Address Token
    Uses fast integer set operations for maximum performance.
    """
    c = 'france'
    s1_names = s1_df['norm_name'].values
    s1_addrs = s1_df['norm_address'].values
    s1_eids = s1_df['entity_id'].values
    
    name_idx = base_engine.country_name_token_idx[c]
    p3_idx = base_engine.country_prefix3_idx[c]
    p4_idx = base_engine.country_prefix4_idx[c]
    addr_idx = base_engine.country_addr_token_idx[c]
    
    ch_keys = {'Country': 1, 'Name Token': 0, 'Prefix-3': 0, 'Prefix-4': 0, 'Address Token': 0}
    ch_raw_cands = {'Country': 0, 'Name Token': 0, 'Prefix-3': 0, 'Prefix-4': 0, 'Address Token': 0}
    ch_unique_cands = {'Country': 0, 'Name Token': 0, 'Prefix-3': 0, 'Prefix-4': 0, 'Address Token': 0}
    ch_true_recov = {'Country': 0, 'Name Token': 0, 'Prefix-3': 0, 'Prefix-4': 0, 'Address Token': 0}
    
    # Count available keys within cap
    for k, v in name_idx.items():
        if df_cap is None or len(v) <= df_cap:
            ch_keys['Name Token'] += 1
    for k, v in p3_idx.items():
        if df_cap is None or len(v) <= df_cap:
            ch_keys['Prefix-3'] += 1
    for k, v in p4_idx.items():
        if df_cap is None or len(v) <= df_cap:
            ch_keys['Prefix-4'] += 1
    for k, v in addr_idx.items():
        if df_cap is None or len(v) <= df_cap:
            ch_keys['Address Token'] += 1
            
    for s1_id, name, addr in zip(s1_eids, s1_names, s1_addrs):
        s1_toks = [t for t in name.split() if len(t) > 1 and t not in base_engine.stop_tokens]
        s1_p3 = name[:3] if len(name) >= 3 else ''
        s1_p4 = name[:4] if len(name) >= 4 else ''
        s1_addr_toks = set([t for t in addr.split() if len(t) >= 4 and t not in base_engine.stop_addrs]) if addr else set()
        
        # Name tokens
        s1_name_cands = []
        for t in s1_toks:
            postings = name_idx.get(t, [])
            if df_cap is None or len(postings) <= df_cap:
                s1_name_cands.extend(postings)
        ch_raw_cands['Name Token'] += len(s1_name_cands)
        set_name = set(s1_name_cands)
        
        # Prefix 3
        s1_p3_cands = []
        if s1_p3:
            postings = p3_idx.get(s1_p3, [])
            if df_cap is None or len(postings) <= df_cap:
                s1_p3_cands.extend(postings)
        ch_raw_cands['Prefix-3'] += len(s1_p3_cands)
        set_p3 = set(s1_p3_cands)
        
        # Prefix 4
        s1_p4_cands = []
        if s1_p4:
            postings = p4_idx.get(s1_p4, [])
            if df_cap is None or len(postings) <= df_cap:
                s1_p4_cands.extend(postings)
        ch_raw_cands['Prefix-4'] += len(s1_p4_cands)
        set_p4 = set(s1_p4_cands)
        
        # Addr tokens
        s1_addr_cands = []
        for t in s1_addr_toks:
            postings = addr_idx.get(t, [])
            if df_cap is None or len(postings) <= df_cap:
                s1_addr_cands.extend(postings)
        ch_raw_cands['Address Token'] += len(s1_addr_cands)
        set_addr = set(s1_addr_cands)
        
        # Unique candidates contributed
        ch_unique_cands['Name Token'] += len(set_name)
        ch_unique_cands['Prefix-3'] += len(set_p3)
        ch_unique_cands['Prefix-4'] += len(set_p4)
        ch_unique_cands['Address Token'] += len(set_addr)
        
        union_all = set_name | set_p3 | set_p4 | set_addr
        ch_raw_cands['Country'] += len(union_all)
        ch_unique_cands['Country'] += len(union_all)
        
        # Fast true pairs recovery via set intersection on target indices
        true_indices = true_target_indices_by_s1.get(s1_id, set())
        if true_indices:
            ch_true_recov['Name Token'] += len(set_name & true_indices)
            ch_true_recov['Prefix-3'] += len(set_p3 & true_indices)
            ch_true_recov['Prefix-4'] += len(set_p4 & true_indices)
            ch_true_recov['Address Token'] += len(set_addr & true_indices)
            ch_true_recov['Country'] += len(union_all & true_indices)
                
    diagnostics = {}
    for ch in ['Country', 'Name Token', 'Prefix-3', 'Prefix-4', 'Address Token']:
        diagnostics[ch] = {
            'keys': ch_keys[ch],
            'raw_candidates': ch_raw_cands[ch],
            'unique_candidates': ch_unique_cands[ch],
            'true_pairs_recovered': ch_true_recov[ch]
        }
    return diagnostics


def run_optimization_e():
    print("=" * 70)
    print("OPTIMIZATION E: ISOLATED BLOCKING FREQUENCY-CAP EXPERIMENT")
    print("=" * 70)

    # Verify locked production fingerprint
    fp = get_config_fingerprint()
    print(f"Production Config Fingerprint: {fp}")
    assert fp == "87f20ceeb84ccc6ea2d48678c7810ac5", f"Fingerprint mismatch: {fp}"
    print(f"LOCKED ARCHITECTURE VERIFIED: 87f20ceeb84ccc6ea2d48678c7810ac5")

    # 1. Load France targets (1,434,993)
    print("\n[1/4] Loading 1,434,993 France targets (Source 2 + Source 3)...")
    t0_targets = time.time()
    s2_chunks = [
        chunk[chunk['country'].apply(normalize_country) == 'france']
        for chunk in pd.read_csv('data/test/test_source2.tsv', sep='\t', dtype=str, na_filter=False, chunksize=500000)
    ]
    s2_fr = pd.concat(s2_chunks, ignore_index=True)
    s2_fr['source'] = 'S2'
    del s2_chunks
    gc.collect()

    s3_chunks = [
        chunk[chunk['country'].apply(normalize_country) == 'france']
        for chunk in pd.read_csv('data/test/test_source3.tsv', sep='\t', dtype=str, na_filter=False, chunksize=500000)
    ]
    s3_fr = pd.concat(s3_chunks, ignore_index=True)
    s3_fr['source'] = 'S3'
    del s3_chunks
    gc.collect()

    targets = pd.concat([s2_fr, s3_fr], ignore_index=True)
    del s2_fr, s3_fr
    gc.collect()

    n_targets = len(targets)
    targets['norm_name'] = targets['business_name'].apply(normalize_business_name)
    targets['norm_address'] = targets['business_address'].apply(normalize_business_address)
    targets['norm_country'] = 'france'
    print(f"  Loaded and normalized {n_targets:,} targets in {time.time()-t0_targets:.2f}s")

    # 2. Build Base Inverted Index
    print("\n[2/4] Building base inverted index (Config A)...")
    t0_eng = time.time()
    base_engine = ConfigABlockingEngine(targets)
    t_eng = time.time() - t0_eng
    proc = psutil.Process(os.getpid())
    print(f"  Index built in {t_eng:.2f}s | RSS: {proc.memory_info().rss / (1024*1024):.1f} MB")

    # 3. Load 500 France S1 Records
    print("\n[3/4] Loading 500 France S1 records...")
    s1_df = pd.read_csv('data/test/test_source1.tsv', sep='\t', dtype=str, nrows=10000, na_filter=False)
    s1_fr = s1_df[s1_df['country'].apply(normalize_country) == 'france'].head(500).copy().reset_index(drop=True)
    del s1_df
    gc.collect()

    s1_fr['norm_name'] = s1_fr['business_name'].apply(normalize_business_name)
    s1_fr['norm_address'] = s1_fr['business_address'].apply(normalize_business_address)
    s1_fr['norm_country'] = 'france'
    n_s1 = len(s1_fr)
    print(f"  Loaded {n_s1} France S1 records.")

    # 4. Precompute Reference Ground-Truth / True Pairs
    print("\n[4/4] Establishing Ground-Truth Reference True Pairs...")
    # Reference 1: Exact Name Matches in France (4,191 true pairs)
    s1_name_map = {}
    for idx, r in s1_fr.iterrows():
        s1_name_map.setdefault(r['norm_name'], []).append(r['entity_id'])

    exact_matches_set = set()
    true_target_indices_by_s1 = defaultdict(set)
    for idx in range(len(base_engine.target_names)):
        t_name = base_engine.target_names[idx]
        if t_name in s1_name_map:
            t_eid = base_engine.target_eids[idx]
            for s1_id in s1_name_map[t_name]:
                exact_matches_set.add((s1_id, t_eid))
                true_target_indices_by_s1[s1_id].add(idx)
    print(f"  Reference True Pairs (Exact Name Matches): {len(exact_matches_set):,} across 406 S1 entities")

    # Configurations to test
    configs = [
        ('E0', None, 'None (Production Baseline)'),
        ('E1', 5000, '5,000'),
        ('E2', 10000, '10,000'),
        ('E3', 20000, '20,000'),
        ('E4', 50000, '50,000'),
    ]

    results = {}
    capped_candidates_dict = {}
    raw_counts_dict = {}

    print("\n" + "=" * 70)
    print("RUNNING EXPERIMENTAL CONFIGURATIONS E0 -> E4")
    print("=" * 70)

    for cfg_id, df_cap, label in configs:
        print(f"\n>>> Running {cfg_id} (DF Cap: {label})...")
        t0_cfg = time.time()
        
        # Build isolated engine
        eng = build_df_capped_engine(base_engine, df_cap=df_cap)
        
        mem_tracker = PeakMemoryTracker(interval=0.05)
        mem_tracker.start()
        
        t0_gen = time.time()
        capped_df, metrics = eng.generate_bounded_candidates(
            s1_fr,
            max_candidates_per_s1=400,
            chunk_size=BLOCKING_CHUNK_SIZE,
            return_metrics=True
        )
        t_gen = time.time() - t0_gen
        peak_bytes = mem_tracker.stop()
        peak_mb = peak_bytes / (1024 * 1024)
        t_total = time.time() - t0_cfg
        
        # Extract candidate metrics
        raw_count = metrics['raw_candidate_count']
        post_cap_count = metrics['post_cap_candidate_count']
        mean_raw = metrics['candidates_per_s1']
        median_raw = metrics['median_candidates_per_s1']
        max_raw = metrics['max_candidates_per_s1']
        zero_cand_s1 = metrics['zero_candidate_s1_count']
        mean_post_cap = post_cap_count / n_s1
        
        # Evaluate Recall on Reference True Pairs (Exact Name Matches)
        post_cap_pairs = set(zip(capped_df['entity_id_s1'], capped_df['entity_id_cand']))
        capped_candidates_dict[cfg_id] = post_cap_pairs
        
        true_recov_post_cap = len(post_cap_pairs & exact_matches_set)
        post_cap_recall = true_recov_post_cap / len(exact_matches_set) if exact_matches_set else 1.0
        
        # Channel diagnostics
        print(f"  Computing channel diagnostics for {cfg_id}...")
        diag = analyze_channels(base_engine, s1_fr, true_target_indices_by_s1, df_cap=df_cap)
        raw_true_recov = diag['Country']['true_pairs_recovered']
        raw_recall = raw_true_recov / len(exact_matches_set) if exact_matches_set else 1.0
        
        print(f"  {cfg_id} Complete in {t_gen:.2f}s | Raw Cands: {raw_count:,} | Post-Cap: {post_cap_count:,} | Post-Cap Recall: {post_cap_recall:.4%}")
        
        results[cfg_id] = {
            'config': cfg_id,
            'df_cap': df_cap,
            'label': label,
            'runtime_seconds': round(t_gen, 2),
            'total_runtime_seconds': round(t_total, 2),
            'peak_rss_mb': round(peak_mb, 1),
            'raw_candidates': raw_count,
            'mean_raw_candidates': round(mean_raw, 2),
            'median_raw_candidates': round(median_raw, 1),
            'max_raw_candidates': max_raw,
            'post_cap_candidates': post_cap_count,
            'mean_post_cap_candidates': round(mean_post_cap, 2),
            'ground_truth_pairs': len(exact_matches_set),
            'raw_true_pairs_recovered': raw_true_recov,
            'raw_recall': round(raw_recall, 6),
            'post_cap_true_pairs_recovered': true_recov_post_cap,
            'post_cap_recall': round(post_cap_recall, 6),
            'zero_candidate_s1_count': zero_cand_s1,
            'channel_diagnostics': diag
        }
        
        del eng, capped_df
        gc.collect()

    # Compute candidate & runtime reductions vs E0
    e0_raw = results['E0']['raw_candidates']
    e0_time = results['E0']['runtime_seconds']
    e0_retained_pairs = capped_candidates_dict['E0']

    for cfg_id in ['E0', 'E1', 'E2', 'E3', 'E4']:
        raw_red = (1.0 - results[cfg_id]['raw_candidates'] / e0_raw) * 100
        time_red = (1.0 - results[cfg_id]['runtime_seconds'] / e0_time) * 100 if e0_time > 0 else 0
        retained_v_e0 = len(capped_candidates_dict[cfg_id] & e0_retained_pairs)
        results[cfg_id]['raw_candidate_reduction_pct'] = round(raw_red, 2)
        results[cfg_id]['runtime_reduction_pct'] = round(time_red, 2)
        results[cfg_id]['e0_candidates_retained'] = retained_v_e0
        results[cfg_id]['e0_retention_pct'] = round(retained_v_e0 / len(e0_retained_pairs) * 100, 2)

    # Critical Recall Analysis: Inspect any true pair lost relative to E0
    print("\n" + "=" * 70)
    print("CRITICAL RECALL ANALYSIS (LOST PAIRS RELATIVE TO E0)")
    print("=" * 70)

    e0_true_post = capped_candidates_dict['E0'] & exact_matches_set
    lost_analysis = {}

    target_dict = {}
    for idx in range(len(base_engine.target_eids)):
        eid = base_engine.target_eids[idx]
        target_dict[eid] = {
            'name': base_engine.target_names[idx],
            'addr': base_engine.target_addrs[idx]
        }
    s1_dict = s1_fr.set_index('entity_id').to_dict('index')

    for cfg_id in ['E1', 'E2', 'E3', 'E4']:
        cfg_true_post = capped_candidates_dict[cfg_id] & exact_matches_set
        lost_pairs = e0_true_post - cfg_true_post
        lost_details = []
        print(f"\n{cfg_id} (Cap {results[cfg_id]['label']}) Lost True Pairs vs E0: {len(lost_pairs)}")
        
        for s1_id, t_id in list(lost_pairs)[:20]:
            s1_info = s1_dict[s1_id]
            t_info = target_dict[t_id]
            s1_name = s1_info['norm_name']
            s1_addr = s1_info['norm_address']
            
            # Identify which key recovered it under baseline and its DF
            reasons = []
            # Name tokens
            for tok in s1_name.split():
                if tok in base_engine.country_name_token_idx['france']:
                    df_val = len(base_engine.country_name_token_idx['france'][tok])
                    if df_val > results[cfg_id]['df_cap']:
                        reasons.append(f"NameToken('{tok}', DF={df_val:,} > cap)")
            # Prefix 3
            p3 = s1_name[:3] if len(s1_name) >= 3 else ''
            if p3 in base_engine.country_prefix3_idx['france']:
                df_val = len(base_engine.country_prefix3_idx['france'][p3])
                if df_val > results[cfg_id]['df_cap']:
                    reasons.append(f"Prefix3('{p3}', DF={df_val:,} > cap)")
            # Prefix 4
            p4 = s1_name[:4] if len(s1_name) >= 4 else ''
            if p4 in base_engine.country_prefix4_idx['france']:
                df_val = len(base_engine.country_prefix4_idx['france'][p4])
                if df_val > results[cfg_id]['df_cap']:
                    reasons.append(f"Prefix4('{p4}', DF={df_val:,} > cap)")
            # Addr tokens
            for atok in s1_addr.split():
                if atok in base_engine.country_addr_token_idx['france']:
                    df_val = len(base_engine.country_addr_token_idx['france'][atok])
                    if df_val > results[cfg_id]['df_cap']:
                        reasons.append(f"AddrToken('{atok}', DF={df_val:,} > cap)")
                        
            lost_details.append({
                's1_id': s1_id,
                'target_id': t_id,
                's1_name': s1_name,
                'target_name': t_info['name'],
                'experimental_reason': "; ".join(reasons) if reasons else "Dropped due to candidate cap / ranking change"
            })
            print(f"  Lost Pair: {s1_id} <-> {t_id} | Name: '{s1_name}' | Reason: {'; '.join(reasons)}")
            
        lost_analysis[cfg_id] = {
            'lost_count': len(lost_pairs),
            'sample_details': lost_details
        }

    # Save comprehensive results
    output_data = {
        'benchmark_summary': {
            's1_count': n_s1,
            'target_count': n_targets,
            'candidate_cap': 400,
            'config_fingerprint': fp,
            'reference_true_pairs': len(exact_matches_set)
        },
        'configurations': results,
        'lost_true_pairs_analysis': lost_analysis
    }

    # Run Labeled Validation Experiment on 1,778 true pairs
    print("\n" + "=" * 70)
    print("RUNNING LABELED TOURNAMENT VALIDATION EXPERIMENT (1,778 TRUE PAIRS)")
    print("=" * 70)
    labeled_res = run_labeled_validation_experiment(configs)
    output_data['labeled_validation'] = labeled_res

    # Save complete JSON
    out_path = 'experiments/optimization_e_results.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nAll experimental results saved to {out_path}")

    # Print summary table
    print("\n" + "=" * 105)
    print(f"{'Config':<8}{'DF Cap':<12}{'Raw Cands':<14}{'Raw Rec':<10}{'Post-Cap Cands':<16}{'Post-Cap Rec':<14}{'Runtime(s)':<12}{'Peak RSS(MB)':<12}{'Cand Reduc %':<14}")
    print("-" * 105)
    for cfg_id in ['E0', 'E1', 'E2', 'E3', 'E4']:
        r = results[cfg_id]
        print(f"{r['config']:<8}{r['label']:<12}{r['raw_candidates']:<14,}{r['raw_recall']:<10.4%}{r['post_cap_candidates']:<16,}{r['post_cap_recall']:<14.4%}{r['runtime_seconds']:<12.2f}{r['peak_rss_mb']:<12.1f}{r['raw_candidate_reduction_pct']:<14.2f}%")
    print("=" * 105)

    print("\n" + "=" * 105)
    print(f"LABELED VALIDATION BENCHMARK (1,778 TRUE PAIRS):")
    print(f"{'Config':<8}{'DF Cap':<12}{'Raw Cands':<14}{'Raw Rec':<12}{'Post-Cap Cands':<16}{'Post-Cap Rec':<14}{'Runtime(s)':<12}")
    print("-" * 105)
    for cfg_id in ['E0', 'E1', 'E2', 'E3', 'E4']:
        lr = labeled_res[cfg_id]
        print(f"{lr['config']:<8}{lr['label']:<12}{lr['raw_candidates']:<14,}{lr['raw_recall']:<12.4%}{lr['post_cap_candidates']:<16,}{lr['post_cap_recall']:<14.4%}{lr['runtime_seconds']:<12.3f}")
    print("=" * 105)

    return output_data


def run_labeled_validation_experiment(configs):
    s1_df = pd.read_csv('scratch/tournament_s1.tsv', sep='\t', dtype=str, na_filter=False)
    targets_df = pd.read_csv('scratch/tournament_targets.tsv', sep='\t', dtype=str, na_filter=False)
    true_pairs_df = pd.read_csv('scratch/tournament_true_pairs.tsv', sep='\t', dtype=str, na_filter=False)

    true_pairs_set = set(zip(true_pairs_df['source1_entity_id'], true_pairs_df['matched_entity_id']))
    s1_sample = s1_df.head(500).copy().reset_index(drop=True)
    sample_true_pairs = {
        (s1, tgt) for (s1, tgt) in true_pairs_set if s1 in set(s1_sample['entity_id'])
    }
    
    base_tourn_engine = ConfigABlockingEngine(targets_df)
    labeled_results = {}

    for cfg_id, df_cap, label in configs:
        eng = build_df_capped_engine(base_tourn_engine, df_cap=df_cap)
        t0 = time.time()
        capped_df, metrics = eng.generate_bounded_candidates(
            s1_sample,
            max_candidates_per_s1=400,
            chunk_size=BLOCKING_CHUNK_SIZE,
            return_metrics=True
        )
        t_gen = time.time() - t0
        
        raw_count = metrics['raw_candidate_count']
        post_cap_count = metrics['post_cap_candidate_count']
        post_pairs = set(zip(capped_df['entity_id_s1'], capped_df['entity_id_cand']))
        recov_true = len(post_pairs & sample_true_pairs)
        rec = recov_true / len(sample_true_pairs) if sample_true_pairs else 1.0
        
        # Build true target indices for channel diagnostics
        true_indices_map = defaultdict(set)
        for idx in range(len(base_tourn_engine.target_eids)):
            eid = base_tourn_engine.target_eids[idx]
            for s1_id, t_eid in sample_true_pairs:
                if t_eid == eid:
                    true_indices_map[s1_id].add(idx)
                    
        diag = analyze_channels(base_tourn_engine, s1_sample, true_indices_map, df_cap=df_cap)
        raw_recov = diag['Country']['true_pairs_recovered']
        raw_rec = raw_recov / len(sample_true_pairs) if sample_true_pairs else 1.0
        
        labeled_results[cfg_id] = {
            'config': cfg_id,
            'label': label,
            'runtime_seconds': round(t_gen, 3),
            'raw_candidates': raw_count,
            'raw_true_pairs_recovered': raw_recov,
            'raw_recall': round(raw_rec, 6),
            'post_cap_candidates': post_cap_count,
            'post_cap_true_pairs_recovered': recov_true,
            'post_cap_recall': round(rec, 6),
            'zero_candidate_s1_count': metrics['zero_candidate_s1_count'],
            'total_sample_true_pairs': len(sample_true_pairs)
        }
        del eng, capped_df
        gc.collect()

    return labeled_results


if __name__ == '__main__':
    run_optimization_e()
