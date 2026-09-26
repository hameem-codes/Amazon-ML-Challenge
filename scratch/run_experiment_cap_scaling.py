"""
PHASE E7 — CANDIDATE CAP SCALING EXPERIMENT
Evaluates the impact of candidate cap scaling (400, 500, 600, 800, 1000)
on post-cap recall, runtime, memory, and marginal efficiency.
Uses the selective DF > 50,000 configuration (E6-A / E7-0).
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


def run_cap_scaling():
    print("=" * 85, flush=True)
    print("E7: CANDIDATE CAP SCALING EXPERIMENT (400 -> 1000 per S1)", flush=True)
    print("=" * 85, flush=True)

    # 0. Production Firewall Check
    fp = get_config_fingerprint()
    print(f"Production Config Fingerprint: {fp}", flush=True)
    assert fp == "87f20ceeb84ccc6ea2d48678c7810ac5", f"Fingerprint mismatch: {fp}"
    print("Production Config Fingerprint VERIFIED: 87f20ceeb84ccc6ea2d48678c7810ac5", flush=True)

    # 1. Load France Targets from cache
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

    # Precompute high-DF postings sets for DF > 50,000
    addr_cap = 50000
    country_addr_tokens = engine.country_addr_token_idx['france']
    high_tokens = {k for k, v in country_addr_tokens.items() if len(v) > addr_cap}
    high_df_postings_sets = {k: set(country_addr_tokens[k]) for k in high_tokens}
    print(f"  High-DF address tokens (> {addr_cap:,}): {len(high_tokens)}", flush=True)

    caps = [
        ('E7-400', 400),
        ('E7-500', 500),
        ('E7-600', 600),
        ('E7-800', 800),
        ('E7-1000', 1000)
    ]

    results = {}
    print("\n" + "=" * 85, flush=True)
    print("RUNNING CANDIDATE CAP EXPERIMENT (400, 500, 600, 800, 1000)", flush=True)
    print("=" * 85, flush=True)

    for cfg_id, cap_val in caps:
        print(f"\n>>> Running {cfg_id} (Candidate Cap = {cap_val} / S1)...", flush=True)
        t0_cfg = time.time()

        mem_tracker = PeakMemoryTracker(interval=0.05)
        mem_tracker.start()

        t0_gen = time.time()
        capped_df, metrics = generate_bounded_candidates_e7(
            engine,
            s1_fr,
            max_candidates_per_s1=cap_val,
            chunk_size=BLOCKING_CHUNK_SIZE,
            addr_df_cap=addr_cap,
            high_df_postings_sets=high_df_postings_sets,
            use_fallback=False
        )
        t_gen = time.time() - t0_gen
        peak_bytes = mem_tracker.stop()
        peak_mb = peak_bytes / (1024 * 1024)

        raw_count = metrics['raw_candidate_count']
        post_cap_count = metrics['post_cap_candidate_count']
        mean_cands_s1 = post_cap_count / n_s1
        zero_cand_s1 = metrics['zero_candidate_s1_count']

        post_cap_pairs = set(zip(capped_df['entity_id_s1'], capped_df['entity_id_cand']))
        true_recov_post = len(post_cap_pairs & exact_matches_set)
        post_cap_recall = true_recov_post / len(exact_matches_set) if exact_matches_set else 1.0
        lost_count = len(exact_matches_set) - true_recov_post
        throughput = n_s1 / t_gen if t_gen > 0 else 0

        print(f"  {cfg_id} Done in {t_gen:.2f}s ({throughput:.2f} S1/s) | Post-Cap: {post_cap_count:,} ({mean_cands_s1:.1f}/S1) | Recall: {post_cap_recall:.4%} ({true_recov_post:,} / {len(exact_matches_set):,}) | Lost: {lost_count}", flush=True)

        results[cfg_id] = {
            'config': cfg_id,
            'candidate_cap': cap_val,
            'raw_candidates': raw_count,
            'raw_recall': 1.0,
            'post_cap_candidates': post_cap_count,
            'candidates_per_s1': round(mean_cands_s1, 1),
            'post_cap_recall': round(post_cap_recall, 6),
            'true_pairs_retained': true_recov_post,
            'true_pairs_lost': lost_count,
            'runtime_seconds': round(t_gen, 2),
            'throughput_s1_per_sec': round(throughput, 2),
            'peak_rss_mb': round(peak_mb, 1),
            'zero_candidate_s1_count': zero_cand_s1
        }

        del capped_df
        gc.collect()

    # Calculate Incremental Metrics vs E7-400
    base_time = results['E7-400']['runtime_seconds']
    base_cands = results['E7-400']['post_cap_candidates']
    base_true = results['E7-400']['true_pairs_retained']

    for cfg_id in ['E7-400', 'E7-500', 'E7-600', 'E7-800', 'E7-1000']:
        r = results[cfg_id]
        inc_time = r['runtime_seconds'] - base_time
        inc_cands = r['post_cap_candidates'] - base_cands
        inc_true = r['true_pairs_retained'] - base_true

        pairs_per_cand = (inc_true / inc_cands) if inc_cands > 0 else 0.0
        pairs_per_sec = (inc_true / inc_time) if inc_time > 0 else 0.0

        r['incremental_runtime_vs_400'] = round(inc_time, 2)
        r['incremental_candidates_vs_400'] = inc_cands
        r['incremental_true_pairs_vs_400'] = inc_true
        r['pairs_recovered_per_candidate'] = round(pairs_per_cand, 6)
        r['pairs_recovered_per_second'] = round(pairs_per_sec, 2)

    # Save JSON results
    out_json = 'experiments/optimization_e7_cap_results.json'
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump({
            'benchmark_summary': {
                's1_count': n_s1,
                'target_count': n_targets,
                'config_fingerprint': fp,
                'reference_true_pairs': len(exact_matches_set)
            },
            'configurations': results
        }, f, indent=2)
    print(f"\nAll cap scaling results saved to {out_json}", flush=True)

    # Summary Table
    print("\n" + "=" * 145, flush=True)
    print(f"{'Config':<10}{'Cap/S1':<8}{'Post Cands':<12}{'Cands/S1':<10}{'True Recov':<12}{'Post Rec':<12}{'True Lost':<11}{'Runtime(s)':<12}{'Peak RSS':<10}{'Inc Time(s)':<13}{'Inc True':<10}{'True/Addl Cand':<16}{'True/Addl Sec':<14}")
    print("-" * 145, flush=True)
    for cfg_id in ['E7-400', 'E7-500', 'E7-600', 'E7-800', 'E7-1000']:
        r = results[cfg_id]
        print(f"{r['config']:<10}{r['candidate_cap']:<8}{r['post_cap_candidates']:<12,}{r['candidates_per_s1']:<10.1f}{r['true_pairs_retained']:<12,}{r['post_cap_recall']:<12.4%}{r['true_pairs_lost']:<11,}{r['runtime_seconds']:<12.2f}{r['peak_rss_mb']:<10.1f}{r['incremental_runtime_vs_400']:<13.2f}{r['incremental_true_pairs_vs_400']:<10,}{r['pairs_recovered_per_candidate']:<16.6f}{r['pairs_recovered_per_second']:<14.2f}", flush=True)
    print("=" * 145, flush=True)

    return results


if __name__ == '__main__':
    run_cap_scaling()
