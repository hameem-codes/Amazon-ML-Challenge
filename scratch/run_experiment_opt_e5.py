"""
OPTIMIZATION E5 — ADDRESS-TOKEN-ONLY DF FILTERING EXPERIMENT
Evaluates document-frequency filtering applied STRICTLY to Address Token blocking.
Name Token, Prefix-3, Prefix-4, and Country remain 100% UNCHANGED.

Configurations:
- E5-0: No address DF filtering (Control, exact E0)
- E5-A: Address Token DF > 20,000 -> ignore key
- E5-B: Address Token DF > 50,000 -> ignore key
- E5-C: Address Token DF > 100,000 -> ignore key
- E5-D: Address Token DF > 150,000 -> ignore key
- E5-E: Address Token DF > 200,000 -> ignore key

Dataset:
- 500 France S1 records
- 1,434,993 France target records (Source 2 + Source 3)
- Safety candidate cap = 400
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


def build_address_df_capped_engine(base_engine, addr_df_cap=None):
    """
    Creates an isolated clone of base_engine with DF filtering applied
    ONLY to Address Token. Name Token, Prefix-3, Prefix-4, and Country are UNTOUCHED.
    """
    if addr_df_cap is None:
        return base_engine

    cloned = copy.copy(base_engine)
    
    # Address Token ONLY is filtered
    cloned.country_addr_token_idx = defaultdict(lambda: defaultdict(list))
    for c, idx in base_engine.country_addr_token_idx.items():
        cloned.country_addr_token_idx[c] = {k: v for k, v in idx.items() if len(v) <= addr_df_cap}
        
    return cloned


def run_optimization_e5():
    print("=" * 70)
    print("OPTIMIZATION E5: ADDRESS-TOKEN-ONLY DF FILTERING EXPERIMENT")
    print("=" * 70)

    # Verify locked production fingerprint
    fp = get_config_fingerprint()
    print(f"Production Config Fingerprint: {fp}")
    assert fp == "87f20ceeb84ccc6ea2d48678c7810ac5", f"Fingerprint mismatch: {fp}"
    print(f"LOCKED ARCHITECTURE VERIFIED: 87f20ceeb84ccc6ea2d48678c7810ac5")

    # 1. Load France targets (1,434,993)
    print("\n[1/3] Loading 1,434,993 France targets (Source 2 + Source 3)...")
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
    print("\n[2/3] Building base inverted index (Config A)...")
    t0_eng = time.time()
    base_engine = ConfigABlockingEngine(targets)
    t_eng = time.time() - t0_eng
    proc = psutil.Process(os.getpid())
    print(f"  Index built in {t_eng:.2f}s | RSS: {proc.memory_info().rss / (1024*1024):.1f} MB")

    # 3. Load 500 France S1 Records
    print("\n[3/3] Loading 500 France S1 records...")
    s1_df = pd.read_csv('data/test/test_source1.tsv', sep='\t', dtype=str, nrows=10000, na_filter=False)
    s1_fr = s1_df[s1_df['country'].apply(normalize_country) == 'france'].head(500).copy().reset_index(drop=True)
    del s1_df
    gc.collect()

    s1_fr['norm_name'] = s1_fr['business_name'].apply(normalize_business_name)
    s1_fr['norm_address'] = s1_fr['business_address'].apply(normalize_business_address)
    s1_fr['norm_country'] = 'france'
    n_s1 = len(s1_fr)
    print(f"  Loaded {n_s1} France S1 records.")

    # Reference Ground-Truth: Exact Name Matches in France (4,191 true pairs across 406 S1 entities)
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

    # Target lookups for critical loss analysis
    target_dict = {}
    for idx in range(len(base_engine.target_eids)):
        eid = base_engine.target_eids[idx]
        target_dict[eid] = {
            'name': base_engine.target_names[idx],
            'addr': base_engine.target_addrs[idx],
            'idx': idx
        }
    s1_dict = s1_fr.set_index('entity_id').to_dict('index')

    configs = [
        ('E5-0', None, 'None (E0 Control)'),
        ('E5-A', 20000, '20,000'),
        ('E5-B', 50000, '50,000'),
        ('E5-C', 100000, '100,000'),
        ('E5-D', 150000, '150,000'),
        ('E5-E', 200000, '200,000'),
    ]

    results = {}
    capped_candidates_dict = {}
    raw_true_recovered_dict = {}

    print("\n" + "=" * 70)
    print("RUNNING E5-0 -> E5-E CONFIGURATIONS")
    print("=" * 70)

    for cfg_id, addr_cap, label in configs:
        print(f"\n>>> Running {cfg_id} (Address DF Cap: {label})...")
        t0_cfg = time.time()
        
        eng = build_address_df_capped_engine(base_engine, addr_df_cap=addr_cap)
        
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
        
        raw_count = metrics['raw_candidate_count']
        post_cap_count = metrics['post_cap_candidate_count']
        mean_raw = metrics['candidates_per_s1']
        median_raw = metrics['median_candidates_per_s1']
        max_raw = metrics['max_candidates_per_s1']
        zero_cand_s1 = metrics['zero_candidate_s1_count']
        mean_post_cap = post_cap_count / n_s1
        
        # Post-cap recall
        post_cap_pairs = set(zip(capped_df['entity_id_s1'], capped_df['entity_id_cand']))
        capped_candidates_dict[cfg_id] = post_cap_pairs
        true_recov_post = len(post_cap_pairs & exact_matches_set)
        post_cap_recall = true_recov_post / len(exact_matches_set) if exact_matches_set else 1.0
        
        # Raw true pairs: evaluate how many true pairs are recovered in candidate universe
        # Since Name Token, Prefix-3, Prefix-4 are untouched, exact matches have identical raw names!
        # Let's count how many true pairs are reached by any channel in eng
        c = 'france'
        raw_recov = 0
        name_idx = eng.country_name_token_idx[c]
        p3_idx = eng.country_prefix3_idx[c]
        p4_idx = eng.country_prefix4_idx[c]
        addr_idx = eng.country_addr_token_idx[c]
        
        for s1_id, r in s1_fr.iterrows():
            eid = r['entity_id']
            true_idx_set = true_target_indices_by_s1.get(eid, set())
            if not true_idx_set:
                continue
            name = r['norm_name']
            addr = r['norm_address']
            
            cands_set = set()
            for t in name.split():
                if len(t) > 1 and t not in eng.stop_tokens:
                    cands_set.update(name_idx.get(t, ()))
            if len(name) >= 3:
                cands_set.update(p3_idx.get(name[:3], ()))
            if len(name) >= 4:
                cands_set.update(p4_idx.get(name[:4], ()))
            if addr:
                for atok in set([t for t in addr.split() if len(t) >= 4 and t not in eng.stop_addrs]):
                    cands_set.update(addr_idx.get(atok, ()))
            raw_recov += len(cands_set & true_idx_set)
            
        raw_recall = raw_recov / len(exact_matches_set) if exact_matches_set else 1.0
        raw_true_recovered_dict[cfg_id] = raw_recov
        
        throughput = n_s1 / t_gen if t_gen > 0 else 0
        print(f"  {cfg_id} Done in {t_gen:.2f}s ({throughput:.2f} S1/s) | Raw: {raw_count:,} | Raw Rec: {raw_recall:.4%} | Post-Cap Rec: {post_cap_recall:.4%}")
        
        results[cfg_id] = {
            'config': cfg_id,
            'addr_df_cap': addr_cap,
            'label': label,
            'runtime_seconds': round(t_gen, 2),
            'total_runtime_seconds': round(t_total, 2),
            'peak_rss_mb': round(peak_mb, 1),
            'throughput_s1_per_sec': round(throughput, 2),
            'raw_candidates': raw_count,
            'mean_raw_candidates': round(mean_raw, 2),
            'median_raw_candidates': round(median_raw, 1),
            'max_raw_candidates': max_raw,
            'post_cap_candidates': post_cap_count,
            'mean_post_cap_candidates': round(mean_post_cap, 2),
            'ground_truth_pairs': len(exact_matches_set),
            'raw_true_pairs_recovered': raw_recov,
            'raw_recall': round(raw_recall, 6),
            'post_cap_true_pairs_recovered': true_recov_post,
            'post_cap_recall': round(post_cap_recall, 6),
            'zero_candidate_s1_count': zero_cand_s1
        }
        
        del eng, capped_df
        gc.collect()

    # Reductions vs E5-0
    e5_0_raw = results['E5-0']['raw_candidates']
    e5_0_time = results['E5-0']['runtime_seconds']
    e5_0_retained_pairs = capped_candidates_dict['E5-0']
    e5_0_true_post = e5_0_retained_pairs & exact_matches_set

    for cfg_id in ['E5-0', 'E5-A', 'E5-B', 'E5-C', 'E5-D', 'E5-E']:
        raw_red = (1.0 - results[cfg_id]['raw_candidates'] / e5_0_raw) * 100
        time_red = (1.0 - results[cfg_id]['runtime_seconds'] / e5_0_time) * 100 if e5_0_time > 0 else 0
        retained_v_e0 = len(capped_candidates_dict[cfg_id] & e5_0_retained_pairs)
        cfg_true_post = capped_candidates_dict[cfg_id] & exact_matches_set
        lost_vs_e5_0 = len(e5_0_true_post - cfg_true_post)
        
        results[cfg_id]['raw_candidate_reduction_pct'] = round(raw_red, 2)
        results[cfg_id]['runtime_reduction_pct'] = round(time_red, 2)
        results[cfg_id]['e5_0_candidates_retained'] = retained_v_e0
        results[cfg_id]['e5_0_retention_pct'] = round(retained_v_e0 / len(e5_0_retained_pairs) * 100, 2)
        results[cfg_id]['true_pairs_lost_vs_e5_0'] = lost_vs_e5_0

    # Critical Loss Analysis
    print("\n" + "=" * 70)
    print("CRITICAL LOSS ANALYSIS (TRUE PAIRS LOST RELATIVE TO E5-0)")
    print("=" * 70)

    loss_analysis = {}
    for cfg_id in ['E5-A', 'E5-B', 'E5-C', 'E5-D', 'E5-E']:
        cfg_true_post = capped_candidates_dict[cfg_id] & exact_matches_set
        lost_pairs = e5_0_true_post - cfg_true_post
        print(f"\n{cfg_id} (Address DF Cap: {results[cfg_id]['label']}): {len(lost_pairs)} True Pairs Lost vs E5-0")
        
        lost_details = []
        for s1_id, t_id in list(lost_pairs)[:20]:
            s1_info = s1_dict[s1_id]
            t_info = target_dict[t_id]
            s1_name = s1_info['norm_name']
            s1_addr = s1_info['norm_address']
            t_name = t_info['name']
            t_addr = t_info['addr']
            
            # Check which address tokens exceeded cap
            capped_tokens = []
            for atok in s1_addr.split():
                if atok in base_engine.country_addr_token_idx['france']:
                    df_val = len(base_engine.country_addr_token_idx['france'][atok])
                    if df_val > results[cfg_id]['addr_df_cap']:
                        capped_tokens.append(f"'{atok}' (DF={df_val:,})")
                        
            # Check other unchanged channels
            recovered_by_other = []
            for tok in s1_name.split():
                if tok in base_engine.country_name_token_idx['france'] and t_info['idx'] in base_engine.country_name_token_idx['france'][tok]:
                    recovered_by_other.append(f"NameToken('{tok}')")
            if len(s1_name) >= 3 and s1_name[:3] in base_engine.country_prefix3_idx['france'] and t_info['idx'] in base_engine.country_prefix3_idx['france'][s1_name[:3]]:
                recovered_by_other.append(f"Prefix3('{s1_name[:3]}')")
            if len(s1_name) >= 4 and s1_name[:4] in base_engine.country_prefix4_idx['france'] and t_info['idx'] in base_engine.country_prefix4_idx['france'][s1_name[:4]]:
                recovered_by_other.append(f"Prefix4('{s1_name[:4]}')")
                
            survived_cap_e0 = (s1_id, t_id) in e5_0_retained_pairs
            
            reason = f"Address token(s) {', '.join(capped_tokens)} exceeded cap {results[cfg_id]['label']}. "
            if recovered_by_other:
                reason += f"Pair WAS generated by {', '.join(recovered_by_other)} but evidence score dropped, falling out of top 400."
            else:
                reason += "Pair was not recovered by name channels and was dropped entirely."
                
            lost_details.append({
                's1_id': s1_id,
                'target_id': t_id,
                's1_name': s1_name,
                'target_name': t_name,
                's1_address': s1_addr,
                'target_address': t_addr,
                'address_tokens_responsible': capped_tokens,
                'recovered_by_unchanged_channels': recovered_by_other,
                'survived_e5_0_cap': survived_cap_e0,
                'reason': reason
            })
            print(f"  Lost Pair: {s1_id} <-> {t_id}")
            print(f"    Name: '{s1_name}' | Addr: '{s1_addr}'")
            print(f"    Capped Tokens: {capped_tokens}")
            print(f"    Other Channels: {recovered_by_other}")
            print(f"    Mechanism: {reason}")
            
        loss_analysis[cfg_id] = {
            'lost_count': len(lost_pairs),
            'sample_details': lost_details
        }

    output_data = {
        'benchmark_summary': {
            's1_count': n_s1,
            'target_count': n_targets,
            'candidate_cap': 400,
            'config_fingerprint': fp,
            'reference_true_pairs': len(exact_matches_set)
        },
        'configurations': results,
        'critical_loss_analysis': loss_analysis
    }

    out_path = 'experiments/optimization_e5_results.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nAll experimental results saved to {out_path}")

    # Summary Table
    print("\n" + "=" * 115)
    print(f"{'Config':<8}{'Addr DF Cap':<14}{'Raw Cands':<14}{'Raw Rec':<10}{'Post-Cap Cands':<16}{'Post-Cap Rec':<14}{'Runtime(s)':<12}{'Peak RSS':<10}{'Cand Reduc %':<14}")
    print("-" * 115)
    for cfg_id in ['E5-0', 'E5-A', 'E5-B', 'E5-C', 'E5-D', 'E5-E']:
        r = results[cfg_id]
        print(f"{r['config']:<8}{r['label']:<14}{r['raw_candidates']:<14,}{r['raw_recall']:<10.4%}{r['post_cap_candidates']:<16,}{r['post_cap_recall']:<14.4%}{r['runtime_seconds']:<12.2f}{r['peak_rss_mb']:<10.1f}{r['raw_candidate_reduction_pct']:<14.2f}%")
    print("=" * 115)

    return output_data


if __name__ == '__main__':
    run_optimization_e5()
