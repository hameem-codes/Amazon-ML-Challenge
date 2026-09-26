"""
Tournament Engine: High-Performance, Vectorized Implementation
Same Data — Same Evaluator — Same Metrics
"""

import os
import heapq
import itertools
from collections import defaultdict, Counter
import numpy as np
import pandas as pd

from config import BLOCKING_CONFIG, CANDIDATE_CAP_CONFIG, GENERIC_BUSINESS_TOKENS, GENERIC_ADDRESS_TOKENS

def load_tournament_data():
    s1_path = 'scratch/tournament_s1.tsv'
    targets_path = 'scratch/tournament_targets.tsv'
    gt_path = 'scratch/tournament_true_pairs.tsv'
    
    assert os.path.exists(s1_path), f"Missing {s1_path}"
    assert os.path.exists(targets_path), f"Missing {targets_path}"
    assert os.path.exists(gt_path), f"Missing {gt_path}"
    
    df_s1 = pd.read_csv(s1_path, sep='\t', dtype=str, na_filter=False)
    df_targets = pd.read_csv(targets_path, sep='\t', dtype=str, na_filter=False)
    df_gt = pd.read_csv(gt_path, sep='\t', dtype=str, na_filter=False)
    
    true_pairs_set = set(zip(df_gt['source1_entity_id'], df_gt['matched_entity_id']))
    return df_s1, df_targets, true_pairs_set

class TournamentEngine:
    def __init__(self, df_targets):
        self.df_targets = df_targets
        self.n_target = len(df_targets)
        self.target_eids = list(df_targets['entity_id'].values)
        self.target_names = list(df_targets['norm_name'].values)
        self.target_addrs = list(df_targets['norm_address'].values)
        self.target_countries = list(df_targets['norm_country'].values)
        self.target_sources = list(df_targets['source'].values)
        
        self._build_indices()
        
    def _get_informative_tokens(self, name):
        if not name:
            return []
        return [t for t in name.split() if t not in GENERIC_BUSINESS_TOKENS and len(t) > 1]
        
    def _get_ngrams(self, text, n=3):
        if not text:
            return []
        clean = text.replace(' ', '')
        if len(clean) < n:
            return [clean]
        return [clean[i:i+n] for i in range(len(clean)-n+1)]
        
    def _build_indices(self):
        print("  Building core inverted indices...", flush=True)
        # 1. Exact Name
        self.exact_name_idx = defaultdict(list)
        for i, name in enumerate(self.target_names):
            if name:
                self.exact_name_idx[name].append(i)
                
        # 2. Selective Token
        self.sel_token_idx = defaultdict(list)
        for i, name in enumerate(self.target_names):
            for t in set(self._get_informative_tokens(name)):
                self.sel_token_idx[t].append(i)
                
        # 3. Rare Token (<= 1% freq)
        max_rare = max(2, int(self.n_target * BLOCKING_CONFIG['rare_token_freq_cap']))
        self.rare_token_idx = {t: idxs for t, idxs in self.sel_token_idx.items() if len(idxs) <= max_rare}
        
        # 4. Token Pair
        self.pair_idx = defaultdict(list)
        for i, name in enumerate(self.target_names):
            toks = sorted(list(set(self._get_informative_tokens(name))))
            if len(toks) >= 2:
                for a, b in itertools.combinations(toks, 2):
                    self.pair_idx[f"{a}||{b}"].append(i)
                    
        # 5. Filtered Char 3-gram with pre-cast NumPy int32 arrays for high-speed bincounting
        ngram_full = defaultdict(list)
        for i, name in enumerate(self.target_names):
            for ng in set(self._get_ngrams(name, n=3)):
                ngram_full[ng].append(i)
        max_ng = max(10, int(self.n_target * BLOCKING_CONFIG['ngram_doc_freq_cap']))
        self.filtered_ngram_idx = {ng: np.array(idxs, dtype=np.int32) for ng, idxs in ngram_full.items() if len(idxs) <= max_ng}
        del ngram_full
        
        # 6. Team Baseline: Country-partitioned selective name tokens, prefix 3, prefix 4, address tokens
        self.country_name_token_idx = defaultdict(lambda: defaultdict(list))
        self.country_p3_idx = defaultdict(lambda: defaultdict(list))
        self.country_p4_idx = defaultdict(lambda: defaultdict(list))
        self.country_addr_token_idx = defaultdict(lambda: defaultdict(list))
        
        # Global Address Token Index (for Config C)
        self.addr_token_idx = defaultdict(list)
        
        for i in range(self.n_target):
            c = self.target_countries[i]
            name = self.target_names[i]
            addr = self.target_addrs[i]
            
            # Country-partitioned name tokens (selective, len > 1)
            for t in name.split():
                if len(t) > 1 and t not in GENERIC_BUSINESS_TOKENS:
                    self.country_name_token_idx[c][t].append(i)
                    
            # Country-partitioned prefix 3 & prefix 4
            if len(name) >= 3:
                self.country_p3_idx[c][name[:3]].append(i)
            if len(name) >= 4:
                self.country_p4_idx[c][name[:4]].append(i)
                
            # Address tokens
            if addr:
                addr_toks = set([t for t in addr.split() if t not in GENERIC_ADDRESS_TOKENS and len(t) >= 4])
                for t in addr_toks:
                    self.country_addr_token_idx[c][t].append(i)
                    self.addr_token_idx[t].append(i)
                    
        print("  Inverted indices built successfully.", flush=True)

    def run_blocking(self, df_s1, config='B', max_cap=400, true_pairs_set=None):
        """
        Runs blocking and candidate capping deterministically with vectorized execution.
        """
        s1_eids = list(df_s1['entity_id'].values)
        s1_names = list(df_s1['norm_name'].values)
        s1_addrs = list(df_s1['norm_address'].values)
        s1_countries = list(df_s1['norm_country'].values)
        
        min_ng = BLOCKING_CONFIG['min_shared_ngrams']
        total_true = len(true_pairs_set) if true_pairs_set else 0
        target_eids = self.target_eids
        
        s1_raw_counts = []
        overflowing_s1 = 0
        total_raw_candidates = 0
        total_post_cap_candidates = 0
        
        found_true_raw = set()
        found_true_post_cap = set()
        
        n_s1 = len(s1_eids)
        for s1_idx in range(n_s1):
            s1_id = s1_eids[s1_idx]
            name = s1_names[s1_idx]
            addr = s1_addrs[s1_idx]
            country = s1_countries[s1_idx]
            
            target_scores = defaultdict(int)
            
            # ---- CONFIG A CHANNELS ----
            if config in ('A', 'D'):
                # 1. Country + Name Token (selective)
                for t in name.split():
                    if len(t) > 1 and t not in GENERIC_BUSINESS_TOKENS:
                        for idx in self.country_name_token_idx[country].get(t, []):
                            target_scores[idx] += 70
                            
                # 2. Country + Prefix 4
                if len(name) >= 4:
                    p4 = name[:4]
                    for idx in self.country_p4_idx[country].get(p4, []):
                        target_scores[idx] += 50
                        
                # 3. Country + Prefix 3
                if len(name) >= 3:
                    p3 = name[:3]
                    for idx in self.country_p3_idx[country].get(p3, []):
                        target_scores[idx] += 30
                        
                # 4. Country + Address Token
                if addr:
                    a_toks = set([t for t in addr.split() if t not in GENERIC_ADDRESS_TOKENS and len(t) >= 4])
                    for t in a_toks:
                        for idx in self.country_addr_token_idx[country].get(t, []):
                            target_scores[idx] += 60

            # ---- CONFIG B CHANNELS (Production Multi-Key) ----
            if config in ('B', 'C', 'D'):
                # 1. Exact Name
                if name and name in self.exact_name_idx:
                    for idx in self.exact_name_idx[name]:
                        target_scores[idx] += 1000
                        
                inf_toks = self._get_informative_tokens(name)
                inf_set = set(inf_toks)
                
                # 2. Selective Token
                for t in inf_set:
                    if t in self.sel_token_idx:
                        for idx in self.sel_token_idx[t]:
                            target_scores[idx] += 105
                            
                # 3. Rare Token
                for t in inf_set:
                    if t in self.rare_token_idx:
                        for idx in self.rare_token_idx[t]:
                            target_scores[idx] += 115
                            
                # 4. Token Pair
                sorted_toks = sorted(list(inf_set))
                if len(sorted_toks) >= 2:
                    for a, b in itertools.combinations(sorted_toks, 2):
                        k = f"{a}||{b}"
                        if k in self.pair_idx:
                            for idx in self.pair_idx[k]:
                                target_scores[idx] += 120
                                
                # 5. Filtered Char 3-gram (Fast np.unique)
                ng_lists = [self.filtered_ngram_idx[ng] for ng in set(self._get_ngrams(name, n=3)) if ng in self.filtered_ngram_idx]
                if ng_lists:
                    all_idxs = np.concatenate(ng_lists) if len(ng_lists) > 1 else ng_lists[0]
                    if len(all_idxs) > 0:
                        u, counts = np.unique(all_idxs, return_counts=True)
                        hit_mask = counts >= min_ng
                        hit_idxs = u[hit_mask]
                        hit_counts = counts[hit_mask]
                        for idx, cnt in zip(hit_idxs, hit_counts):
                            target_scores[idx] += 100 + int(cnt) * 10

            # ---- CONFIG C ADDRESS TOKEN CHANNEL ----
            if config in ('C', 'D'):
                if addr:
                    a_toks = set([t for t in addr.split() if t not in GENERIC_ADDRESS_TOKENS and len(t) >= 4])
                    for t in a_toks:
                        if t in self.addr_token_idx:
                            for idx in self.addr_token_idx[t]:
                                target_scores[idx] += 80

            # Raw candidates for this S1
            n_raw_s1 = len(target_scores)
            s1_raw_counts.append(n_raw_s1)
            total_raw_candidates += n_raw_s1
            
            # Ground truth check on raw candidates
            for idx in target_scores:
                pair = (s1_id, target_eids[idx])
                if true_pairs_set and pair in true_pairs_set:
                    found_true_raw.add(pair)
                    
            # Apply Candidate Safety Cap (O(N log K) fast heapq)
            if n_raw_s1 > max_cap:
                overflowing_s1 += 1
                capped_indices = heapq.nlargest(max_cap, target_scores.keys(), key=target_scores.__getitem__)
            else:
                capped_indices = target_scores.keys()
                
            total_post_cap_candidates += len(capped_indices)
            for idx in capped_indices:
                pair = (s1_id, target_eids[idx])
                if true_pairs_set and pair in true_pairs_set:
                    found_true_post_cap.add(pair)
                    
            if (s1_idx + 1) % 2500 == 0:
                print(f"    [{config}] Processed {s1_idx + 1:,} / {n_s1:,} S1 entities...", flush=True)
                    
        # Metrics summary
        raw_found = len(found_true_raw)
        raw_missed = total_true - raw_found
        raw_recall = (raw_found / total_true * 100) if total_true > 0 else 0.0
        
        post_cap_found = len(found_true_post_cap)
        post_cap_recall = (post_cap_found / total_true * 100) if total_true > 0 else 0.0
        cap_retention = (post_cap_found / raw_found * 100) if raw_found > 0 else 100.0
        
        avg_s1 = float(np.mean(s1_raw_counts))
        med_s1 = float(np.median(s1_raw_counts))
        max_s1 = int(np.max(s1_raw_counts)) if s1_raw_counts else 0
        zero_s1 = int(sum(1 for c in s1_raw_counts if c == 0))
        cand_eff = float(total_raw_candidates / raw_found) if raw_found > 0 else float('inf')
        
        metrics = {
            'candidates': int(total_raw_candidates),
            'avg_s1': float(avg_s1),
            'median_s1': float(med_s1),
            'max_s1': int(max_s1),
            'zero_s1': int(zero_s1),
            'true_found': int(raw_found),
            'true_missed': int(raw_missed),
            'recall': float(raw_recall),
            'post_cap_candidates': int(total_post_cap_candidates),
            'overflowing_s1': int(overflowing_s1),
            'post_cap_recall': float(post_cap_recall),
            'cap_retention': float(cap_retention),
            'cand_efficiency': float(cand_eff)
        }
        
        return metrics, found_true_raw, found_true_post_cap
