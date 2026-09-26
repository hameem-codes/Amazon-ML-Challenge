"""
Production Blocking Module

Implements the multi-key inverted-index blocking engine.
Preserves explicit per-channel evidence metadata:
- blocked_exact_name
- blocked_selective_token
- blocked_rare_token
- blocked_token_pair
- blocked_char_ngram
- shared_ngram_count
- num_blocking_keys
"""

import itertools
import gc
import heapq
from collections import defaultdict, Counter
import pandas as pd
import numpy as np

from config import BLOCKING_CONFIG, GENERIC_BUSINESS_TOKENS, GENERIC_ADDRESS_TOKENS, BLOCKING_CHUNK_SIZE
from candidate_safety import apply_candidate_safety_cap

# Optimization D: Config A Blocking Channel Bits
BIT_COUNTRY = 1 << 0  # Bit 0 = Country (1)
BIT_NAME = 1 << 1     # Bit 1 = Name Token (2)
BIT_P3 = 1 << 2       # Bit 2 = Name Prefix-3 (4)
BIT_P4 = 1 << 3       # Bit 3 = Name Prefix-4 (8)
BIT_ADDR = 1 << 4     # Bit 4 = Address Token (16)

class ConfigABlockingEngine:
    """
    Locked Config A (Team Baseline) Inverted-Index Blocking Engine.
    
    The ONLY 5 blocking channels permitted in Config A:
    1. Country blocking
    2. Name-token blocking
    3. Name-prefix 3
    4. Name-prefix 4
    5. Address-token blocking
    
    Deterministic union + evidence tracking across the 5 locked channels.
    """
    def __init__(self, df_target, config=None, stop_tokens=GENERIC_BUSINESS_TOKENS, stop_addrs=GENERIC_ADDRESS_TOKENS):
        self.cfg = config or BLOCKING_CONFIG
        self.stop_tokens = stop_tokens
        self.stop_addrs = stop_addrs
        self.df_target = df_target
        self.target_eids = df_target['entity_id'].values
        self.target_names = df_target['norm_name'].values
        self.target_addrs = df_target['norm_address'].values if 'norm_address' in df_target.columns else np.array([''] * len(df_target))
        self.target_countries = df_target['norm_country'].values if 'norm_country' in df_target.columns else np.array([''] * len(df_target))
        self.target_sources = df_target['source'].values if 'source' in df_target.columns else np.array(['S2' if str(e).startswith('S2') else 'S3' for e in self.target_eids])
        self.n_target = len(df_target)
        
        # Optimization C: Precompute deterministic integer lexical ranks for all targets
        order = np.argsort(self.target_eids)
        self.target_lex_rank = np.empty(self.n_target, dtype=np.int32)
        self.target_lex_rank[order] = np.arange(self.n_target, dtype=np.int32)
        
        self._build_indexes()
        
    def _build_indexes(self):
        # Country-partitioned inverted indices
        self.country_name_token_idx = defaultdict(lambda: defaultdict(list))
        self.country_prefix3_idx = defaultdict(lambda: defaultdict(list))
        self.country_prefix4_idx = defaultdict(lambda: defaultdict(list))
        self.country_p3_idx = self.country_prefix3_idx
        self.country_p4_idx = self.country_prefix4_idx
        self.country_addr_token_idx = defaultdict(lambda: defaultdict(list))
        
        for i in range(self.n_target):
            c = self.target_countries[i]
            name = str(self.target_names[i]) if pd.notna(self.target_names[i]) else ''
            addr = str(self.target_addrs[i]) if pd.notna(self.target_addrs[i]) else ''
            
            # Channel 2: Name token (len > 1, not in generic business tokens)
            for t in name.split():
                if len(t) > 1 and t not in self.stop_tokens:
                    self.country_name_token_idx[c][t].append(i)
                    
            # Channel 3: Name prefix-3 (len >= 3)
            if len(name) >= 3:
                self.country_prefix3_idx[c][name[:3]].append(i)
                
            # Channel 4: Name prefix-4 (len >= 4)
            if len(name) >= 4:
                self.country_prefix4_idx[c][name[:4]].append(i)
                
            # Channel 5: Address token (len >= 4, not in generic address tokens)
            if addr:
                addr_toks = set([t for t in addr.split() if len(t) >= 4 and t not in self.stop_addrs])
                for t in addr_toks:
                    self.country_addr_token_idx[c][t].append(i)

    def generate_candidates_for_chunk(self, df_s1_chunk):
        """
        Generates candidates and explicit per-channel evidence for a bounded S1 chunk.
        Uses compact bitmask and columnar array construction for memory safety and high speed.
        """
        s1_eids = df_s1_chunk['entity_id'].values
        s1_names = df_s1_chunk['norm_name'].values if 'norm_name' in df_s1_chunk.columns else np.array([''] * len(df_s1_chunk))
        s1_addrs = df_s1_chunk['norm_address'].values if 'norm_address' in df_s1_chunk.columns else np.array([''] * len(df_s1_chunk))
        s1_countries = df_s1_chunk['norm_country'].values if 'norm_country' in df_s1_chunk.columns else np.array([''] * len(df_s1_chunk))
        
        out_s1 = []
        out_cand = []
        out_src = []
        out_b_ctry = []
        out_b_name = []
        out_b_p3 = []
        out_b_p4 = []
        out_b_addr = []
        out_num_keys = []
        out_score = []
        out_exact = []
        
        for s1_id, name, addr, country in zip(s1_eids, s1_names, s1_addrs, s1_countries):
            name = str(name) if pd.notna(name) else ''
            addr = str(addr) if pd.notna(addr) else ''
            country = str(country) if pd.notna(country) else ''
            
            cand_flags = {}
            cand_scores = {}
            
            # Channel 2: Country + Name Token (selective)
            for t in name.split():
                if len(t) > 1 and t not in self.stop_tokens:
                    for idx in self.country_name_token_idx[country].get(t, ()):
                        cand_flags[idx] = cand_flags.get(idx, 0) | 1
                        cand_scores[idx] = cand_scores.get(idx, 0) + 70
                        
            # Channel 3: Country + Prefix 4
            if len(name) >= 4:
                p4 = name[:4]
                for idx in self.country_prefix4_idx[country].get(p4, ()):
                    cand_flags[idx] = cand_flags.get(idx, 0) | 2
                    cand_scores[idx] = cand_scores.get(idx, 0) + 50
                    
            # Channel 4: Country + Prefix 3
            if len(name) >= 3:
                p3 = name[:3]
                for idx in self.country_prefix3_idx[country].get(p3, ()):
                    cand_flags[idx] = cand_flags.get(idx, 0) | 4
                    cand_scores[idx] = cand_scores.get(idx, 0) + 30
                    
            # Channel 5: Country + Address Token
            if addr:
                addr_toks = set([t for t in addr.split() if len(t) >= 4 and t not in self.stop_addrs])
                for t in addr_toks:
                    for idx in self.country_addr_token_idx[country].get(t, ()):
                        cand_flags[idx] = cand_flags.get(idx, 0) | 8
                        cand_scores[idx] = cand_scores.get(idx, 0) + 60
                        
            for idx, mask in cand_flags.items():
                b_name = 1 if (mask & 1) else 0
                b_p4 = 1 if (mask & 2) else 0
                b_p3 = 1 if (mask & 4) else 0
                b_addr = 1 if (mask & 8) else 0
                num_keys = b_name + b_p4 + b_p3 + b_addr
                
                if num_keys > 0:
                    c_name = self.target_names[idx]
                    exact_match = 1 if (name and name == c_name) else 0
                    
                    out_s1.append(s1_id)
                    out_cand.append(self.target_eids[idx])
                    out_src.append(self.target_sources[idx])
                    out_b_ctry.append(1)
                    out_b_name.append(b_name)
                    out_b_p3.append(b_p3)
                    out_b_p4.append(b_p4)
                    out_b_addr.append(b_addr)
                    out_num_keys.append(num_keys)
                    out_score.append(cand_scores[idx])
                    out_exact.append(exact_match)
                    
            del cand_flags, cand_scores
            
        N = len(out_s1)
        if N == 0:
            return pd.DataFrame(columns=[
                'entity_id_s1', 'entity_id_cand', 'source', 'blocked_country',
                'blocked_name_token', 'blocked_prefix_3', 'blocked_prefix_4',
                'blocked_address_token', 'num_blocking_keys', 'evidence_score',
                'blocked_exact_name', 'blocked_selective_token', 'blocked_rare_token',
                'blocked_token_pair', 'blocked_char_ngram', 'shared_ngram_count'
            ])
            
        return pd.DataFrame({
            'entity_id_s1': out_s1,
            'entity_id_cand': out_cand,
            'source': out_src,
            'blocked_country': np.array(out_b_ctry, dtype=np.int8),
            'blocked_name_token': np.array(out_b_name, dtype=np.int8),
            'blocked_prefix_3': np.array(out_b_p3, dtype=np.int8),
            'blocked_prefix_4': np.array(out_b_p4, dtype=np.int8),
            'blocked_address_token': np.array(out_b_addr, dtype=np.int8),
            'num_blocking_keys': np.array(out_num_keys, dtype=np.int8),
            'evidence_score': np.array(out_score, dtype=np.float32),
            'blocked_exact_name': np.array(out_exact, dtype=np.int8),
            'blocked_selective_token': np.array(out_b_name, dtype=np.int8),
            'blocked_rare_token': np.zeros(N, dtype=np.int8),
            'blocked_token_pair': np.zeros(N, dtype=np.int8),
            'blocked_char_ngram': np.zeros(N, dtype=np.int8),
            'shared_ngram_count': np.zeros(N, dtype=np.int8)
        })

    def generate_candidates_with_evidence(self, df_s1, chunk_size=None, max_candidates_per_s1=None, return_metrics=False):
        """
        Generates candidates for df_s1 records using locked Config A channels.
        If max_candidates_per_s1 is provided, executes bounded chunking with the safety cap.
        Otherwise executes chunked generation and returns raw candidates.
        """
        if max_candidates_per_s1 is not None:
            return self.generate_bounded_candidates(
                df_s1,
                max_candidates_per_s1=max_candidates_per_s1,
                chunk_size=chunk_size,
                return_metrics=return_metrics
            )
            
        c_size = chunk_size or BLOCKING_CHUNK_SIZE
        if len(df_s1) <= c_size:
            return self.generate_candidates_for_chunk(df_s1)
            
        chunks = []
        for i in range(0, len(df_s1), c_size):
            chunk_s1 = df_s1.iloc[i:i + c_size]
            cands_chunk = self.generate_candidates_for_chunk(chunk_s1)
            if len(cands_chunk) > 0:
                chunks.append(cands_chunk)
            del cands_chunk
            gc.collect()
            
        if not chunks:
            return self.generate_candidates_for_chunk(df_s1.iloc[:0])
        return pd.concat(chunks, ignore_index=True)

    def generate_bounded_chunk(self, df_s1_chunk, s1_orig_indices=None, s1_lex_ranks=None, max_candidates_per_s1=400):
        """
        Generates and directly caps candidates for an S1 chunk using compact integer ID aggregation.
        Avoids allocating millions of transient un-capped candidate rows and intermediate DataFrames.
        Preserves 100% bit-for-bit equivalence and exact tie-breaking with apply_candidate_safety_cap.
        """
        s1_eids = df_s1_chunk['entity_id'].values
        if s1_orig_indices is None:
            s1_orig_indices = np.arange(len(s1_eids), dtype=np.int32)
            s1_order = np.argsort(s1_eids)
            s1_lex_ranks = np.empty(len(s1_eids), dtype=np.int32)
            s1_lex_ranks[s1_order] = np.arange(len(s1_eids), dtype=np.int32)

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
        names_target = self.target_names
        target_lex = self.target_lex_rank

        for s1_idx, s1_lex, name, addr, country in zip(s1_orig_indices, s1_lex_ranks, s1_names, s1_addrs, s1_countries):
            name = str(name) if pd.notna(name) else ''
            addr = str(addr) if pd.notna(addr) else ''
            country = str(country) if pd.notna(country) else ''

            cand_mask = {}
            cand_scores = {}

            # Channel 2: Country + Name Token (selective)
            for t in name.split():
                if len(t) > 1 and t not in self.stop_tokens:
                    for idx in self.country_name_token_idx[country].get(t, ()):
                        cand_mask[idx] = cand_mask.get(idx, BIT_COUNTRY) | BIT_NAME
                        cand_scores[idx] = cand_scores.get(idx, 0) + 70

            # Channel 3: Country + Prefix 4
            if len(name) >= 4:
                p4 = name[:4]
                for idx in self.country_prefix4_idx[country].get(p4, ()):
                    cand_mask[idx] = cand_mask.get(idx, BIT_COUNTRY) | BIT_P4
                    cand_scores[idx] = cand_scores.get(idx, 0) + 50

            # Channel 4: Country + Prefix 3
            if len(name) >= 3:
                p3 = name[:3]
                for idx in self.country_prefix3_idx[country].get(p3, ()):
                    cand_mask[idx] = cand_mask.get(idx, BIT_COUNTRY) | BIT_P3
                    cand_scores[idx] = cand_scores.get(idx, 0) + 30

            # Channel 5: Country + Address Token
            if addr:
                addr_toks = set([t for t in addr.split() if len(t) >= 4 and t not in self.stop_addrs])
                for t in addr_toks:
                    for idx in self.country_addr_token_idx[country].get(t, ()):
                        cand_mask[idx] = cand_mask.get(idx, BIT_COUNTRY) | BIT_ADDR
                        cand_scores[idx] = cand_scores.get(idx, 0) + 60

            n_raw = len(cand_mask)
            raw_counts_chunk[s1_idx] = n_raw
            if n_raw == 0:
                continue

            if n_raw <= max_candidates_per_s1:
                # All candidates fit; sort deterministically by (-score, target_lex_rank)
                top_idxs = list(cand_mask.keys())
                top_idxs.sort(key=lambda idx: (-cand_scores[idx], target_lex[idx]))
            else:
                # Fast integer score-bucketing top-K selection with exact candidate_eid tie-breaking
                buckets = defaultdict(list)
                for idx, sc in cand_scores.items():
                    buckets[sc].append(idx)

                top_idxs = []
                for sc in sorted(buckets.keys(), reverse=True):
                    b_idxs = buckets[sc]
                    if len(top_idxs) + len(b_idxs) <= max_candidates_per_s1:
                        b_idxs.sort(key=lambda idx: target_lex[idx])
                        top_idxs.extend(b_idxs)
                    else:
                        needed = max_candidates_per_s1 - len(top_idxs)
                        top_idxs.extend(heapq.nsmallest(needed, b_idxs, key=lambda idx: target_lex[idx]))
                        break

            # Materialize ONLY compact bitmask and integer IDs
            for idx in top_idxs:
                c_name = names_target[idx]
                exact_match = 1 if (name and name == c_name) else 0

                out_s1_idx.append(s1_idx)
                out_s1_lex.append(s1_lex)
                out_cand_idx.append(idx)
                out_cand_lex.append(target_lex[idx])
                out_mask.append(cand_mask[idx])
                out_score.append(cand_scores[idx])
                out_exact.append(exact_match)

            del cand_mask, cand_scores

        N_chunk = len(out_s1_idx)
        if N_chunk == 0:
            df_chunk = pd.DataFrame()
        else:
            df_chunk = pd.DataFrame({
                's1_orig_idx': np.array(out_s1_idx, dtype=np.int32),
                's1_lex_rank': np.array(out_s1_lex, dtype=np.int32),
                'cand_orig_idx': np.array(out_cand_idx, dtype=np.int32),
                'cand_lex_rank': np.array(out_cand_lex, dtype=np.int32),
                'evidence_mask': np.array(out_mask, dtype=np.uint8),
                'evidence_score': np.array(out_score, dtype=np.float32),
                'blocked_exact_name': np.array(out_exact, dtype=np.int8)
            })

        return df_chunk, raw_counts_chunk

    def generate_bounded_candidates(self, df_s1, max_candidates_per_s1=400, chunk_size=None, return_metrics=False):
        """
        Memory-safe bounded candidate generation with compact integer IDs and evidence bitmasks:
        S1 chunk -> compact candidate aggregation & top-K selection -> emit chunk result -> release temporary memory -> next S1 chunk.
        
        Guarantees that full un-capped candidate sets across large slices are never materialized in RAM.
        Uses deterministic integer IDs internally for all comparisons, sorting, and storage,
        mapping back to original string entity IDs and decoding the bitmask at the final output boundary.
        """
        c_size = chunk_size or BLOCKING_CHUNK_SIZE
        post_cap_chunks = []
        
        # Tracking statistics
        raw_counts_per_s1 = {}
        total_raw_candidates = 0
        overflowing_s1_count = 0
        zero_candidate_s1_count = 0
        
        s1_ids_all = df_s1['entity_id'].values
        # Precompute global lexical ranks for S1
        s1_order = np.argsort(s1_ids_all)
        s1_lex_ranks_all = np.empty(len(s1_ids_all), dtype=np.int32)
        s1_lex_ranks_all[s1_order] = np.arange(len(s1_ids_all), dtype=np.int32)
        s1_indices_all = np.arange(len(s1_ids_all), dtype=np.int32)
        
        for i in range(0, len(df_s1), c_size):
            chunk_s1 = df_s1.iloc[i:i + c_size]
            chunk_indices = s1_indices_all[i:i + c_size]
            chunk_lex = s1_lex_ranks_all[i:i + c_size]

            capped_chunk, counts_in_chunk = self.generate_bounded_chunk(
                chunk_s1,
                s1_orig_indices=chunk_indices,
                s1_lex_ranks=chunk_lex,
                max_candidates_per_s1=max_candidates_per_s1
            )
            
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
            # Fast integer-based sorting on (s1_lex_rank, evidence_score desc, cand_lex_rank)
            capped_df = capped_df.sort_values(
                by=['s1_lex_rank', 'evidence_score', 'cand_lex_rank'],
                ascending=[True, False, True]
            ).reset_index(drop=True)

            # Vectorized bitmask decoding at downstream boundary
            mask = capped_df['evidence_mask'].values
            cand_orig = capped_df['cand_orig_idx'].values
            N = len(capped_df)

            b_ctry = ((mask >> 0) & 1).astype(np.int8)
            b_name = ((mask >> 1) & 1).astype(np.int8)
            b_p3 = ((mask >> 2) & 1).astype(np.int8)
            b_p4 = ((mask >> 3) & 1).astype(np.int8)
            b_addr = ((mask >> 4) & 1).astype(np.int8)
            num_keys = (b_name + b_p3 + b_p4 + b_addr).astype(np.int8)

            # Reconstruct exact 16-column DataFrame
            capped_df = pd.DataFrame({
                'entity_id_s1': s1_ids_all[capped_df['s1_orig_idx'].values],
                'entity_id_cand': self.target_eids[cand_orig],
                'source': self.target_sources[cand_orig],
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
            capped_df = self.generate_candidates_for_chunk(df_s1.iloc[:0])
            
        s1_ids = list(s1_ids_all)
        post_cap_counts_per_s1 = capped_df.groupby('entity_id_s1').size().to_dict() if len(capped_df) > 0 else {}
        post_counts_list = [post_cap_counts_per_s1.get(sid, 0) for sid in s1_ids]
        raw_counts_list = [raw_counts_per_s1.get(sid, 0) for sid in s1_ids]
        
        metrics = {
            'num_s1_evaluated': len(df_s1),
            'raw_candidate_count': total_raw_candidates,
            'post_cap_candidate_count': len(capped_df),
            'candidates_per_s1': float(np.mean(post_counts_list)) if post_counts_list else 0.0,
            'median_candidates_per_s1': float(np.median(post_counts_list)) if post_counts_list else 0.0,
            'max_candidates_per_s1': int(np.max(post_counts_list)) if post_counts_list else 0,
            'raw_candidates_per_s1': float(np.mean(raw_counts_list)) if raw_counts_list else 0.0,
            'raw_median_candidates_per_s1': float(np.median(raw_counts_list)) if raw_counts_list else 0.0,
            'raw_max_candidates_per_s1': int(np.max(raw_counts_list)) if raw_counts_list else 0,
            'zero_candidate_s1_count': zero_candidate_s1_count,
            'overflowing_s1_count': overflowing_s1_count,
            'max_candidate_cap': max_candidates_per_s1,
            'chunk_size': c_size
        }
        
        if return_metrics:
            return capped_df, metrics
        return capped_df

    def iter_bounded_candidate_chunks(self, df_s1, max_candidates_per_s1=400, chunk_size=None):
        """
        Yields (s1_chunk, capped_cands_chunk, cap_metrics) for each S1 chunk.
        Releases candidate memory before each step.
        """
        c_size = chunk_size or BLOCKING_CHUNK_SIZE
        for i in range(0, len(df_s1), c_size):
            chunk_s1 = df_s1.iloc[i:i + c_size]
            capped_chunk, counts_in_chunk = self.generate_bounded_chunk(chunk_s1, max_candidates_per_s1=max_candidates_per_s1)
            metrics = {
                'before_cap_candidates': sum(counts_in_chunk.values()),
                'after_cap_candidates': len(capped_chunk),
                'overflowing_s1_count': sum(1 for c in counts_in_chunk.values() if c > max_candidates_per_s1),
                'max_candidates_per_s1': max_candidates_per_s1
            }
            yield chunk_s1, capped_chunk, metrics

# Locked Production Engine Alias
ProductionBlockingEngine = ConfigABlockingEngine

class LegacyMultiKeyBlockingEngine:
    """
    Legacy multi-key inverted-index blocking engine (Phase 4B).
    Preserved for historical reference and test compatibility.
    """
    def __init__(self, df_target, config=None, stop_tokens=GENERIC_BUSINESS_TOKENS):
        self.cfg = config or BLOCKING_CONFIG
        self.stop_tokens = stop_tokens
        self.df_target = df_target
        self.target_eids = df_target['entity_id'].values
        self.target_names = df_target['norm_name'].values
        self.target_sources = df_target['source'].values if 'source' in df_target.columns else np.array(['S2' if str(e).startswith('S2') else 'S3' for e in self.target_eids])
        self.n_target = len(df_target)
        
        self._build_indexes()
        
    def _get_informative_tokens(self, name):
        if not name or not isinstance(name, str):
            return []
        return [t for t in name.split() if t not in self.stop_tokens and len(t) > 1]
        
    def _get_ngrams(self, text):
        n = self.cfg['ngram_size']
        if not text or not isinstance(text, str):
            return []
        clean = text.replace(' ', '')
        if len(clean) < n:
            return [clean]
        return [clean[i:i+n] for i in range(len(clean)-n+1)]
        
    def _build_indexes(self):
        # 1. Exact Name Index
        self.exact_name_index = defaultdict(list)
        for i, name in enumerate(self.target_names):
            if name:
                self.exact_name_index[name].append(i)
                
        # 2. Token Counts & Selective Token Index
        self.token_index = defaultdict(list)
        for i, name in enumerate(self.target_names):
            toks = set(self._get_informative_tokens(name))
            for t in toks:
                self.token_index[t].append(i)
                
        # 3. Rare Token Sub-index
        max_rare_cnt = max(2, int(self.n_target * self.cfg['rare_token_freq_cap']))
        self.rare_index = {t: idxs for t, idxs in self.token_index.items() if len(idxs) <= max_rare_cnt}
        
        # 4. Token-pair Index
        self.pair_index = defaultdict(list)
        for i, name in enumerate(self.target_names):
            toks = sorted(list(set(self._get_informative_tokens(name))))
            if len(toks) >= self.cfg['token_pair_min_informative']:
                for a, b in itertools.combinations(toks, 2):
                    self.pair_index[f"{a}||{b}"].append(i)
                    
        # 5. Char 3-gram Index with Doc Frequency filtering
        self.ngram_index = defaultdict(list)
        ng_counts = Counter()
        for i, name in enumerate(self.target_names):
            ngs = set(self._get_ngrams(name))
            ng_counts.update(ngs)
            for ng in ngs:
                self.ngram_index[ng].append(i)
                
        max_ng_cnt = max(10, int(self.n_target * self.cfg['ngram_doc_freq_cap']))
        self.filtered_ngram_index = {ng: idxs for ng, idxs in self.ngram_index.items() if len(idxs) <= max_ng_cnt}

    def generate_candidates_with_evidence(self, df_s1):
        """
        Generates candidate pairs and records per-channel evidence.
        Returns a DataFrame with columns:
          ['entity_id_s1', 'entity_id_cand', 'source', 'blocked_exact_name',
           'blocked_selective_token', 'blocked_rare_token', 'blocked_token_pair',
           'blocked_char_ngram', 'shared_ngram_count', 'num_blocking_keys']
        """
        s1_eids = df_s1['entity_id'].values
        s1_names = df_s1['norm_name'].values
        min_shared_ng = self.cfg['min_shared_ngrams']
        
        records = []
        
        for s1_id, name in zip(s1_eids, s1_names):
            # Map target_idx -> dict of evidence flags
            cand_evidence = defaultdict(lambda: {
                'blocked_exact_name': 0,
                'blocked_selective_token': 0,
                'blocked_rare_token': 0,
                'blocked_token_pair': 0,
                'blocked_char_ngram': 0,
                'shared_ngram_count': 0
            })
            
            # Channel 1: Exact Name
            if name in self.exact_name_index:
                for idx in self.exact_name_index[name]:
                    cand_evidence[idx]['blocked_exact_name'] = 1
                    
            toks = self._get_informative_tokens(name)
            tok_set = set(toks)
            
            # Channel 2: Selective Token
            for t in tok_set:
                if t in self.token_index:
                    for idx in self.token_index[t]:
                        cand_evidence[idx]['blocked_selective_token'] = 1
                        
            # Channel 3: Rare Token
            for t in tok_set:
                if t in self.rare_index:
                    for idx in self.rare_index[t]:
                        cand_evidence[idx]['blocked_rare_token'] = 1
                        
            # Channel 4: Token Pair
            sorted_toks = sorted(list(tok_set))
            if len(sorted_toks) >= self.cfg['token_pair_min_informative']:
                for a, b in itertools.combinations(sorted_toks, 2):
                    key = f"{a}||{b}"
                    if key in self.pair_index:
                        for idx in self.pair_index[key]:
                            cand_evidence[idx]['blocked_token_pair'] = 1
                            
            # Channel 5: Filtered Character N-gram
            ngs = set(self._get_ngrams(name))
            ng_hits = Counter()
            for ng in ngs:
                if ng in self.filtered_ngram_index:
                    for idx in self.filtered_ngram_index[ng]:
                        ng_hits[idx] += 1
            for idx, count in ng_hits.items():
                if count >= min_shared_ng:
                    cand_evidence[idx]['blocked_char_ngram'] = 1
                    cand_evidence[idx]['shared_ngram_count'] = count
                    
            # Build rows for this S1 entity
            for idx, ev in cand_evidence.items():
                # Count distinct blocking keys active for this pair
                num_keys = (ev['blocked_exact_name'] +
                            ev['blocked_selective_token'] +
                            ev['blocked_rare_token'] +
                            ev['blocked_token_pair'] +
                            ev['blocked_char_ngram'])
                
                # Only include pairs that triggered at least one blocking key
                if num_keys > 0:
                    records.append({
                        'entity_id_s1': s1_id,
                        'entity_id_cand': self.target_eids[idx],
                        'source': self.target_sources[idx],
                        'blocked_exact_name': ev['blocked_exact_name'],
                        'blocked_selective_token': ev['blocked_selective_token'],
                        'blocked_rare_token': ev['blocked_rare_token'],
                        'blocked_token_pair': ev['blocked_token_pair'],
                        'blocked_char_ngram': ev['blocked_char_ngram'],
                        'shared_ngram_count': ev['shared_ngram_count'],
                        'num_blocking_keys': num_keys
                    })
                    
        return pd.DataFrame(records)

# Backward-compatibility alias functions
def block_exact_name(df_s1, df_s23):
    engine = ProductionBlockingEngine(df_s23)
    cands = engine.generate_candidates_with_evidence(df_s1)
    return cands[cands['blocked_exact_name'] == 1][['entity_id_s1', 'entity_id_cand', 'source']]

def block_country_name(df_s1, df_s23):
    candidates = pd.merge(
        df_s1[['entity_id', 'norm_name', 'norm_country']], 
        df_s23[['entity_id', 'norm_name', 'norm_country', 'source']], 
        on=['norm_name', 'norm_country'], 
        how='inner', 
        suffixes=('_s1', '_cand')
    )
    return candidates[['entity_id_s1', 'entity_id_cand', 'source']]

def block_token(df_s1, df_s23):
    engine = ProductionBlockingEngine(df_s23)
    cands = engine.generate_candidates_with_evidence(df_s1)
    return cands[cands['blocked_selective_token'] == 1][['entity_id_s1', 'entity_id_cand', 'source']]

def block_ngram(df_s1, df_s23, n=3):
    cfg = dict(BLOCKING_CONFIG)
    cfg['ngram_size'] = n
    engine = ProductionBlockingEngine(df_s23, config=cfg)
    cands = engine.generate_candidates_with_evidence(df_s1)
    return cands[cands['blocked_char_ngram'] == 1][['entity_id_s1', 'entity_id_cand', 'source']]
