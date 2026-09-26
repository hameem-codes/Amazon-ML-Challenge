import sys, time, gc, heapq
from collections import defaultdict
sys.path.insert(0, 'src')
import pandas as pd
import numpy as np
from normalize import normalize_business_name, normalize_country, normalize_business_address
from blocking import ConfigABlockingEngine

# Load France targets
s2_chunks = [chunk[chunk['country'].apply(normalize_country) == 'france'] for chunk in pd.read_csv('data/test/test_source2.tsv', sep='\t', dtype=str, na_filter=False, chunksize=500000)]
s3_chunks = [chunk[chunk['country'].apply(normalize_country) == 'france'] for chunk in pd.read_csv('data/test/test_source3.tsv', sep='\t', dtype=str, na_filter=False, chunksize=500000)]
targets = pd.concat(s2_chunks + s3_chunks, ignore_index=True)
targets['norm_name'] = targets['business_name'].apply(normalize_business_name)
targets['norm_address'] = targets['business_address'].apply(normalize_business_address)
targets['norm_country'] = 'france'
targets['source'] = ['S2' if i < 703378 else 'S3' for i in range(len(targets))]
engine = ConfigABlockingEngine(targets)

# Load 25 France S1
s1_df = pd.read_csv('data/test/test_source1.tsv', sep='\t', dtype=str, nrows=5000, na_filter=False)
s1_fr = s1_df[s1_df['country'].apply(normalize_country) == 'france'].head(25).copy().reset_index(drop=True)
s1_fr['norm_name'] = s1_fr['business_name'].apply(normalize_business_name)
s1_fr['norm_address'] = s1_fr['business_address'].apply(normalize_business_address)
s1_fr['norm_country'] = 'france'

# 1. Baseline: Current Optimization B
t0 = time.time()
res_b, metrics_b = engine.generate_bounded_candidates(s1_fr, max_candidates_per_s1=400, chunk_size=25, return_metrics=True)
t_b = time.time() - t0
print(f"Opt B Baseline: {t_b:.3f}s | Candidates: {len(res_b):,}")

# 2. Optimization C: Integer-based candidate processing
class OptCEngine(ConfigABlockingEngine):
    def __init__(self, df_target, config=None):
        super().__init__(df_target, config=config)
        # Precompute deterministic integer lexical rank for all targets
        order = np.argsort(self.target_eids)
        self.target_lex_rank = np.empty(self.n_target, dtype=np.int32)
        self.target_lex_rank[order] = np.arange(self.n_target, dtype=np.int32)

    def generate_bounded_chunk_opt_c(self, df_s1_chunk, s1_orig_indices, s1_lex_ranks, max_candidates_per_s1=400):
        s1_names = df_s1_chunk['norm_name'].values if 'norm_name' in df_s1_chunk.columns else np.array([''] * len(df_s1_chunk))
        s1_addrs = df_s1_chunk['norm_address'].values if 'norm_address' in df_s1_chunk.columns else np.array([''] * len(df_s1_chunk))
        s1_countries = df_s1_chunk['norm_country'].values if 'norm_country' in df_s1_chunk.columns else np.array([''] * len(df_s1_chunk))

        out_s1_idx = []
        out_s1_lex = []
        out_cand_idx = []
        out_cand_lex = []
        out_src = []
        out_b_ctry = []
        out_b_name = []
        out_b_p3 = []
        out_b_p4 = []
        out_b_addr = []
        out_num_keys = []
        out_score = []
        out_exact = []

        raw_counts_chunk = {}
        sources_target = self.target_sources
        names_target = self.target_names
        target_lex = self.target_lex_rank

        for s1_idx, s1_lex, name, addr, country in zip(s1_orig_indices, s1_lex_ranks, s1_names, s1_addrs, s1_countries):
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

            n_raw = len(cand_flags)
            raw_counts_chunk[s1_idx] = n_raw
            if n_raw == 0:
                continue

            if n_raw <= max_candidates_per_s1:
                # All fit: sort deterministically by (-score, target_lex_rank)
                top_idxs = list(cand_flags.keys())
                top_idxs.sort(key=lambda idx: (-cand_scores[idx], target_lex[idx]))
            else:
                # Bounded integer score-bucketing top-K selection
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

            # Materialize ONLY the retained top candidates using compact integer IDs
            for idx in top_idxs:
                mask = cand_flags[idx]
                b_name = 1 if (mask & 1) else 0
                b_p4 = 1 if (mask & 2) else 0
                b_p3 = 1 if (mask & 4) else 0
                b_addr = 1 if (mask & 8) else 0
                num_keys = b_name + b_p4 + b_p3 + b_addr

                c_name = names_target[idx]
                exact_match = 1 if (name and name == c_name) else 0

                out_s1_idx.append(s1_idx)
                out_s1_lex.append(s1_lex)
                out_cand_idx.append(idx)
                out_cand_lex.append(target_lex[idx])
                out_src.append(sources_target[idx])
                out_b_ctry.append(1)
                out_b_name.append(b_name)
                out_b_p3.append(b_p3)
                out_b_p4.append(b_p4)
                out_b_addr.append(b_addr)
                out_num_keys.append(num_keys)
                out_score.append(cand_scores[idx])
                out_exact.append(exact_match)

            del cand_flags, cand_scores

        N_chunk = len(out_s1_idx)
        if N_chunk == 0:
            df_chunk = pd.DataFrame()
        else:
            df_chunk = pd.DataFrame({
                's1_orig_idx': np.array(out_s1_idx, dtype=np.int32),
                's1_lex_rank': np.array(out_s1_lex, dtype=np.int32),
                'cand_orig_idx': np.array(out_cand_idx, dtype=np.int32),
                'cand_lex_rank': np.array(out_cand_lex, dtype=np.int32),
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
                'blocked_rare_token': np.zeros(N_chunk, dtype=np.int8),
                'blocked_token_pair': np.zeros(N_chunk, dtype=np.int8),
                'blocked_char_ngram': np.zeros(N_chunk, dtype=np.int8),
                'shared_ngram_count': np.zeros(N_chunk, dtype=np.int8)
            })

        return df_chunk, raw_counts_chunk

    def generate_bounded_candidates_opt_c(self, df_s1, max_candidates_per_s1=400, chunk_size=25, return_metrics=False):
        c_size = chunk_size
        post_cap_chunks = []
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

            capped_chunk, counts_in_chunk = self.generate_bounded_chunk_opt_c(
                chunk_s1, chunk_indices, chunk_lex, max_candidates_per_s1=max_candidates_per_s1
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

            # Map integer IDs back to original string IDs at the boundary
            capped_df['entity_id_s1'] = s1_ids_all[capped_df['s1_orig_idx'].values]
            capped_df['entity_id_cand'] = self.target_eids[capped_df['cand_orig_idx'].values]

            # Reorder columns to exact schema and drop temporary integer rank columns
            cols = [
                'entity_id_s1', 'entity_id_cand', 'source', 'blocked_country',
                'blocked_name_token', 'blocked_prefix_3', 'blocked_prefix_4',
                'blocked_address_token', 'num_blocking_keys', 'evidence_score',
                'blocked_exact_name', 'blocked_selective_token', 'blocked_rare_token',
                'blocked_token_pair', 'blocked_char_ngram', 'shared_ngram_count'
            ]
            capped_df = capped_df[cols]
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
        return capped_df, metrics

# Instantiate Opt C engine
t0_init = time.time()
engine_c = OptCEngine(targets)
print(f"Engine C initialized in {time.time()-t0_init:.2f}s")

t0 = time.time()
res_c, metrics_c = engine_c.generate_bounded_candidates_opt_c(s1_fr, max_candidates_per_s1=400, chunk_size=25, return_metrics=True)
t_c = time.time() - t0
print(f"Opt C: {t_c:.3f}s | Candidates: {len(res_c):,}")

# Assert 100% bit-for-bit equivalence
assert len(res_b) == len(res_c), f"Length mismatch: {len(res_b)} vs {len(res_c)}"
assert list(res_b.columns) == list(res_c.columns), f"Columns mismatch"
for col in res_b.columns:
    if col == 'evidence_score':
        diff = np.abs(res_b[col].values - res_c[col].values).max()
        assert diff == 0, f"Diff in {col}: {diff}"
    else:
        mismatches = (res_b[col].values != res_c[col].values).sum()
        assert mismatches == 0, f"Mismatches in column {col}: {mismatches}"

print("=" * 60)
print("ALL 16 COLUMNS MATCH 100% BIT-FOR-BIT IN EXACT DETERMINISTIC ORDER!")
print(f"Speedup: {t_b / t_c:.2f}x faster!")
print("=" * 60)
