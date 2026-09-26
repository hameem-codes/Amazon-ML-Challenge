import sys, os, time, gc, psutil
from collections import defaultdict
sys.path.insert(0, 'src')
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from normalize import normalize_business_name, normalize_country, normalize_business_address
from blocking import ConfigABlockingEngine

proc = psutil.Process(os.getpid())

print("Loading targets...")
t0 = time.time()
targets = pd.read_pickle('scratch/targets_france.pkl')
engine = ConfigABlockingEngine(targets)
target_names = engine.target_names
target_addrs = engine.target_addrs
target_eids = engine.target_eids
target_lex = engine.target_lex_rank
n_targets = len(targets)
print(f"Loaded {n_targets:,} targets in {time.time()-t0:.2f}s | RAM: {proc.memory_info().rss/1024**2:.1f} MB")

# Benchmark S1
s1_df = pd.read_csv('data/test/test_source1.tsv', sep='\t', dtype=str, nrows=10000, na_filter=False)
s1_fr = s1_df[s1_df['country'].apply(normalize_country) == 'france'].head(500).copy().reset_index(drop=True)
s1_fr['norm_name'] = s1_fr['business_name'].apply(normalize_business_name)
s1_fr['norm_address'] = s1_fr['business_address'].apply(normalize_business_address)
s1_fr['norm_country'] = 'france'
val_s1_ids = list(s1_fr['entity_id'].values)

s1_name_map = defaultdict(list)
for idx, r in s1_fr.iterrows():
    s1_name_map[r['norm_name']].append(r['entity_id'])

exact_matches_set = set()
for idx in range(len(target_names)):
    t_name = target_names[idx]
    if t_name in s1_name_map:
        t_eid = target_eids[idx]
        for s1_id in s1_name_map[t_name]:
            exact_matches_set.add((s1_id, t_eid))

TOTAL_GT_PAIRS = len(exact_matches_set)
assert TOTAL_GT_PAIRS == 4191, f"Expected 4191, got {TOTAL_GT_PAIRS}"
print(f"GT True Pairs: {TOTAL_GT_PAIRS}")

# -------------------------------------------------------------
# REPRESENTATION A: Business Name (Character 3-gram TF-IDF)
# -------------------------------------------------------------
print("\n" + "="*60)
print("REPRESENTATION A: Business Name (Char 3-gram TF-IDF)")
print("="*60)

t_emb0 = time.time()
vec_name = TfidfVectorizer(analyzer="char", ngram_range=(3, 3), min_df=5, max_features=30000, sublinear_tf=True)
target_names_clean = pd.Series(target_names).fillna('')
X_target_name = vec_name.fit_transform(target_names_clean)
time_vec_build_a = time.time() - t_emb0
index_size_mb_a = (X_target_name.data.nbytes + X_target_name.indices.nbytes + X_target_name.indptr.nbytes) / 1024**2

print(f"Index built in {time_vec_build_a:.2f}s | Shape: {X_target_name.shape} | NNZ: {X_target_name.nnz:,} | Matrix Size: {index_size_mb_a:.1f} MB")
print(f"Peak RAM: {proc.memory_info().rss/1024**2:.1f} MB")

t_q0 = time.time()
s1_vecs_a = vec_name.transform(s1_fr['norm_name'])
# Batch query: compute dot products in chunks of 50 to maintain low peak memory
CHUNK_SIZE = 50
n_s1 = len(s1_fr)

topk_pairs_a = {50: set(), 100: set(), 200: set(), 400: set(), 800: set()}

for start in range(0, n_s1, CHUNK_SIZE):
    end = min(start + CHUNK_SIZE, n_s1)
    chunk_vecs = s1_vecs_a[start:end]
    # Dot product: shape (n_targets, chunk_size)
    scores_chunk = X_target_name.dot(chunk_vecs.T).toarray()
    
    for i, s1_idx in enumerate(range(start, end)):
        s1_id = val_s1_ids[s1_idx]
        sc = scores_chunk[:, i]
        # Partition top 800
        # Deterministic sorting: score DESC, target_lex ASC
        top800_cand_idx = np.argpartition(sc, -800)[-800:]
        top800_scores = sc[top800_cand_idx]
        top800_lex = target_lex[top800_cand_idx]
        
        # Sort top 800
        sort_keys = (top800_lex, -top800_scores)
        sorted_order = np.lexsort(sort_keys)
        sorted_cand_idx = top800_cand_idx[sorted_order]
        
        for K in [50, 100, 200, 400, 800]:
            k_cands = sorted_cand_idx[:K]
            for cidx in k_cands:
                topk_pairs_a[K].add((s1_id, target_eids[cidx]))

time_query_a = time.time() - t_q0
total_time_a = time_vec_build_a + time_query_a
print(f"500 Queries completed in {time_query_a:.2f}s | Total retrieval time: {total_time_a:.2f}s")

for K in [50, 100, 200, 400, 800]:
    tp = len(topk_pairs_a[K] & exact_matches_set)
    rec = tp / TOTAL_GT_PAIRS * 100
    print(f"  Vector-A Top-{K:3d}: TP={tp:4d}/{TOTAL_GT_PAIRS} ({rec:7.4f}%), Total Pairs={len(topk_pairs_a[K]):,}")

# -------------------------------------------------------------
# REPRESENTATION B: Business Name + Address
# -------------------------------------------------------------
print("\n" + "="*60)
print("REPRESENTATION B: Business Name + Address (Char 3-gram TF-IDF)")
print("="*60)

t_emb0 = time.time()
target_combined = (pd.Series(target_names).fillna('') + " " + pd.Series(target_addrs).fillna('')).values
vec_comb = TfidfVectorizer(analyzer="char", ngram_range=(3, 3), min_df=5, max_features=30000, sublinear_tf=True)
X_target_comb = vec_comb.fit_transform(target_combined)
time_vec_build_b = time.time() - t_emb0
index_size_mb_b = (X_target_comb.data.nbytes + X_target_comb.indices.nbytes + X_target_comb.indptr.nbytes) / 1024**2

print(f"Index built in {time_vec_build_b:.2f}s | Shape: {X_target_comb.shape} | NNZ: {X_target_comb.nnz:,} | Matrix Size: {index_size_mb_b:.1f} MB")
print(f"Peak RAM: {proc.memory_info().rss/1024**2:.1f} MB")

t_q0 = time.time()
s1_combined = (s1_fr['norm_name'].fillna('') + " " + s1_fr['norm_address'].fillna('')).values
s1_vecs_b = vec_comb.transform(s1_combined)

topk_pairs_b = {50: set(), 100: set(), 200: set(), 400: set(), 800: set()}

for start in range(0, n_s1, CHUNK_SIZE):
    end = min(start + CHUNK_SIZE, n_s1)
    chunk_vecs = s1_vecs_b[start:end]
    scores_chunk = X_target_comb.dot(chunk_vecs.T).toarray()
    
    for i, s1_idx in enumerate(range(start, end)):
        s1_id = val_s1_ids[s1_idx]
        sc = scores_chunk[:, i]
        top800_cand_idx = np.argpartition(sc, -800)[-800:]
        top800_scores = sc[top800_cand_idx]
        top800_lex = target_lex[top800_cand_idx]
        
        sort_keys = (top800_lex, -top800_scores)
        sorted_order = np.lexsort(sort_keys)
        sorted_cand_idx = top800_cand_idx[sorted_order]
        
        for K in [50, 100, 200, 400, 800]:
            k_cands = sorted_cand_idx[:K]
            for cidx in k_cands:
                topk_pairs_b[K].add((s1_id, target_eids[cidx]))

time_query_b = time.time() - t_q0
total_time_b = time_vec_build_b + time_query_b
print(f"500 Queries completed in {time_query_b:.2f}s | Total retrieval time: {total_time_b:.2f}s")

for K in [50, 100, 200, 400, 800]:
    tp = len(topk_pairs_b[K] & exact_matches_set)
    rec = tp / TOTAL_GT_PAIRS * 100
    print(f"  Vector-B Top-{K:3d}: TP={tp:4d}/{TOTAL_GT_PAIRS} ({rec:7.4f}%), Total Pairs={len(topk_pairs_b[K]):,}")
