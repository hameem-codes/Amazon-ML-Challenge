"""
Phase 4A: Feature Engineering Module

Computes pairwise features between Source 1 records and candidate (Source 2 / Source 3) records.
Handles:
- Group A: Name Similarity
- Group B: Address Similarity
- Group C: Country
- Group D: Directional Missingness
- Group E: Structural Features
- Group F: Blocking Evidence
- Group G: Source Information
- TF-IDF Cosine Similarity (Word-level)

All numerical outputs are strictly finite (no NaN, inf, -inf).
"""

import math
import numpy as np
import pandas as pd
import rapidfuzz.distance.Levenshtein as lev
import rapidfuzz.distance.JaroWinkler as jw
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import paired_cosine_distances

FEATURE_NAMES = [
    # Group A: Name Similarity
    'name_exact',
    's1_name_length',
    'candidate_name_length',
    'name_length_ratio',
    'name_levenshtein_similarity',
    'name_jaro_winkler',
    'name_char_similarity',
    'name_token_jaccard',
    'name_token_overlap',
    's1_name_token_count',
    'candidate_name_token_count',
    'name_token_count_diff',
    'name_s1_contains_candidate',
    'name_candidate_contains_s1',
    # Group B: Address Similarity
    'address_exact',
    'address_levenshtein_similarity',
    'address_jaro_winkler',
    'address_char_similarity',
    'address_token_jaccard',
    'address_token_overlap',
    'address_length_ratio',
    's1_address_token_count',
    'candidate_address_token_count',
    'address_token_count_diff',
    # Group C: Country
    'country_exact',
    'country_mismatch',
    'country_missing_both',
    'country_missing_s1',
    'country_missing_candidate',
    # Group D: Directional Missingness
    's1_name_missing',
    'candidate_name_missing',
    'both_name_missing',
    'both_name_present',
    's1_address_missing',
    'candidate_address_missing',
    'both_address_missing',
    'both_address_present',
    # Group E: Structural Features
    'name_length_diff',
    'address_length_diff',
    'name_digit_count_s1',
    'name_digit_count_candidate',
    'name_has_digits_s1',
    'name_has_digits_candidate',
    'address_digit_count_s1',
    'address_digit_count_candidate',
    'address_has_digits_s1',
    'address_has_digits_candidate',
    # Group F: Blocking Evidence
    'blocked_exact_name',
    'blocked_selective_token',
    'blocked_rare_token',
    'blocked_token_pair',
    'blocked_char_ngram',
    'num_blocking_keys',
    # Group G: Source Information
    'candidate_is_source2',
    'candidate_is_source3',
    # TF-IDF Cosine
    'name_tfidf_cosine',
    'address_tfidf_cosine'
]

def _safe_str(val):
    if val is None or pd.isna(val):
        return ""
    s = str(val).strip()
    return "" if s.lower() == "nan" else s

def _get_char_similarity(s1, s2):
    """
    Normalized character set intersection over union.
    Returns 0.0 if either string is empty.
    """
    if not s1 or not s2:
        return 0.0
    c1 = set(s1)
    c2 = set(s2)
    union = c1 | c2
    if not union:
        return 0.0
    return len(c1 & c2) / len(union)

def _get_token_jaccard_and_overlap(toks1, toks2):
    """
    Computes Jaccard similarity and symmetric token overlap (overlap coefficient).
    Returns (jaccard, overlap).
    """
    if not toks1 or not toks2:
        return 0.0, 0.0
    s1 = set(toks1)
    s2 = set(toks2)
    intersection = len(s1 & s2)
    union = len(s1 | s2)
    min_size = min(len(s1), len(s2))
    
    jaccard = intersection / union if union > 0 else 0.0
    overlap = intersection / min_size if min_size > 0 else 0.0
    return jaccard, overlap

def _count_digits(s):
    return sum(1 for c in s if c.isdigit())

def build_single_pair_features(s1_name, s1_addr, s1_country,
                                cand_name, cand_addr, cand_country,
                                cand_source, blocking_info=None):
    """
    Extracts features for a single entity pair.
    Used for unit testing, individual row inspection, and sanity tests.
    """
    s1_name = _safe_str(s1_name)
    cand_name = _safe_str(cand_name)
    s1_addr = _safe_str(s1_addr)
    cand_addr = _safe_str(cand_addr)
    s1_c = _safe_str(s1_country)
    cand_c = _safe_str(cand_country)
    
    # Missingness
    s1_nm_miss = 1 if not s1_name else 0
    c_nm_miss = 1 if not cand_name else 0
    both_nm_miss = 1 if (s1_nm_miss and c_nm_miss) else 0
    both_nm_pres = 1 if (not s1_nm_miss and not c_nm_miss) else 0
    
    s1_ad_miss = 1 if not s1_addr else 0
    c_ad_miss = 1 if not cand_addr else 0
    both_ad_miss = 1 if (s1_ad_miss and c_ad_miss) else 0
    both_ad_pres = 1 if (not s1_ad_miss and not c_ad_miss) else 0
    
    # Group A: Name Similarity
    if both_nm_pres:
        nm_exact = 1 if s1_name == cand_name else 0
        nm_len1 = len(s1_name)
        nm_len2 = len(cand_name)
        nm_max_len = max(nm_len1, nm_len2)
        nm_min_len = min(nm_len1, nm_len2)
        nm_len_ratio = nm_min_len / nm_max_len if nm_max_len > 0 else 1.0
        
        nm_lev = float(lev.normalized_similarity(s1_name, cand_name))
        nm_jw = float(jw.similarity(s1_name, cand_name))
        nm_char_sim = _get_char_similarity(s1_name, cand_name)
        
        t1 = s1_name.split()
        t2 = cand_name.split()
        nm_tok_jacc, nm_tok_over = _get_token_jaccard_and_overlap(t1, t2)
        
        s1_nm_tok_cnt = len(t1)
        c_nm_tok_cnt = len(t2)
        nm_tok_cnt_diff = abs(s1_nm_tok_cnt - c_nm_tok_cnt)
        
        nm_s1_contains_c = 1 if cand_name in s1_name else 0
        nm_c_contains_s1 = 1 if s1_name in cand_name else 0
    else:
        nm_exact = 0
        nm_len1 = len(s1_name)
        nm_len2 = len(cand_name)
        nm_len_ratio = 0.0
        nm_lev = 0.0
        nm_jw = 0.0
        nm_char_sim = 0.0
        nm_tok_jacc = 0.0
        nm_tok_over = 0.0
        t1 = s1_name.split() if s1_name else []
        t2 = cand_name.split() if cand_name else []
        s1_nm_tok_cnt = len(t1)
        c_nm_tok_cnt = len(t2)
        nm_tok_cnt_diff = abs(s1_nm_tok_cnt - c_nm_tok_cnt)
        nm_s1_contains_c = 0
        nm_c_contains_s1 = 0
        
    # Group B: Address Similarity
    if both_ad_pres:
        ad_exact = 1 if s1_addr == cand_addr else 0
        ad_len1 = len(s1_addr)
        ad_len2 = len(cand_addr)
        ad_max_len = max(ad_len1, ad_len2)
        ad_min_len = min(ad_len1, ad_len2)
        ad_len_ratio = ad_min_len / ad_max_len if ad_max_len > 0 else 1.0
        
        ad_lev = float(lev.normalized_similarity(s1_addr, cand_addr))
        ad_jw = float(jw.similarity(s1_addr, cand_addr))
        ad_char_sim = _get_char_similarity(s1_addr, cand_addr)
        
        at1 = s1_addr.split()
        at2 = cand_addr.split()
        ad_tok_jacc, ad_tok_over = _get_token_jaccard_and_overlap(at1, at2)
        
        s1_ad_tok_cnt = len(at1)
        c_ad_tok_cnt = len(at2)
        ad_tok_cnt_diff = abs(s1_ad_tok_cnt - c_ad_tok_cnt)
    else:
        ad_exact = 0
        ad_len1 = len(s1_addr)
        ad_len2 = len(cand_addr)
        ad_len_ratio = 0.0
        ad_lev = 0.0
        ad_jw = 0.0
        ad_char_sim = 0.0
        ad_tok_jacc = 0.0
        ad_tok_over = 0.0
        at1 = s1_addr.split() if s1_addr else []
        at2 = cand_addr.split() if cand_addr else []
        s1_ad_tok_cnt = len(at1)
        c_ad_tok_cnt = len(at2)
        ad_tok_cnt_diff = abs(s1_ad_tok_cnt - c_ad_tok_cnt)

    # Group C: Country
    c_s1_miss = 1 if not s1_c else 0
    c_cand_miss = 1 if not cand_c else 0
    c_both_miss = 1 if (c_s1_miss and c_cand_miss) else 0
    
    if not c_s1_miss and not c_cand_miss:
        c_exact = 1 if s1_c == cand_c else 0
        c_mismatch = 1 if s1_c != cand_c else 0
    else:
        c_exact = 0
        c_mismatch = 0
        
    # Group E: Structural Features
    nm_len_diff = abs(nm_len1 - nm_len2)
    ad_len_diff = abs(ad_len1 - ad_len2)
    
    nm_d_s1 = _count_digits(s1_name)
    nm_d_c = _count_digits(cand_name)
    ad_d_s1 = _count_digits(s1_addr)
    ad_d_c = _count_digits(cand_addr)
    
    # Group F: Blocking Evidence
    b_info = blocking_info or {}
    b_exact = b_info.get('blocked_exact_name', 0)
    b_sel = b_info.get('blocked_selective_token', 0)
    b_rare = b_info.get('blocked_rare_token', 0)
    b_pair = b_info.get('blocked_token_pair', 0)
    b_ngram = b_info.get('blocked_char_ngram', 0)
    num_b = b_info.get('num_blocking_keys', (b_exact + b_sel + b_rare + b_pair + b_ngram))
    
    # Group G: Source Information
    src_str = str(cand_source).upper()
    is_s2 = 1 if 'S2' in src_str else 0
    is_s3 = 1 if 'S3' in src_str else 0
    
    return {
        'name_exact': nm_exact,
        's1_name_length': nm_len1,
        'candidate_name_length': nm_len2,
        'name_length_ratio': nm_len_ratio,
        'name_levenshtein_similarity': nm_lev,
        'name_jaro_winkler': nm_jw,
        'name_char_similarity': nm_char_sim,
        'name_token_jaccard': nm_tok_jacc,
        'name_token_overlap': nm_tok_over,
        's1_name_token_count': s1_nm_tok_cnt,
        'candidate_name_token_count': c_nm_tok_cnt,
        'name_token_count_diff': nm_tok_cnt_diff,
        'name_s1_contains_candidate': nm_s1_contains_c,
        'name_candidate_contains_s1': nm_c_contains_s1,
        
        'address_exact': ad_exact,
        'address_levenshtein_similarity': ad_lev,
        'address_jaro_winkler': ad_jw,
        'address_char_similarity': ad_char_sim,
        'address_token_jaccard': ad_tok_jacc,
        'address_token_overlap': ad_tok_over,
        'address_length_ratio': ad_len_ratio,
        's1_address_token_count': s1_ad_tok_cnt,
        'candidate_address_token_count': c_ad_tok_cnt,
        'address_token_count_diff': ad_tok_cnt_diff,
        
        'country_exact': c_exact,
        'country_mismatch': c_mismatch,
        'country_missing_both': c_both_miss,
        'country_missing_s1': c_s1_miss,
        'country_missing_candidate': c_cand_miss,
        
        's1_name_missing': s1_nm_miss,
        'candidate_name_missing': c_nm_miss,
        'both_name_missing': both_nm_miss,
        'both_name_present': both_nm_pres,
        's1_address_missing': s1_ad_miss,
        'candidate_address_missing': c_ad_miss,
        'both_address_missing': both_ad_miss,
        'both_address_present': both_ad_pres,
        
        'name_length_diff': nm_len_diff,
        'address_length_diff': ad_len_diff,
        'name_digit_count_s1': nm_d_s1,
        'name_digit_count_candidate': nm_d_c,
        'name_has_digits_s1': 1 if nm_d_s1 > 0 else 0,
        'name_has_digits_candidate': 1 if nm_d_c > 0 else 0,
        'address_digit_count_s1': ad_d_s1,
        'address_digit_count_candidate': ad_d_c,
        'address_has_digits_s1': 1 if ad_d_s1 > 0 else 0,
        'address_has_digits_candidate': 1 if ad_d_c > 0 else 0,
        
        'blocked_exact_name': b_exact,
        'blocked_selective_token': b_sel,
        'blocked_rare_token': b_rare,
        'blocked_token_pair': b_pair,
        'blocked_char_ngram': b_ngram,
        'num_blocking_keys': num_b,
        
        'candidate_is_source2': is_s2,
        'candidate_is_source3': is_s3,
        
        'name_tfidf_cosine': 0.0,
        'address_tfidf_cosine': 0.0
    }

def build_vectorized_features(df_pairs, s1_dict, target_dict, tfidf_models=None):
    """
    Vectorized and highly efficient feature builder for a DataFrame of candidate pairs.
    df_pairs columns: ['entity_id_s1', 'entity_id_cand', 'source'] 
    plus optional blocking flags: ['blocked_exact_name', 'blocked_selective_token', etc.]
    """
    n = len(df_pairs)
    if n == 0:
        return pd.DataFrame(columns=['source1_entity_id', 'candidate_entity_id', 'candidate_source'] + FEATURE_NAMES)
        
    s1_ids = df_pairs['entity_id_s1'].values
    cand_ids = df_pairs['entity_id_cand'].values
    sources = df_pairs['source'].values if 'source' in df_pairs.columns else np.array(['S2' if str(c).startswith('S2') else 'S3' for c in cand_ids])
    
    # Extract string arrays
    s1_names = [s1_dict[i].get('norm_name', '') for i in s1_ids]
    cand_names = [target_dict[i].get('norm_name', '') for i in cand_ids]
    s1_addrs = [s1_dict[i].get('norm_address', '') for i in s1_ids]
    cand_addrs = [target_dict[i].get('norm_address', '') for i in cand_ids]
    s1_countries = [s1_dict[i].get('norm_country', '') for i in s1_ids]
    cand_countries = [target_dict[i].get('norm_country', '') for i in cand_ids]
    
    # Group D: Directional Missingness
    s1_nm_miss = np.array([1 if not s else 0 for s in s1_names], dtype=np.int32)
    c_nm_miss = np.array([1 if not s else 0 for s in cand_names], dtype=np.int32)
    both_nm_miss = np.bitwise_and(s1_nm_miss, c_nm_miss)
    both_nm_pres = (1 - s1_nm_miss) * (1 - c_nm_miss)
    
    s1_ad_miss = np.array([1 if not s else 0 for s in s1_addrs], dtype=np.int32)
    c_ad_miss = np.array([1 if not s else 0 for s in cand_addrs], dtype=np.int32)
    both_ad_miss = np.bitwise_and(s1_ad_miss, c_ad_miss)
    both_ad_pres = (1 - s1_ad_miss) * (1 - c_ad_miss)
    
    # Group A: Name features
    nm_len1 = np.array([len(s) for s in s1_names], dtype=np.int32)
    nm_len2 = np.array([len(s) for s in cand_names], dtype=np.int32)
    nm_len_diff = np.abs(nm_len1 - nm_len2)
    
    # Safe length ratio
    nm_max_len = np.maximum(nm_len1, nm_len2)
    nm_min_len = np.minimum(nm_len1, nm_len2)
    with np.errstate(divide='ignore', invalid='ignore'):
        nm_len_ratio = np.where(nm_max_len > 0, nm_min_len / nm_max_len, 0.0)
    nm_len_ratio = np.where(both_nm_pres == 1, nm_len_ratio, 0.0)
    
    # Exact matches
    name_exact = np.array([1 if (both_nm_pres[i] and s1_names[i] == cand_names[i]) else 0 for i in range(n)], dtype=np.int32)
    
    # Levenshtein & JaroWinkler
    nm_lev = np.zeros(n, dtype=np.float32)
    nm_jw = np.zeros(n, dtype=np.float32)
    nm_char_sim = np.zeros(n, dtype=np.float32)
    nm_tok_jacc = np.zeros(n, dtype=np.float32)
    nm_tok_over = np.zeros(n, dtype=np.float32)
    s1_nm_tok_cnt = np.zeros(n, dtype=np.int32)
    c_nm_tok_cnt = np.zeros(n, dtype=np.int32)
    nm_s1_contains_c = np.zeros(n, dtype=np.int32)
    nm_c_contains_s1 = np.zeros(n, dtype=np.int32)
    
    for i in range(n):
        s1 = s1_names[i]
        s2 = cand_names[i]
        t1 = s1.split() if s1 else []
        t2 = s2.split() if s2 else []
        s1_nm_tok_cnt[i] = len(t1)
        c_nm_tok_cnt[i] = len(t2)
        
        if both_nm_pres[i]:
            nm_lev[i] = lev.normalized_similarity(s1, s2)
            nm_jw[i] = jw.similarity(s1, s2)
            nm_char_sim[i] = _get_char_similarity(s1, s2)
            jacc, over = _get_token_jaccard_and_overlap(t1, t2)
            nm_tok_jacc[i] = jacc
            nm_tok_over[i] = over
            if s2 in s1:
                nm_s1_contains_c[i] = 1
            if s1 in s2:
                nm_c_contains_s1[i] = 1
                
    nm_tok_cnt_diff = np.abs(s1_nm_tok_cnt - c_nm_tok_cnt)
    
    # Group B: Address features
    ad_len1 = np.array([len(s) for s in s1_addrs], dtype=np.int32)
    ad_len2 = np.array([len(s) for s in cand_addrs], dtype=np.int32)
    ad_len_diff = np.abs(ad_len1 - ad_len2)
    
    ad_max_len = np.maximum(ad_len1, ad_len2)
    ad_min_len = np.minimum(ad_len1, ad_len2)
    with np.errstate(divide='ignore', invalid='ignore'):
        ad_len_ratio = np.where(ad_max_len > 0, ad_min_len / ad_max_len, 0.0)
    ad_len_ratio = np.where(both_ad_pres == 1, ad_len_ratio, 0.0)
    
    address_exact = np.array([1 if (both_ad_pres[i] and s1_addrs[i] == cand_addrs[i]) else 0 for i in range(n)], dtype=np.int32)
    
    ad_lev = np.zeros(n, dtype=np.float32)
    ad_jw = np.zeros(n, dtype=np.float32)
    ad_char_sim = np.zeros(n, dtype=np.float32)
    ad_tok_jacc = np.zeros(n, dtype=np.float32)
    ad_tok_over = np.zeros(n, dtype=np.float32)
    s1_ad_tok_cnt = np.zeros(n, dtype=np.int32)
    c_ad_tok_cnt = np.zeros(n, dtype=np.int32)
    
    for i in range(n):
        a1 = s1_addrs[i]
        a2 = cand_addrs[i]
        at1 = a1.split() if a1 else []
        at2 = a2.split() if a2 else []
        s1_ad_tok_cnt[i] = len(at1)
        c_ad_tok_cnt[i] = len(at2)
        
        if both_ad_pres[i]:
            ad_lev[i] = lev.normalized_similarity(a1, a2)
            ad_jw[i] = jw.similarity(a1, a2)
            ad_char_sim[i] = _get_char_similarity(a1, a2)
            jacc, over = _get_token_jaccard_and_overlap(at1, at2)
            ad_tok_jacc[i] = jacc
            ad_tok_over[i] = over
            
    ad_tok_cnt_diff = np.abs(s1_ad_tok_cnt - c_ad_tok_cnt)
    
    # Group C: Country
    c_s1_miss = np.array([1 if not s else 0 for s in s1_countries], dtype=np.int32)
    c_cand_miss = np.array([1 if not s else 0 for s in cand_countries], dtype=np.int32)
    c_both_miss = np.bitwise_and(c_s1_miss, c_cand_miss)
    
    c_exact = np.zeros(n, dtype=np.int32)
    c_mismatch = np.zeros(n, dtype=np.int32)
    for i in range(n):
        if not c_s1_miss[i] and not c_cand_miss[i]:
            if s1_countries[i] == cand_countries[i]:
                c_exact[i] = 1
            else:
                c_mismatch[i] = 1
                
    # Group E: Structural / Digit features
    nm_d_s1 = np.array([_count_digits(s) for s in s1_names], dtype=np.int32)
    nm_d_c = np.array([_count_digits(s) for s in cand_names], dtype=np.int32)
    nm_has_d_s1 = np.where(nm_d_s1 > 0, 1, 0).astype(np.int32)
    nm_has_d_c = np.where(nm_d_c > 0, 1, 0).astype(np.int32)
    
    ad_d_s1 = np.array([_count_digits(s) for s in s1_addrs], dtype=np.int32)
    ad_d_c = np.array([_count_digits(s) for s in cand_addrs], dtype=np.int32)
    ad_has_d_s1 = np.where(ad_d_s1 > 0, 1, 0).astype(np.int32)
    ad_has_d_c = np.where(ad_d_c > 0, 1, 0).astype(np.int32)
    
    # Group F: Blocking Evidence
    b_exact = df_pairs['blocked_exact_name'].values if 'blocked_exact_name' in df_pairs.columns else name_exact
    b_sel = df_pairs['blocked_selective_token'].values if 'blocked_selective_token' in df_pairs.columns else np.zeros(n, dtype=np.int32)
    b_rare = df_pairs['blocked_rare_token'].values if 'blocked_rare_token' in df_pairs.columns else np.zeros(n, dtype=np.int32)
    b_pair = df_pairs['blocked_token_pair'].values if 'blocked_token_pair' in df_pairs.columns else np.zeros(n, dtype=np.int32)
    b_ngram = df_pairs['blocked_char_ngram'].values if 'blocked_char_ngram' in df_pairs.columns else np.zeros(n, dtype=np.int32)
    num_b = df_pairs['num_blocking_keys'].values if 'num_blocking_keys' in df_pairs.columns else (b_exact + b_sel + b_rare + b_pair + b_ngram)
    
    # Group G: Source Information
    is_s2 = np.array([1 if 'S2' in str(s).upper() else 0 for s in sources], dtype=np.int32)
    is_s3 = np.array([1 if 'S3' in str(s).upper() else 0 for s in sources], dtype=np.int32)
    
    # TF-IDF Cosine Features
    name_tfidf_cos = np.zeros(n, dtype=np.float32)
    addr_tfidf_cos = np.zeros(n, dtype=np.float32)
    
    if tfidf_models:
        nm_vec = tfidf_models.get('name_tfidf')
        if nm_vec and both_nm_pres.sum() > 0:
            v_s1 = nm_vec.transform(s1_names)
            v_cand = nm_vec.transform(cand_names)
            # paired cosine similarity = 1 - paired cosine distance
            # Note: handle rows with 0 vectors
            cos_dist = paired_cosine_distances(v_s1, v_cand)
            cos_sim = 1.0 - cos_dist
            # zero out missing pairs or non-finite values
            cos_sim = np.nan_to_num(cos_sim, nan=0.0, posinf=0.0, neginf=0.0)
            name_tfidf_cos = np.where(both_nm_pres == 1, np.clip(cos_sim, 0.0, 1.0), 0.0).astype(np.float32)
            
        ad_vec = tfidf_models.get('address_tfidf')
        if ad_vec and both_ad_pres.sum() > 0:
            v_s1_ad = ad_vec.transform(s1_addrs)
            v_cand_ad = ad_vec.transform(cand_addrs)
            cos_dist_ad = paired_cosine_distances(v_s1_ad, v_cand_ad)
            cos_sim_ad = 1.0 - cos_dist_ad
            cos_sim_ad = np.nan_to_num(cos_sim_ad, nan=0.0, posinf=0.0, neginf=0.0)
            addr_tfidf_cos = np.where(both_ad_pres == 1, np.clip(cos_sim_ad, 0.0, 1.0), 0.0).astype(np.float32)
            
    feat_dict = {
        'source1_entity_id': s1_ids,
        'candidate_entity_id': cand_ids,
        'candidate_source': sources,
        
        # Group A
        'name_exact': name_exact,
        's1_name_length': nm_len1,
        'candidate_name_length': nm_len2,
        'name_length_ratio': nm_len_ratio,
        'name_levenshtein_similarity': nm_lev,
        'name_jaro_winkler': nm_jw,
        'name_char_similarity': nm_char_sim,
        'name_token_jaccard': nm_tok_jacc,
        'name_token_overlap': nm_tok_over,
        's1_name_token_count': s1_nm_tok_cnt,
        'candidate_name_token_count': c_nm_tok_cnt,
        'name_token_count_diff': nm_tok_cnt_diff,
        'name_s1_contains_candidate': nm_s1_contains_c,
        'name_candidate_contains_s1': nm_c_contains_s1,
        
        # Group B
        'address_exact': address_exact,
        'address_levenshtein_similarity': ad_lev,
        'address_jaro_winkler': ad_jw,
        'address_char_similarity': ad_char_sim,
        'address_token_jaccard': ad_tok_jacc,
        'address_token_overlap': ad_tok_over,
        'address_length_ratio': ad_len_ratio,
        's1_address_token_count': s1_ad_tok_cnt,
        'candidate_address_token_count': c_ad_tok_cnt,
        'address_token_count_diff': ad_tok_cnt_diff,
        
        # Group C
        'country_exact': c_exact,
        'country_mismatch': c_mismatch,
        'country_missing_both': c_both_miss,
        'country_missing_s1': c_s1_miss,
        'country_missing_candidate': c_cand_miss,
        
        # Group D
        's1_name_missing': s1_nm_miss,
        'candidate_name_missing': c_nm_miss,
        'both_name_missing': both_nm_miss,
        'both_name_present': both_nm_pres,
        's1_address_missing': s1_ad_miss,
        'candidate_address_missing': c_ad_miss,
        'both_address_missing': both_ad_miss,
        'both_address_present': both_ad_pres,
        
        # Group E
        'name_length_diff': nm_len_diff,
        'address_length_diff': ad_len_diff,
        'name_digit_count_s1': nm_d_s1,
        'name_digit_count_candidate': nm_d_c,
        'name_has_digits_s1': nm_has_d_s1,
        'name_has_digits_candidate': nm_has_d_c,
        'address_digit_count_s1': ad_d_s1,
        'address_digit_count_candidate': ad_d_c,
        'address_has_digits_s1': ad_has_d_s1,
        'address_has_digits_candidate': ad_has_d_c,
        
        # Group F
        'blocked_exact_name': b_exact,
        'blocked_selective_token': b_sel,
        'blocked_rare_token': b_rare,
        'blocked_token_pair': b_pair,
        'blocked_char_ngram': b_ngram,
        'num_blocking_keys': num_b,
        
        # Group G
        'candidate_is_source2': is_s2,
        'candidate_is_source3': is_s3,
        
        # TF-IDF
        'name_tfidf_cosine': name_tfidf_cos,
        'address_tfidf_cosine': addr_tfidf_cos
    }
    
    df_out = pd.DataFrame(feat_dict)
    
    # Assert numerical safety: replace any accidental NaN/inf with 0.0
    for col in FEATURE_NAMES:
        if df_out[col].isna().any() or np.isinf(df_out[col]).any():
            df_out[col] = np.nan_to_num(df_out[col].values, nan=0.0, posinf=0.0, neginf=0.0)
            
    return df_out
