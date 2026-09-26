"""
Candidate Safety Cap & Evidence-Ranked Overflow Filtering

Implements deterministic ranking and capping of candidates per Source 1 entity:
1. When candidates for an S1 entity exceed max_candidates_per_s1, candidates are ranked
   using blocking evidence (exact match, key count, shared ngrams, token pairs).
2. The top-K highest-evidence candidates are deterministically preserved.
3. Tracks before/after statistics, overflow counts, and recall retention.
"""

import pandas as pd
import numpy as np
from config import CANDIDATE_CAP_CONFIG

def compute_evidence_score(df_candidates, weights=None):
    """
    Computes a deterministic evidence score for each candidate pair based on blocking signals.
    """
    if 'evidence_score' in df_candidates.columns:
        return df_candidates['evidence_score'].values.astype(np.float32)
        
    w = weights or CANDIDATE_CAP_CONFIG['ranking_weights']
    score = np.zeros(len(df_candidates), dtype=np.float32)
    
    # Config A Channels
    if 'blocked_name_token' in df_candidates.columns and 'name_token_weight' in w:
        score += df_candidates['blocked_name_token'].values * w['name_token_weight']
    if 'blocked_prefix_4' in df_candidates.columns and 'prefix_4_weight' in w:
        score += df_candidates['blocked_prefix_4'].values * w['prefix_4_weight']
    if 'blocked_prefix_3' in df_candidates.columns and 'prefix_3_weight' in w:
        score += df_candidates['blocked_prefix_3'].values * w['prefix_3_weight']
    if 'blocked_address_token' in df_candidates.columns and 'address_token_weight' in w:
        score += df_candidates['blocked_address_token'].values * w['address_token_weight']
    if 'num_blocking_keys' in df_candidates.columns and 'num_keys_weight' in w:
        score += df_candidates['num_blocking_keys'].values * w['num_keys_weight']
    if 'blocked_exact_name' in df_candidates.columns and 'exact_name_weight' in w:
        score += df_candidates['blocked_exact_name'].values * w['exact_name_weight']
        
    # Backward compatibility
    if 'blocked_selective_token' in df_candidates.columns and 'selective_token_weight' in w and 'blocked_name_token' not in df_candidates.columns:
        score += df_candidates['blocked_selective_token'].values * w['selective_token_weight']
    if 'shared_ngram_count' in df_candidates.columns and 'shared_ngrams_weight' in w:
        score += df_candidates['shared_ngram_count'].values * w['shared_ngrams_weight']
    if 'blocked_token_pair' in df_candidates.columns and 'token_pair_weight' in w:
        score += df_candidates['blocked_token_pair'].values * w['token_pair_weight']
    if 'blocked_rare_token' in df_candidates.columns and 'rare_token_weight' in w:
        score += df_candidates['blocked_rare_token'].values * w['rare_token_weight']
        
    return score

def apply_candidate_safety_cap(df_candidates, max_candidates_per_s1=None, weights=None, true_pairs_set=None):
    """
    Applies the candidate safety cap per Source 1 entity.
    
    Returns:
      filtered_df: DataFrame of capped candidates
      cap_metrics: Dict containing before/after statistics and recall impact
    """
    max_cands = max_candidates_per_s1 or CANDIDATE_CAP_CONFIG['max_candidates_per_s1']
    
    if len(df_candidates) == 0:
        return df_candidates.copy(), {
            'before_cap_candidates': 0,
            'after_cap_candidates': 0,
            'overflowing_s1_count': 0,
            'max_candidates_per_s1': max_cands,
            'true_pairs_before_cap': 0,
            'true_pairs_after_cap': 0,
            'cap_recall_retention': 1.0
        }
        
    df = df_candidates.copy()
    if 'evidence_score' not in df.columns:
        df['evidence_score'] = compute_evidence_score(df, weights=weights)
        
    # Measure true pairs before cap if ground truth provided
    true_pairs_before = 0
    if true_pairs_set is not None:
        pairs_before = set(zip(df['entity_id_s1'], df['entity_id_cand']))
        true_pairs_before = len(pairs_before.intersection(true_pairs_set))
        
    counts_per_s1 = df.groupby('entity_id_s1').size()
    overflowing_s1 = set(counts_per_s1[counts_per_s1 > max_cands].index)
    
    # Sort deterministically by: entity_id_s1 (asc), evidence_score (desc), entity_id_cand (asc)
    df_sorted = df.sort_values(
        by=['entity_id_s1', 'evidence_score', 'entity_id_cand'],
        ascending=[True, False, True]
    )
    
    # Take top-K per entity_id_s1
    filtered_df = df_sorted.groupby('entity_id_s1').head(max_cands).reset_index(drop=True)
    
    # Measure true pairs after cap
    true_pairs_after = 0
    if true_pairs_set is not None:
        pairs_after = set(zip(filtered_df['entity_id_s1'], filtered_df['entity_id_cand']))
        true_pairs_after = len(pairs_after.intersection(true_pairs_set))
        
    cap_recall_retention = (true_pairs_after / true_pairs_before) if true_pairs_before > 0 else 1.0
    
    cap_metrics = {
        'before_cap_candidates': len(df),
        'after_cap_candidates': len(filtered_df),
        'overflowing_s1_count': len(overflowing_s1),
        'max_candidates_per_s1': max_cands,
        'true_pairs_before_cap': true_pairs_before,
        'true_pairs_after_cap': true_pairs_after,
        'cap_recall_retention': cap_recall_retention
    }
    
    return filtered_df, cap_metrics
