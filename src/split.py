"""
Entity-Level Train / Validation Split Module

Enforces strict Source 1 entity-level splitting to prevent reference entity leakage.
No Source 1 entity ID will ever appear in both training and validation splits.
"""

import numpy as np
import pandas as pd
from config import SPLIT_CONFIG

def split_by_entity_id(s1_entity_ids, train_ratio=None, random_seed=None):
    """
    Deterministically splits an iterable of Source 1 entity IDs into train and validation sets.
    """
    ratio = train_ratio or SPLIT_CONFIG['train_ratio']
    seed = random_seed or SPLIT_CONFIG['random_seed']
    
    unique_ids = sorted(list(set(s1_entity_ids)))
    rng = np.random.RandomState(seed)
    shuffled = rng.permutation(unique_ids)
    
    split_idx = int(len(shuffled) * ratio)
    train_ids = set(shuffled[:split_idx])
    val_ids = set(shuffled[split_idx:])
    
    return train_ids, val_ids

def split_candidate_pairs(df_candidates, train_ratio=None, random_seed=None):
    """
    Splits a candidate pairs DataFrame strictly by Source 1 entity_id_s1.
    Returns: (train_candidates_df, val_candidates_df)
    """
    train_s1_ids, val_s1_ids = split_by_entity_id(
        df_candidates['entity_id_s1'].unique(),
        train_ratio=train_ratio,
        random_seed=random_seed
    )
    
    train_df = df_candidates[df_candidates['entity_id_s1'].isin(train_s1_ids)].copy().reset_index(drop=True)
    val_df = df_candidates[df_candidates['entity_id_s1'].isin(val_s1_ids)].copy().reset_index(drop=True)
    
    return train_df, val_df, train_s1_ids, val_s1_ids
