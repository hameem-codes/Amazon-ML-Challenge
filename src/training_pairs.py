"""
Training Pair Construction & Hard-Negative Sampling Module

Constructs training candidate pairs:
1. Retains all ground-truth positive pairs that survived blocking and candidate capping.
2. Selects hard negatives (high evidence, multiple blocking keys, shared n-grams, exact name collisions).
3. Applies deterministic sampling with fixed random seeds.
4. Reports dynamic scale_pos_weight for downstream XGBoost modeling.
"""

import numpy as np
import pandas as pd
from config import TRAINING_PAIRS_CONFIG

def construct_training_pairs(df_candidates, true_pairs_set,
                             hard_neg_ratio=None,
                             random_seed=None):
    """
    Constructs a training pair dataset by:
    - Labeling every surviving candidate pair (1 = positive match, 0 = negative).
    - Preserving 100% of true positive pairs.
    - Selecting hard negatives ranked by blocking evidence score, plus controlled random negatives.
    
    Returns:
      df_training_pairs: DataFrame with 'label' column and evidence attributes.
      stats: Dict with positive count, negative count, hard negative count, and dynamic scale_pos_weight.
    """
    ratio = hard_neg_ratio or TRAINING_PAIRS_CONFIG['hard_negative_ratio']
    seed = random_seed or TRAINING_PAIRS_CONFIG['random_seed']
    
    df = df_candidates.copy()
    
    # 1. Label pairs
    s1_vals = df['entity_id_s1'].values
    c_vals = df['entity_id_cand'].values
    is_pos = np.array([(s1_vals[i], c_vals[i]) in true_pairs_set for i in range(len(df))], dtype=bool)
    df['label'] = np.where(is_pos, 1, 0)
    
    df_pos = df[df['label'] == 1].copy()
    df_neg = df[df['label'] == 0].copy()
    
    num_pos = len(df_pos)
    num_neg_total = len(df_neg)
    
    # Target negative count
    target_neg_count = min(num_neg_total, int(num_pos * ratio))
    
    # 2. Select hard negatives
    # Score negatives deterministically by evidence: composite evidence score (desc), entity_id_s1 (asc), entity_id_cand (asc)
    from candidate_safety import compute_evidence_score
    df_neg['evidence_score'] = compute_evidence_score(df_neg)
    df_neg_sorted = df_neg.sort_values(
        by=['evidence_score', 'entity_id_s1', 'entity_id_cand'],
        ascending=[False, True, True]
    )
    
    # Take top 70% of target budget from highest-evidence hard negatives, and 30% random negatives for diversity
    hard_quota = int(target_neg_count * 0.70)
    random_quota = target_neg_count - hard_quota
    
    df_hard_neg = df_neg_sorted.head(hard_quota)
    remaining_neg = df_neg_sorted.iloc[hard_quota:]
    
    if len(remaining_neg) > 0 and random_quota > 0:
        rng = np.random.RandomState(seed)
        sample_indices = rng.choice(len(remaining_neg), size=min(random_quota, len(remaining_neg)), replace=False)
        df_rand_neg = remaining_neg.iloc[sample_indices]
    else:
        df_rand_neg = pd.DataFrame(columns=df.columns)
        
    df_neg_selected = pd.concat([df_hard_neg, df_rand_neg], ignore_index=True)
    
    # 3. Combine positives and selected negatives
    df_train_pairs = pd.concat([df_pos, df_neg_selected], ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    
    selected_neg_count = len(df_neg_selected)
    scale_pos_weight = (selected_neg_count / num_pos) if num_pos > 0 else 1.0
    
    stats = {
        'total_candidates': len(df),
        'positive_count': num_pos,
        'negative_pool_count': num_neg_total,
        'selected_negative_count': selected_neg_count,
        'hard_negative_count': len(df_hard_neg),
        'random_negative_count': len(df_rand_neg),
        'scale_pos_weight': scale_pos_weight,
        'sampling_seed': seed,
        'target_ratio': ratio
    }
    
    return df_train_pairs, stats
