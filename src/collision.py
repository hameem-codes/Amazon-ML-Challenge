"""
Collision & Many-to-Many Match Diagnostic Module

Analyzes candidate and final predictions:
- How many Source 2 entities match multiple Source 1 entities
- How many Source 3 entities match multiple Source 1 entities
- Frequency distribution of multi-S1 associations
- Assesses potential false merge risk without forcing artificial one-to-one constraints.
"""

from collections import Counter
import pandas as pd

def analyze_collisions(df_predictions, s1_col='source1_entity_id', cand_col='candidate_entity_id'):
    """
    Performs diagnostic collision analysis on candidate or prediction pairs.
    df_predictions can have (s1_col, cand_col).
    """
    if len(df_predictions) == 0:
        return {
            'total_predictions': 0,
            'unique_s1': 0,
            'unique_candidates': 0,
            'colliding_candidates_count': 0,
            'collision_rate': 0.0,
            's2_collisions': 0,
            's3_collisions': 0,
            'multi_match_histogram': {}
        }
        
    cand_counts = Counter(df_predictions[cand_col])
    
    colliding_cands = {c: count for c, count in cand_counts.items() if count > 1}
    s2_colliding = sum(1 for c in colliding_cands if str(c).startswith('S2'))
    s3_colliding = sum(1 for c in colliding_cands if str(c).startswith('S3'))
    
    # Histogram of S1 entities matched per candidate
    hist = Counter(cand_counts.values())
    
    unique_s1 = df_predictions[s1_col].nunique()
    unique_cand = len(cand_counts)
    
    report = {
        'total_predictions': len(df_predictions),
        'unique_s1': unique_s1,
        'unique_candidates': unique_cand,
        'colliding_candidates_count': len(colliding_cands),
        'collision_rate': len(colliding_cands) / unique_cand if unique_cand > 0 else 0.0,
        's2_collisions': s2_colliding,
        's3_collisions': s3_colliding,
        'multi_match_histogram': dict(sorted(hist.items()))
    }
    return report
