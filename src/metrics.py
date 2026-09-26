"""
Evaluation and Metrics Module

Provides metric calculations for Entity Resolution:
1. compute_pairwise_metrics: Pair-level precision, recall, F0.5, TP, FP, FN
2. compute_macro_f05: Entity-level (Source 1 centered) Macro F0.5 per competition specification
3. evaluate_threshold_grid: Sweeps classification thresholds and calculates both pairwise and macro metrics
4. compute_calibration_diagnostics: Evaluates probability calibration across bins, Brier score, and ECE
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import brier_score_loss

def compute_pairwise_metrics(y_true, y_pred):
    """
    Computes pair-level classification metrics:
    Precision, Recall, F0.5, TP, FP, FN, TN
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred = np.asarray(y_pred, dtype=np.int32)
    
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # F_beta with beta = 0.5 (beta^2 = 0.25)
    beta_sq = 0.25
    denom = (beta_sq * precision) + recall
    f05 = ((1.0 + beta_sq) * precision * recall) / denom if denom > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f0_5': f05,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn,
        'predicted_matches': tp + fp
    }

def compute_macro_f05(val_s1_ids, true_pairs_set, predicted_pairs_set):
    """
    Computes macro-averaged F0.5 across all validation Source 1 entities:
    - S1 with no true matches and no predicted matches -> score = 1.0 (correct singleton/empty prediction)
    - S1 with true matches but no predicted matches -> score = 0.0
    - S1 with no true matches but predicted matches -> score = 0.0
    - S1 with true matches and predicted matches -> (1 + beta^2)*P*R / (beta^2*P + R)
    """
    # Group true matches by S1
    gt_by_s1 = defaultdict(set)
    for s1_id, cand_id in true_pairs_set:
        if s1_id in val_s1_ids:
            gt_by_s1[s1_id].add(cand_id)
            
    # Group predicted matches by S1
    pred_by_s1 = defaultdict(set)
    for s1_id, cand_id in predicted_pairs_set:
        if s1_id in val_s1_ids:
            pred_by_s1[s1_id].add(cand_id)
            
    f05_list = []
    precision_list = []
    recall_list = []
    zero_match_count = 0
    one_or_more_count = 0
    
    beta_sq = 0.25
    
    for s1_id in val_s1_ids:
        t_set = gt_by_s1.get(s1_id, set())
        p_set = pred_by_s1.get(s1_id, set())
        
        if len(p_set) == 0:
            zero_match_count += 1
        else:
            one_or_more_count += 1
            
        if len(t_set) == 0 and len(p_set) == 0:
            f05_list.append(1.0)
            precision_list.append(1.0)
            recall_list.append(1.0)
        elif len(t_set) == 0 and len(p_set) > 0:
            f05_list.append(0.0)
            precision_list.append(0.0)
            recall_list.append(1.0)
        elif len(t_set) > 0 and len(p_set) == 0:
            f05_list.append(0.0)
            precision_list.append(1.0)
            recall_list.append(0.0)
        else:
            tp = len(t_set.intersection(p_set))
            fp = len(p_set - t_set)
            fn = len(t_set - p_set)
            
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            denom = (beta_sq * p) + r
            f05 = ((1.0 + beta_sq) * p * r) / denom if denom > 0 else 0.0
            
            f05_list.append(f05)
            precision_list.append(p)
            recall_list.append(r)
            
    return {
        'macro_f0_5': float(np.mean(f05_list)) if f05_list else 0.0,
        'macro_precision': float(np.mean(precision_list)) if precision_list else 0.0,
        'macro_recall': float(np.mean(recall_list)) if recall_list else 0.0,
        'zero_match_s1_count': zero_match_count,
        'one_or_more_match_s1_count': one_or_more_count
    }

def evaluate_threshold_grid(df_val_with_probs, val_s1_ids, true_pairs_set, thresholds=None):
    """
    Sweeps a grid of thresholds on the validation dataframe.
    Computes both pairwise and entity-level macro F0.5 metrics.
    """
    if thresholds is None:
        thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.95, 0.97, 0.98]
        
    y_true = df_val_with_probs['label'].values
    y_probs = df_val_with_probs['predicted_probability'].values
    s1_ids = df_val_with_probs['source1_entity_id'].values
    cand_ids = df_val_with_probs['candidate_entity_id'].values
    
    results = []
    
    for thresh in thresholds:
        y_pred = (y_probs >= thresh).astype(np.int32)
        pair_metrics = compute_pairwise_metrics(y_true, y_pred)
        
        # Build set of predicted pairs for macro F0.5
        pred_indices = np.where(y_pred == 1)[0]
        pred_pairs = {(s1_ids[i], cand_ids[i]) for i in pred_indices}
        
        macro_metrics = compute_macro_f05(val_s1_ids, true_pairs_set, pred_pairs)
        
        res = {
            'threshold': thresh,
            'macro_f0_5': macro_metrics['macro_f0_5'],
            'macro_precision': macro_metrics['macro_precision'],
            'macro_recall': macro_metrics['macro_recall'],
            'pairwise_f0_5': pair_metrics['f0_5'],
            'pairwise_precision': pair_metrics['precision'],
            'pairwise_recall': pair_metrics['recall'],
            'tp': pair_metrics['tp'],
            'fp': pair_metrics['fp'],
            'fn': pair_metrics['fn'],
            'predicted_matches': pair_metrics['predicted_matches'],
            'zero_match_s1_count': macro_metrics['zero_match_s1_count'],
            'one_or_more_s1_count': macro_metrics['one_or_more_match_s1_count']
        }
        results.append(res)
        
    return pd.DataFrame(results)

def compute_calibration_diagnostics(y_true, y_probs, n_bins=10):
    """
    Computes calibration bins, empirical match rates, Brier score, and Expected Calibration Error (ECE).
    """
    y_true = np.asarray(y_true, dtype=np.float32)
    y_probs = np.asarray(y_probs, dtype=np.float32)
    
    brier = float(brier_score_loss(y_true, y_probs))
    
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_rows = []
    total_samples = len(y_probs)
    ece = 0.0
    
    for i in range(n_bins):
        low, high = bins[i], bins[i+1]
        mask = (y_probs >= low) & (y_probs <= high if i == n_bins - 1 else y_probs < high)
        count = int(np.sum(mask))
        if count > 0:
            mean_prob = float(np.mean(y_probs[mask]))
            empirical_rate = float(np.mean(y_true[mask]))
            bin_ece = (count / total_samples) * abs(empirical_rate - mean_prob)
            ece += bin_ece
            bin_rows.append({
                'bin_range': f"{low:.1f}-{high:.1f}",
                'count': count,
                'mean_predicted_prob': mean_prob,
                'empirical_match_rate': empirical_rate,
                'absolute_gap': abs(empirical_rate - mean_prob)
            })
        else:
            bin_rows.append({
                'bin_range': f"{low:.1f}-{high:.1f}",
                'count': 0,
                'mean_predicted_prob': 0.0,
                'empirical_match_rate': 0.0,
                'absolute_gap': 0.0
            })
            
    return {
        'brier_score': brier,
        'expected_calibration_error': float(ece),
        'bins_table': pd.DataFrame(bin_rows)
    }
