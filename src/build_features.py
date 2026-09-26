"""
Phase 4A: Feature Pipeline Builder & Training Label Generator

Utilities for:
1. Generating ground truth labels for candidate pairs from train_ground_truth.tsv
2. Building TF-IDF models on training corpus in a leakage-safe way
3. Building the complete feature matrix (with or without labels)
"""

import ast
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from features import build_vectorized_features, FEATURE_NAMES

def parse_matched_ids(val_str):
    if pd.isna(val_str) or not val_str:
        return []
    val_str = str(val_str).strip()
    if val_str.startswith('[') and val_str.endswith(']'):
        val_str = val_str[1:-1]
    if ',' in val_str:
        raw_items = val_str.split(',')
    elif ' ' in val_str:
        raw_items = val_str.split(' ')
    else:
        raw_items = [val_str]
    items = []
    for x in raw_items:
        clean = x.strip().strip("'").strip('"')
        if clean:
            items.append(clean)
    return items

def load_ground_truth_map(gt_path='data/train/train_ground_truth.tsv', s1_ids=None):
    """
    Loads ground truth mapping as a set of (source1_entity_id, candidate_entity_id) true pairs.
    """
    gt = pd.read_csv(gt_path, sep='\t', dtype=str, na_filter=False)
    if s1_ids is not None:
        gt = gt[gt['source1_entity_id'].isin(s1_ids)]
        
    true_pairs = set()
    for _, row in gt.iterrows():
        s1_id = row['source1_entity_id']
        matches = parse_matched_ids(row['matched_entity_ids'])
        for m in matches:
            true_pairs.add((s1_id, m))
    return true_pairs

def attach_ground_truth_labels(df_features, true_pairs_set):
    """
    Creates a binary 'label' column:
      1 if (source1_entity_id, candidate_entity_id) in true_pairs_set else 0.
    Separated strictly from the core feature matrix.
    """
    s1_ids = df_features['source1_entity_id'].values
    c_ids = df_features['candidate_entity_id'].values
    labels = np.array([1 if (s1_ids[i], c_ids[i]) in true_pairs_set else 0 for i in range(len(df_features))], dtype=np.int32)
    return labels

def fit_tfidf_models(s1_records, target_records, max_features=None):
    """
    Fits transductive unlabeled TF-IDF vectorizers on names and addresses.
    Delegates to TransductiveTFIDF to ensure a single authoritative implementation.
    """
    from tfidf_model import TransductiveTFIDF
    names = []
    addrs = []
    
    for r in list(s1_records) + list(target_records):
        n = r.get('norm_name', '') if isinstance(r, dict) else ''
        if n:
            names.append(n)
        a = r.get('norm_address', '') if isinstance(r, dict) else ''
        if a:
            addrs.append(a)
            
    tfidf = TransductiveTFIDF(max_features=max_features)
    tfidf.fit_from_text_iterables(names, addrs)
    return tfidf.get_models_dict()

def build_feature_table(df_pairs, s1_df, target_df, tfidf_models=None):
    """
    Converts DataFrames to lookup dicts and generates the complete feature table.
    """
    s1_dict = s1_df.set_index('entity_id').to_dict(orient='index')
    target_dict = target_df.set_index('entity_id').to_dict(orient='index')
    
    return build_vectorized_features(df_pairs, s1_dict, target_dict, tfidf_models=tfidf_models)
