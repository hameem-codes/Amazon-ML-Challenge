"""
Central Configuration Module for Business Entity Resolution Pipeline

Centralizes all parameters, thresholds, random seeds, and provides
a deterministic configuration fingerprint (hash) to prevent train/test drift.
"""

import hashlib
import json

# Global Random Seed
RANDOM_SEED = 42

# Normalization Tokens
GENERIC_BUSINESS_TOKENS = {
    'llc', 'inc', 'ltd', 'limited', 'private', 'pvt', 'company', 'corporation',
    'services', 'service', 'group', 'enterprise', 'enterprises', 'co', 'corp'
}

GENERIC_ADDRESS_TOKENS = {
    'street', 'st', 'road', 'rd', 'avenue', 'ave', 'lane', 'ln', 'drive', 'dr',
    'boulevard', 'blvd', 'court', 'ct', 'suite', 'ste', 'floor', 'fl', 'building',
    'bldg', 'near', 'opp', 'opposite', 'behind', 'post', 'po', 'box', 'city', 'state',
    'district', 'nagar', 'colony', 'marg', 'road', 'salai', 'cross', 'main', 'layout'
}

# Locked Blocking Architecture: Config A — Team Baseline
BLOCKING_ARCHITECTURE = "CONFIG_A_TEAM_BASELINE"
BLOCKING_CHANNELS = [
    "country",
    "name_token",
    "name_prefix3",
    "name_prefix4",
    "address_token"
]

BLOCKING_CONFIG = {
    'architecture': BLOCKING_ARCHITECTURE,
    'channels': BLOCKING_CHANNELS,
    'country_blocking': True,
    'name_token_blocking': True,
    'prefix_3_blocking': True,
    'prefix_4_blocking': True,
    'address_token_blocking': True,
    'min_name_token_len': 2,
    'min_prefix_len_3': 3,
    'min_prefix_len_4': 4,
    'min_addr_token_len': 4
}

# Execution & Memory Parameters (Not part of config fingerprint)
BLOCKING_CHUNK_SIZE = 25  # Default S1 records per bounded processing chunk

# Candidate Safety Cap Configuration
CANDIDATE_CAP_CONFIG = {
    'max_candidates_per_s1': 400,      # Evaluated tournament safety cap per S1 entity
    'ranking_weights': {
        'name_token_weight': 70,
        'address_token_weight': 60,
        'prefix_4_weight': 50,
        'prefix_3_weight': 30,
        'num_keys_weight': 100,
        'exact_name_weight': 1000,
        # backward-compat keys
        'selective_token_weight': 70,
        'rare_token_weight': 0,
        'token_pair_weight': 0,
        'shared_ngrams_weight': 0
    }
}

# Training Data Construction Configuration
TRAINING_PAIRS_CONFIG = {
    'hard_negative_ratio': 15,         # Target negative-to-positive ratio for training
    'random_seed': RANDOM_SEED,
    'hard_negative_priority': [
        'evidence_score',
        'num_blocking_keys',
        'blocked_name_token',
        'blocked_address_token'
    ]
}

# Train/Validation Split Configuration
SPLIT_CONFIG = {
    'train_ratio': 0.8,
    'val_ratio': 0.2,
    'random_seed': RANDOM_SEED,
}

# TF-IDF Configuration
TFIDF_CONFIG = {
    'max_features': 15000,
    'token_pattern': r'(?u)\b\w+\b',
    'lowercase': False,                # Already normalized by conservative normalizer
}

# Model & Inference Defaults
MODEL_CONFIG = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'tree_method': 'hist',
    'learning_rate': 0.05,
    'max_depth': 6,
    'n_estimators': 300,
    'random_state': RANDOM_SEED,
    'early_stopping_rounds': 30,
}

PIPELINE_VERSION = "5.0.0"

def get_config_fingerprint():
    """
    Computes a deterministic MD5 hash of all operational pipeline settings.
    Used by train and test scripts to verify strict configuration alignment.
    """
    settings = {
        'version': PIPELINE_VERSION,
        'seed': RANDOM_SEED,
        'blocking': BLOCKING_CONFIG,
        'candidate_cap': CANDIDATE_CAP_CONFIG,
        'training_pairs': TRAINING_PAIRS_CONFIG,
        'split': SPLIT_CONFIG,
        'tfidf': TFIDF_CONFIG,
        'model': MODEL_CONFIG
    }
    dumped = json.dumps(settings, sort_keys=True)
    return hashlib.md5(dumped.encode('utf-8')).hexdigest()
