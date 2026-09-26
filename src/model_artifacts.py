"""
Phase 4B: Model Serialization and Artifact Locking Module

Provides utilities to:
1. save_locked_model: serializes XGBoost model in native JSON format
2. save_feature_schema: saves locked feature schema and ordering
3. save_locked_threshold: saves chosen threshold, validation metrics, and config fingerprint
4. save_metadata: records complete training, split, recall, and evaluation metadata
5. verify_and_load_locked_model: verifies that loaded model, schema, and threshold match current pipeline fingerprint
"""

import os
import json
import xgboost as xgb
from config import get_config_fingerprint, PIPELINE_VERSION

def save_locked_model(model, filepath='models/xgboost_entity_resolution_phase4b.json'):
    """
    Saves XGBoost classifier in native JSON format.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    model.save_model(filepath)
    return filepath

def save_feature_schema(feature_names, filepath='models/phase4b_feature_schema.json'):
    """
    Saves frozen feature schema with strict column ordering.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    schema_data = {
        'pipeline_version': PIPELINE_VERSION,
        'feature_count': len(feature_names),
        'feature_names': list(feature_names),
        'config_fingerprint': get_config_fingerprint()
    }
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(schema_data, f, indent=2)
    return filepath

def save_locked_threshold(threshold, validation_f0_5, filepath='models/phase4b_threshold.json'):
    """
    Locks the selected classification threshold and associated validation score.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    thresh_data = {
        'threshold': float(threshold),
        'metric': 'macro_F0.5',
        'validation_f0_5': float(validation_f0_5),
        'config_fingerprint': get_config_fingerprint(),
        'pipeline_version': PIPELINE_VERSION
    }
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(thresh_data, f, indent=2)
    return filepath

def save_phase4b_metadata(metadata_dict, filepath='models/phase4b_metadata.json'):
    """
    Saves comprehensive execution metadata.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    full_meta = {
        'pipeline_version': PIPELINE_VERSION,
        'config_fingerprint': get_config_fingerprint(),
        **metadata_dict
    }
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(full_meta, f, indent=2, default=str)
    return filepath

def verify_and_load_locked_model(model_path='models/xgboost_entity_resolution_phase4b.json',
                                  schema_path='models/phase4b_feature_schema.json',
                                  thresh_path='models/phase4b_threshold.json'):
    """
    Loads and strictly validates model artifacts against the current configuration fingerprint.
    """
    # 1. Verify schema
    with open(schema_path, 'r', encoding='utf-8') as f:
        schema = json.load(f)
    current_fp = get_config_fingerprint()
    if schema.get('config_fingerprint') != current_fp:
        raise ValueError(f"Schema config fingerprint mismatch! Saved: {schema.get('config_fingerprint')}, Current: {current_fp}")
        
    # 2. Verify threshold
    with open(thresh_path, 'r', encoding='utf-8') as f:
        thresh_info = json.load(f)
    if thresh_info.get('config_fingerprint') != current_fp:
        raise ValueError(f"Threshold config fingerprint mismatch! Saved: {thresh_info.get('config_fingerprint')}, Current: {current_fp}")
        
    # 3. Load model
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    
    return model, schema['feature_names'], thresh_info['threshold']
