"""
Phase 4B: Model Artifacts Unit Tests

Tests:
1. Saving and loading native XGBoost model JSON
2. Feature schema serialization and ordering preservation
3. Threshold locking and metadata serialization
4. Strict configuration fingerprint verification on load
5. Rejection of tampered configuration fingerprint
"""

import unittest
import os
import json
import xgboost as xgb
import numpy as np

from config import get_config_fingerprint, PIPELINE_VERSION
from model_artifacts import (
    save_locked_model,
    save_feature_schema,
    save_locked_threshold,
    save_phase4b_metadata,
    verify_and_load_locked_model
)

class TestModelArtifacts(unittest.TestCase):

    def setUp(self):
        self.test_dir = 'scratch/test_models'
        os.makedirs(self.test_dir, exist_ok=True)
        self.model_path = os.path.join(self.test_dir, 'test_xgb.json')
        self.schema_path = os.path.join(self.test_dir, 'test_schema.json')
        self.thresh_path = os.path.join(self.test_dir, 'test_thresh.json')
        self.meta_path = os.path.join(self.test_dir, 'test_meta.json')
        
    def test_save_and_load_lifecycle(self):
        # 1. Fit small toy model
        X = np.array([[1.0, 2.0], [2.0, 3.0], [0.1, 0.2], [0.3, 0.4]])
        y = np.array([1, 1, 0, 0])
        model = xgb.XGBClassifier(n_estimators=3, max_depth=2, random_state=42)
        model.fit(X, y)
        
        # 2. Save artifacts
        save_locked_model(model, filepath=self.model_path)
        feature_names = ['feat_a', 'feat_b']
        save_feature_schema(feature_names, filepath=self.schema_path)
        save_locked_threshold(0.75, 0.912, filepath=self.thresh_path)
        save_phase4b_metadata({'scale_pos_weight': 15.0}, filepath=self.meta_path)
        
        # 3. Verify files exist
        self.assertTrue(os.path.exists(self.model_path))
        self.assertTrue(os.path.exists(self.schema_path))
        self.assertTrue(os.path.exists(self.thresh_path))
        self.assertTrue(os.path.exists(self.meta_path))
        
        # 4. Verify and load
        loaded_model, loaded_features, loaded_thresh = verify_and_load_locked_model(
            model_path=self.model_path,
            schema_path=self.schema_path,
            thresh_path=self.thresh_path
        )
        self.assertEqual(loaded_features, feature_names)
        self.assertEqual(loaded_thresh, 0.75)
        
        # 5. Check prediction equivalence
        orig_preds = model.predict_proba(X)
        load_preds = loaded_model.predict_proba(X)
        np.testing.assert_allclose(orig_preds, load_preds, rtol=1e-5)

    def test_fingerprint_mismatch_rejection(self):
        # Save schema with tampered fingerprint
        tampered_schema = {
            'pipeline_version': PIPELINE_VERSION,
            'feature_count': 2,
            'feature_names': ['f1', 'f2'],
            'config_fingerprint': 'tampered_fake_hash_12345'
        }
        with open(self.schema_path, 'w', encoding='utf-8') as f:
            json.dump(tampered_schema, f)
            
        with self.assertRaises(ValueError):
            verify_and_load_locked_model(
                model_path=self.model_path,
                schema_path=self.schema_path,
                thresh_path=self.thresh_path
            )

if __name__ == '__main__':
    unittest.main()
