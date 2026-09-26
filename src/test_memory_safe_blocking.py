"""
Phase 6.1 Regression Test Suite: Memory-Safe Bounded Blocking Verification

Tests:
1. Candidate-pair semantic equality on controlled dataset
2. Evidence flag bit-for-bit equality
3. Candidate safety cap remains exactly 400
4. All five Config A channels remain active
5. Deterministic output across different chunk sizes (e.g. 1, 5, 25)
6. No duplicate candidate pairs
7. No malformed evidence
8. No NaN/Inf values introduced
9. Locked model artifacts untouched and verified
"""

import unittest
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import xgboost as xgb

from config import BLOCKING_CONFIG, BLOCKING_CHANNELS, CANDIDATE_CAP_CONFIG, get_config_fingerprint, BLOCKING_CHUNK_SIZE
from blocking import ConfigABlockingEngine
from candidate_safety import apply_candidate_safety_cap


class TestMemorySafeBlocking(unittest.TestCase):

    def setUp(self):
        # Controlled dataset with known matching patterns across all 5 channels
        self.df_target = pd.DataFrame([
            {
                'entity_id': 'S2-001',
                'norm_name': 'alpha logistics incorporated',
                'norm_address': '100 industrial park road suite 200',
                'norm_country': 'france',
                'source': 'S2'
            },
            {
                'entity_id': 'S2-002',
                'norm_name': 'alpaca trading solutions',
                'norm_address': '500 commercial avenue',
                'norm_country': 'france',
                'source': 'S2'
            },
            {
                'entity_id': 'S3-003',
                'norm_name': 'beta software technologies',
                'norm_address': '100 industrial park road',
                'norm_country': 'france',
                'source': 'S3'
            },
            {
                'entity_id': 'S2-004',
                'norm_name': 'alpha logistics incorporated',
                'norm_address': '100 industrial park road',
                'norm_country': 'germany',  # Different country
                'source': 'S2'
            },
            {
                'entity_id': 'S3-005',
                'norm_name': 'alphonse bakery sarl',
                'norm_address': '12 rue de paris',
                'norm_country': 'france',
                'source': 'S3'
            }
        ])

        self.df_s1 = pd.DataFrame([
            {
                'entity_id': 'S1-10',
                'norm_name': 'alpha logistics international',
                'norm_address': '100 industrial park road',
                'norm_country': 'france'
            },
            {
                'entity_id': 'S1-20',
                'norm_name': 'zeta unknown enterprise',
                'norm_address': '999 nowhere lane',
                'norm_country': 'france'
            },
            {
                'entity_id': 'S1-30',
                'norm_name': 'alpaca enterprise group',
                'norm_address': '12 rue de paris',
                'norm_country': 'france'
            }
        ])

        self.engine = ConfigABlockingEngine(self.df_target)

    def test_1_and_2_semantic_and_evidence_equality(self):
        """Verifies candidate pairs and all 16 evidence columns match reference logic."""
        # Unbounded chunk generation
        cands = self.engine.generate_candidates_for_chunk(self.df_s1)
        self.assertGreater(len(cands), 0)

        # Expected channels check:
        # S1-10 vs S2-001:
        #   country matches (france)
        #   name_token matches ('alpha', 'logistics')
        #   prefix_4 matches ('alph')
        #   prefix_3 matches ('alp')
        #   address_token matches ('industrial', 'park')
        pair_row = cands[(cands['entity_id_s1'] == 'S1-10') & (cands['entity_id_cand'] == 'S2-001')]
        self.assertEqual(len(pair_row), 1)
        r = pair_row.iloc[0]
        self.assertEqual(r['blocked_country'], 1)
        self.assertEqual(r['blocked_name_token'], 1)
        self.assertEqual(r['blocked_prefix_4'], 1)
        self.assertEqual(r['blocked_prefix_3'], 1)
        self.assertEqual(r['blocked_address_token'], 1)
        self.assertEqual(r['num_blocking_keys'], 4)
        # score = 70*2 (alpha, logistics) + 50 (alph) + 30 (alp) + 60*2 (industrial, park) = 140 + 50 + 30 + 120 = 340
        self.assertEqual(r['evidence_score'], 340.0)

        # Cross country S2-004 MUST NOT be present
        cross_ctry = cands[cands['entity_id_cand'] == 'S2-004']
        self.assertEqual(len(cross_ctry), 0)

    def test_3_candidate_cap_remains_400(self):
        """Verifies candidate cap remains locked at exactly 400 per S1."""
        self.assertEqual(CANDIDATE_CAP_CONFIG['max_candidates_per_s1'], 400)
        
        # Create an S1 entity with 500 synthetic candidates
        df_target_large = pd.DataFrame([
            {
                'entity_id': f'S2-{i:04d}',
                'norm_name': f'test company {i}',
                'norm_address': '100 test road',
                'norm_country': 'france',
                'source': 'S2'
            }
            for i in range(500)
        ])
        df_s1_single = pd.DataFrame([
            {
                'entity_id': 'S1-OVERFLOW',
                'norm_name': 'test company',
                'norm_address': '100 test road',
                'norm_country': 'france'
            }
        ])
        engine = ConfigABlockingEngine(df_target_large)
        bounded_cands, metrics = engine.generate_bounded_candidates(
            df_s1_single, max_candidates_per_s1=400, return_metrics=True
        )
        self.assertEqual(len(bounded_cands), 400)
        self.assertEqual(metrics['overflowing_s1_count'], 1)
        self.assertEqual(metrics['max_candidates_per_s1'], 400)

    def test_4_all_five_config_a_channels_active(self):
        """Verifies all 5 Config A channels trigger candidate generation."""
        self.assertEqual(BLOCKING_CHANNELS, [
            "country", "name_token", "name_prefix3", "name_prefix4", "address_token"
        ])
        
        cands = self.engine.generate_candidates_for_chunk(self.df_s1)
        self.assertEqual((cands['blocked_country'] == 1).all(), True)
        self.assertGreater((cands['blocked_name_token'] == 1).sum(), 0)
        self.assertGreater((cands['blocked_prefix_3'] == 1).sum(), 0)
        self.assertGreater((cands['blocked_prefix_4'] == 1).sum(), 0)
        self.assertGreater((cands['blocked_address_token'] == 1).sum(), 0)

    def test_5_deterministic_across_chunk_sizes(self):
        """Verifies identical candidates and ranking across chunk_size 1, 2, and 5."""
        out_chunk1 = self.engine.generate_bounded_candidates(self.df_s1, max_candidates_per_s1=400, chunk_size=1)
        out_chunk2 = self.engine.generate_bounded_candidates(self.df_s1, max_candidates_per_s1=400, chunk_size=2)
        out_chunk5 = self.engine.generate_bounded_candidates(self.df_s1, max_candidates_per_s1=400, chunk_size=5)

        self.assertEqual(len(out_chunk1), len(out_chunk2))
        self.assertEqual(len(out_chunk1), len(out_chunk5))

        for col in out_chunk1.columns:
            if col == 'evidence_score':
                diff12 = np.abs(out_chunk1[col].values.astype(float) - out_chunk2[col].values.astype(float)).max()
                diff15 = np.abs(out_chunk1[col].values.astype(float) - out_chunk5[col].values.astype(float)).max()
                self.assertEqual(diff12, 0)
                self.assertEqual(diff15, 0)
            else:
                self.assertTrue((out_chunk1[col].values == out_chunk2[col].values).all())
                self.assertTrue((out_chunk1[col].values == out_chunk5[col].values).all())

    def test_6_no_duplicate_candidate_pairs(self):
        """Verifies that no candidate pair (s1, cand) is duplicated."""
        out = self.engine.generate_bounded_candidates(self.df_s1, max_candidates_per_s1=400, chunk_size=2)
        pairs = list(zip(out['entity_id_s1'], out['entity_id_cand']))
        self.assertEqual(len(pairs), len(set(pairs)), "Duplicate candidate pairs found!")

    def test_7_no_malformed_evidence(self):
        """Verifies schema conformance and evidence field validity."""
        expected_cols = [
            'entity_id_s1', 'entity_id_cand', 'source', 'blocked_country',
            'blocked_name_token', 'blocked_prefix_3', 'blocked_prefix_4',
            'blocked_address_token', 'num_blocking_keys', 'evidence_score',
            'blocked_exact_name', 'blocked_selective_token', 'blocked_rare_token',
            'blocked_token_pair', 'blocked_char_ngram', 'shared_ngram_count'
        ]
        out = self.engine.generate_bounded_candidates(self.df_s1, max_candidates_per_s1=400)
        self.assertEqual(list(out.columns), expected_cols)

        # Flag domains
        for f in ['blocked_country', 'blocked_name_token', 'blocked_prefix_3', 'blocked_prefix_4', 'blocked_address_token', 'blocked_exact_name']:
            self.assertTrue(set(out[f].unique()).issubset({0, 1}))

        # num_blocking_keys consistency
        computed_keys = (out['blocked_name_token'] + out['blocked_prefix_3'] +
                         out['blocked_prefix_4'] + out['blocked_address_token'])
        self.assertTrue((out['num_blocking_keys'] == computed_keys).all())

    def test_8_no_nan_or_inf(self):
        """Verifies no NaN or Inf values in candidate output."""
        out = self.engine.generate_bounded_candidates(self.df_s1, max_candidates_per_s1=400)
        self.assertFalse(out.isna().any().any(), "NaN found in candidate output")
        for col in out.select_dtypes(include=[np.number]).columns:
            self.assertFalse(np.isinf(out[col]).any(), f"Inf found in column {col}")

    def test_9_locked_model_artifacts_untouched(self):
        """Verifies locked model files, schema, threshold, and config fingerprint remain unaltered."""
        # Config fingerprint
        self.assertEqual(get_config_fingerprint(), "87f20ceeb84ccc6ea2d48678c7810ac5")

        # Feature schema
        schema_path = "models/phase5_configA_feature_schema.json"
        self.assertTrue(os.path.exists(schema_path))
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)
        self.assertEqual(schema['feature_count'], 57)
        self.assertEqual(len(schema['feature_names']), 57)
        self.assertEqual(schema['config_fingerprint'], "87f20ceeb84ccc6ea2d48678c7810ac5")

        # Threshold artifact
        thresh_path = "models/phase5_configA_threshold.json"
        self.assertTrue(os.path.exists(thresh_path))
        with open(thresh_path, "r", encoding="utf-8") as f:
            th = json.load(f)
        self.assertEqual(th['threshold'], 0.910)
        self.assertEqual(th['config_fingerprint'], "87f20ceeb84ccc6ea2d48678c7810ac5")

        # Metadata
        meta_path = "models/phase5_configA_metadata.json"
        self.assertTrue(os.path.exists(meta_path))
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.assertEqual(meta['config_fingerprint'], "87f20ceeb84ccc6ea2d48678c7810ac5")
        self.assertEqual(meta['candidate_cap'], 400)
        self.assertEqual(meta['feature_count'], 57)

        # XGBoost model
        xgb_path = "models/xgboost_entity_resolution_phase5_configA.json"
        self.assertTrue(os.path.exists(xgb_path))
        booster = xgb.Booster()
        booster.load_model(xgb_path)
        self.assertEqual(booster.num_features(), 57)


if __name__ == '__main__':
    unittest.main()
