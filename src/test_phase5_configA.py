"""
Phase 5 Unit and Integration Tests: Config A Blocker & Pipeline Verification

Verifies:
1. Config A channels are exactly: country, name_token, name_prefix3, name_prefix4, address_token
2. Forbidden old blocking channels (char 3-gram, rare token, selective token, token pair, exact-name) are not active
3. Feature schema has exactly 57 features
4. Config A blocking execution generates valid candidates with evidence
5. Candidate safety cap preserves evidence ranking
"""

import unittest
import pandas as pd
import numpy as np

from config import BLOCKING_CONFIG, BLOCKING_ARCHITECTURE, BLOCKING_CHANNELS, CANDIDATE_CAP_CONFIG
from blocking import ConfigABlockingEngine, ProductionBlockingEngine
from candidate_safety import apply_candidate_safety_cap, compute_evidence_score
from features import FEATURE_NAMES

class TestPhase5ConfigA(unittest.TestCase):

    def test_config_a_channels_exact(self):
        """Verifies Config A architecture name and exact 5 locked channels."""
        self.assertEqual(BLOCKING_ARCHITECTURE, "CONFIG_A_TEAM_BASELINE")
        expected_channels = ["country", "name_token", "name_prefix3", "name_prefix4", "address_token"]
        self.assertEqual(BLOCKING_CHANNELS, expected_channels)
        self.assertEqual(BLOCKING_CONFIG['architecture'], "CONFIG_A_TEAM_BASELINE")
        self.assertEqual(BLOCKING_CONFIG['channels'], expected_channels)

    def test_forbidden_channels_inactive(self):
        """Verifies forbidden old channels are NOT present in Config A blocking channels."""
        forbidden = [
            'char_3gram', 'char_ngram', 'ngram', 'rare_token', 
            'token_pair', 'exact_name', 'fuzzy', 'external_lookup'
        ]
        for f in forbidden:
            self.assertNotIn(f, BLOCKING_CHANNELS, f"Forbidden channel {f} must not be active in Config A")

    def test_feature_schema_57(self):
        """Verifies the authoritative feature schema has exactly 57 features."""
        self.assertEqual(len(FEATURE_NAMES), 57)
        self.assertEqual(len(set(FEATURE_NAMES)), 57, "Feature names must be unique")

    def test_config_a_blocking_execution(self):
        """Verifies Config A generates valid candidates partitioned by country."""
        df_target = pd.DataFrame([
            {
                'entity_id': 'S2-001',
                'norm_name': 'acme logistics global',
                'norm_address': '123 industrial park road suite 400',
                'norm_country': 'united states',
                'source': 'S2'
            },
            {
                'entity_id': 'S3-002',
                'norm_name': 'zenith technology solutions',
                'norm_address': '456 technology park drive',
                'norm_country': 'united states',
                'source': 'S3'
            },
            {
                'entity_id': 'S2-003',
                'norm_name': 'acme logistics global',
                'norm_address': '123 industrial park road',
                'norm_country': 'canada',  # Different country
                'source': 'S2'
            }
        ])
        
        df_s1 = pd.DataFrame([
            {
                'entity_id': 'S1-100',
                'norm_name': 'acme logistics services',
                'norm_address': '123 industrial park road',
                'norm_country': 'united states'
            }
        ])
        
        engine = ConfigABlockingEngine(df_target)
        cands = engine.generate_candidates_with_evidence(df_s1)
        
        # Verify candidate results
        self.assertGreater(len(cands), 0)
        cand_ids = set(cands['entity_id_cand'])
        # S2-001 matches on country, name_token ('acme', 'logistics'), prefix3 ('acm'), prefix4 ('acme'), address_token ('industrial', 'park')
        self.assertIn('S2-001', cand_ids)
        # S2-003 has different country ('canada' vs 'united states') -> MUST NOT MATCH
        self.assertNotIn('S2-003', cand_ids, "Cross-country candidate must be blocked")
        
        # Verify evidence columns
        row = cands[cands['entity_id_cand'] == 'S2-001'].iloc[0]
        self.assertEqual(row['blocked_country'], 1)
        self.assertEqual(row['blocked_name_token'], 1)
        self.assertEqual(row['blocked_prefix_3'], 1)
        self.assertEqual(row['blocked_prefix_4'], 1)
        self.assertEqual(row['blocked_address_token'], 1)
        self.assertGreaterEqual(row['num_blocking_keys'], 4)
        self.assertGreater(row['evidence_score'], 150)

    def test_candidate_safety_cap(self):
        """Verifies candidate safety cap deterministic sorting and retention."""
        df_cands = pd.DataFrame([
            {'entity_id_s1': 'S1-1', 'entity_id_cand': f'S2-{i}', 'evidence_score': i * 10, 'num_blocking_keys': 1}
            for i in range(10)
        ])
        
        capped, metrics = apply_candidate_safety_cap(df_cands, max_candidates_per_s1=5)
        self.assertEqual(len(capped), 5)
        self.assertEqual(metrics['overflowing_s1_count'], 1)
        # Should keep top 5 highest evidence scores (90, 80, 70, 60, 50)
        self.assertEqual(set(capped['entity_id_cand']), {'S2-9', 'S2-8', 'S2-7', 'S2-6', 'S2-5'})

if __name__ == '__main__':
    unittest.main()
