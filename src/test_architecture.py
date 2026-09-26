"""
Architecture Alignment and Component Unit Tests

Tests:
1. Candidate safety cap and deterministic evidence ranking
2. Blocking evidence preservation (exact, token, pair, ngram)
3. Entity-level train/validation split isolation (no S1 leakage)
4. Hard-negative selection & dynamic scale_pos_weight
5. Transductive unlabeled TF-IDF vectorizer consistency
6. Many-to-many collision diagnostic analysis
7. Central configuration hashing and reproducibility
"""

import unittest
import pandas as pd
import numpy as np

from config import get_config_fingerprint, BLOCKING_CONFIG, CANDIDATE_CAP_CONFIG
from candidate_safety import apply_candidate_safety_cap, compute_evidence_score
from split import split_by_entity_id, split_candidate_pairs
from training_pairs import construct_training_pairs
from tfidf_model import TransductiveTFIDF
from collision import analyze_collisions
from blocking import ProductionBlockingEngine

class TestArchitectureAlignment(unittest.TestCase):

    def test_config_fingerprint(self):
        h1 = get_config_fingerprint()
        h2 = get_config_fingerprint()
        self.assertEqual(h1, h2, "Config hash must be strictly deterministic")
        self.assertEqual(len(h1), 32)

    def test_blocking_evidence_preservation(self):
        # Target dataframe
        df_target = pd.DataFrame([
            {'entity_id': 'S2-101', 'norm_name': 'acme electronics limited', 'source': 'S2'},
            {'entity_id': 'S3-202', 'norm_name': 'acme robotics inc', 'source': 'S3'},
            {'entity_id': 'S2-303', 'norm_name': 'completely different name', 'source': 'S2'}
        ])
        
        engine = ProductionBlockingEngine(df_target)
        
        # Source 1 dataframe
        df_s1 = pd.DataFrame([
            {'entity_id': 'S1-1', 'norm_name': 'acme electronics limited'},
            {'entity_id': 'S1-2', 'norm_name': 'acme robotics inc'}
        ])
        
        candidates = engine.generate_candidates_with_evidence(df_s1)
        
        self.assertGreater(len(candidates), 0)
        expected_cols = [
            'entity_id_s1', 'entity_id_cand', 'source', 'blocked_exact_name',
            'blocked_selective_token', 'blocked_rare_token', 'blocked_token_pair',
            'blocked_char_ngram', 'shared_ngram_count', 'num_blocking_keys'
        ]
        for col in expected_cols:
            self.assertIn(col, candidates.columns)
            
        # Check exact name hit for S1-1 -> S2-101
        match = candidates[(candidates['entity_id_s1'] == 'S1-1') & (candidates['entity_id_cand'] == 'S2-101')]
        self.assertEqual(len(match), 1)
        self.assertEqual(match.iloc[0]['blocked_exact_name'], 1)
        self.assertGreaterEqual(match.iloc[0]['num_blocking_keys'], 1)

    def test_candidate_safety_cap(self):
        # Create 10 candidates for S1-1 with varied evidence scores
        rows = []
        for i in range(10):
            rows.append({
                'entity_id_s1': 'S1-1',
                'entity_id_cand': f'S2-{i}',
                'source': 'S2',
                'blocked_exact_name': 1 if i == 0 else 0,
                'num_blocking_keys': 5 if i == 0 else (10 - i),
                'shared_ngram_count': 10 - i,
                'blocked_token_pair': 1 if i < 3 else 0,
                'blocked_rare_token': 0,
                'blocked_selective_token': 1
            })
        df_cands = pd.DataFrame(rows)
        
        # Cap at 3 candidates
        capped_df, metrics = apply_candidate_safety_cap(df_cands, max_candidates_per_s1=3)
        
        self.assertEqual(len(capped_df), 3)
        self.assertEqual(metrics['before_cap_candidates'], 10)
        self.assertEqual(metrics['after_cap_candidates'], 3)
        self.assertEqual(metrics['overflowing_s1_count'], 1)
        # Highest evidence should be S2-0
        self.assertEqual(capped_df.iloc[0]['entity_id_cand'], 'S2-0')

    def test_entity_level_split_isolation(self):
        s1_ids = [f'S1-{i}' for i in range(100)]
        df_cands = pd.DataFrame({
            'entity_id_s1': np.repeat(s1_ids, 5),
            'entity_id_cand': [f'S2-{j}' for j in range(500)],
            'source': 'S2'
        })
        
        train_df, val_df, train_s1, val_s1 = split_candidate_pairs(df_cands, train_ratio=0.8, random_seed=42)
        
        # Check strict disjointness
        overlap = train_s1.intersection(val_s1)
        self.assertEqual(len(overlap), 0, "No S1 entity ID may exist in both train and validation splits")
        
        train_pairs_s1 = set(train_df['entity_id_s1'])
        val_pairs_s1 = set(val_df['entity_id_s1'])
        self.assertEqual(len(train_pairs_s1.intersection(val_pairs_s1)), 0)

    def test_training_pair_hard_negative_construction(self):
        # 2 positive pairs, 20 negative pairs
        true_pairs = {('S1-1', 'S2-1'), ('S1-2', 'S2-2')}
        
        rows = [
            {'entity_id_s1': 'S1-1', 'entity_id_cand': 'S2-1', 'source': 'S2', 'num_blocking_keys': 3, 'shared_ngram_count': 5, 'blocked_exact_name': 1},
            {'entity_id_s1': 'S1-2', 'entity_id_cand': 'S2-2', 'source': 'S2', 'num_blocking_keys': 4, 'shared_ngram_count': 6, 'blocked_exact_name': 1}
        ]
        # 20 negatives
        for i in range(3, 23):
            rows.append({
                'entity_id_s1': 'S1-1',
                'entity_id_cand': f'S2-{i}',
                'source': 'S2',
                'num_blocking_keys': 2 if i < 10 else 1,
                'shared_ngram_count': 4 if i < 10 else 1,
                'blocked_exact_name': 0
            })
        df_cands = pd.DataFrame(rows)
        
        df_train_pairs, stats = construct_training_pairs(df_cands, true_pairs, hard_neg_ratio=5, random_seed=42)
        
        # 2 positives + min(20, 2*5)=10 negatives = 12 total pairs
        self.assertEqual(stats['positive_count'], 2)
        self.assertEqual(stats['selected_negative_count'], 10)
        self.assertEqual(len(df_train_pairs), 12)
        self.assertAlmostEqual(stats['scale_pos_weight'], 5.0)

    def test_transductive_tfidf_consistency(self):
        names = ["apple computer inc", "google llc", "amazon web services", "france telematics"]
        addrs = ["cupertino california", "mountain view", "seattle washington", "paris france"]
        
        tfidf = TransductiveTFIDF(max_features=50)
        tfidf.fit_from_text_iterables(names, addrs)
        
        models = tfidf.get_models_dict()
        self.assertIn('name_tfidf', models)
        self.assertIn('address_tfidf', models)
        
        v1 = models['name_tfidf'].transform(["apple computer"]).toarray()
        self.assertEqual(v1.shape[1], min(50, len(models['name_tfidf'].vocabulary_)))

    def test_collision_analysis(self):
        df_preds = pd.DataFrame([
            {'source1_entity_id': 'S1-1', 'candidate_entity_id': 'S2-10'},
            {'source1_entity_id': 'S1-2', 'candidate_entity_id': 'S2-10'}, # collision on S2-10
            {'source1_entity_id': 'S1-3', 'candidate_entity_id': 'S3-20'}
        ])
        report = analyze_collisions(df_preds)
        self.assertEqual(report['total_predictions'], 3)
        self.assertEqual(report['colliding_candidates_count'], 1)
        self.assertEqual(report['s2_collisions'], 1)
        self.assertEqual(report['s3_collisions'], 0)

if __name__ == '__main__':
    unittest.main()
