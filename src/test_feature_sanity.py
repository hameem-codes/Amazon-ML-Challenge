"""
Phase 4A: Feature Sanity Tests

Tests all 10 edge cases required by Phase 4A specification:
 CASE 1: Identical records
 CASE 2: Completely different names
 CASE 3: Same name, different address
 CASE 4: Missing candidate name
 CASE 5: Both names missing
 CASE 6: Same country
 CASE 7: Different countries
 CASE 8: Reordered tokens
 CASE 9: Punctuation variation
 CASE 10: Unicode / non-English text
"""

import unittest
from features import build_single_pair_features
from normalize import normalize_business_name, normalize_business_address, normalize_country

class TestFeatureSanity(unittest.TestCase):

    def test_case_1_identical_records(self):
        f = build_single_pair_features(
            s1_name="Acme Corporation",
            s1_addr="123 Main Street",
            s1_country="United States",
            cand_name="Acme Corporation",
            cand_addr="123 Main Street",
            cand_country="United States",
            cand_source="S2"
        )
        self.assertEqual(f['name_exact'], 1)
        self.assertEqual(f['address_exact'], 1)
        self.assertEqual(f['country_exact'], 1)
        self.assertAlmostEqual(f['name_levenshtein_similarity'], 1.0, places=4)
        self.assertAlmostEqual(f['name_jaro_winkler'], 1.0, places=4)
        self.assertAlmostEqual(f['name_token_jaccard'], 1.0, places=4)

    def test_case_2_completely_different_names(self):
        f = build_single_pair_features(
            s1_name="Apple Inc",
            s1_addr="1 Infinite Loop Cupertino",
            s1_country="US",
            cand_name="Zebra Logistics",
            cand_addr="456 Elm Street",
            cand_country="US",
            cand_source="S3"
        )
        self.assertEqual(f['name_exact'], 0)
        self.assertLess(f['name_levenshtein_similarity'], 0.3)
        self.assertLess(f['name_jaro_winkler'], 0.55)
        self.assertEqual(f['name_token_jaccard'], 0.0)

    def test_case_3_same_name_different_address(self):
        f = build_single_pair_features(
            s1_name="Starbucks Coffee",
            s1_addr="123 Pike Place Seattle",
            s1_country="US",
            cand_name="Starbucks Coffee",
            cand_addr="999 Broadway New York",
            cand_country="US",
            cand_source="S2"
        )
        self.assertEqual(f['name_exact'], 1)
        self.assertAlmostEqual(f['name_levenshtein_similarity'], 1.0)
        self.assertEqual(f['address_exact'], 0)
        self.assertLess(f['address_levenshtein_similarity'], 0.5)

    def test_case_4_missing_candidate_name(self):
        f = build_single_pair_features(
            s1_name="Tata Consultancy Services",
            s1_addr="Nariman Point Mumbai",
            s1_country="India",
            cand_name="",
            cand_addr="Nariman Point Mumbai",
            cand_country="India",
            cand_source="S2"
        )
        self.assertEqual(f['candidate_name_missing'], 1)
        self.assertEqual(f['s1_name_missing'], 0)
        self.assertEqual(f['both_name_missing'], 0)
        self.assertEqual(f['name_exact'], 0)
        self.assertEqual(f['name_levenshtein_similarity'], 0.0)
        self.assertEqual(f['name_token_jaccard'], 0.0)

    def test_case_5_both_names_missing(self):
        f = build_single_pair_features(
            s1_name="",
            s1_addr="100 Technology Square",
            s1_country="US",
            cand_name="",
            cand_addr="100 Technology Square",
            cand_country="US",
            cand_source="S3"
        )
        self.assertEqual(f['both_name_missing'], 1)
        self.assertEqual(f['both_name_present'], 0)
        self.assertEqual(f['name_exact'], 0)
        self.assertEqual(f['name_levenshtein_similarity'], 0.0)
        self.assertEqual(f['name_jaro_winkler'], 0.0)

    def test_case_6_same_country(self):
        f = build_single_pair_features(
            s1_name="Alpha", s1_addr="Addr", s1_country="France",
            cand_name="Alpha", cand_addr="Addr", cand_country="France",
            cand_source="S2"
        )
        self.assertEqual(f['country_exact'], 1)
        self.assertEqual(f['country_mismatch'], 0)

    def test_case_7_different_countries(self):
        f = build_single_pair_features(
            s1_name="Alpha", s1_addr="Addr", s1_country="France",
            cand_name="Alpha", cand_addr="Addr", cand_country="India",
            cand_source="S3"
        )
        self.assertEqual(f['country_exact'], 0)
        self.assertEqual(f['country_mismatch'], 1)

    def test_case_8_reordered_tokens(self):
        f = build_single_pair_features(
            s1_name="ABC INDUSTRIES PRIVATE LIMITED",
            s1_addr="Road 1",
            s1_country="India",
            cand_name="PRIVATE LIMITED ABC INDUSTRIES",
            cand_addr="Road 1",
            cand_country="India",
            cand_source="S2"
        )
        self.assertEqual(f['name_exact'], 0)
        self.assertAlmostEqual(f['name_token_jaccard'], 1.0, places=4)
        self.assertAlmostEqual(f['name_token_overlap'], 1.0, places=4)

    def test_case_9_punctuation_variation(self):
        # After normalization
        s1_n = normalize_business_name("Amazon.com, Inc.")
        c_n = normalize_business_name("Amazon com Inc")
        f = build_single_pair_features(
            s1_name=s1_n, s1_addr="", s1_country="US",
            cand_name=c_n, cand_addr="", cand_country="US",
            cand_source="S2"
        )
        self.assertEqual(f['name_exact'], 1)
        self.assertAlmostEqual(f['name_levenshtein_similarity'], 1.0)

    def test_case_10_unicode_non_english(self):
        # Hindi: रिलायंस इंडस्ट्रीज (Reliance Industries)
        s1_n = normalize_business_name("रिलायंस इंडस्ट्रीज लिमिटेड")
        c_n = normalize_business_name("रिलायंस इंडस्ट्रीज")
        f = build_single_pair_features(
            s1_name=s1_n, s1_addr="मुंबई", s1_country="भारत",
            cand_name=c_n, cand_addr="मुंबई", cand_country="भारत",
            cand_source="S2"
        )
        self.assertEqual(f['name_s1_contains_candidate'], 1)
        self.assertGreater(f['name_token_overlap'], 0.99)
        self.assertGreater(f['name_jaro_winkler'], 0.8)
        self.assertEqual(f['address_exact'], 1)
        self.assertEqual(f['country_exact'], 1)

if __name__ == '__main__':
    unittest.main()
