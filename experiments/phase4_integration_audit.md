# Phase 4 — Final Integration Audit Report

This report documents the comprehensive integration audit and end-to-end dry run conducted on the Amazon ML Challenge 2026 Business Entity Resolution architecture prior to model training.

---

## A. Pipeline Trace & Execution Path

The table below traces the exact code path and authoritative Python functions/classes responsible for each step of the pipeline:

| Step | Pipeline Stage | Authoritative Module / Class / Function | Purpose & Output |
| :---: | :--- | :--- | :--- |
| **1** | Raw Ingestion | `scan_target_file` (`src/test_blocking_scale.py`) / Pandas TSV loaders | Streaming line-based loading of raw S1, S2, S3 without modifying raw inputs. |
| **2** | Conservative Normalization | `normalize_business_name`, `normalize_business_address`, `normalize_country` (`src/normalize.py`) | Unicode NFKC, casefolding, punctuation replacement, whitespace collapse. Retains raw and normalized columns. |
| **3** | Multi-Key Blocking | `ProductionBlockingEngine` (`src/blocking.py`) | Inverted-index multi-key blocking across 5 channels (filtered char 3-gram, selective token, rare token, token-pair, exact name). |
| **4** | Evidence Retention | `ProductionBlockingEngine.generate_candidates_with_evidence()` | Emits candidate pairs with explicit channel bitflags (`blocked_exact_name`, `blocked_selective_token`, etc.) and `num_blocking_keys`. |
| **5** | Candidate Safety Cap | `apply_candidate_safety_cap()` (`src/candidate_safety.py`) | Ranks overflowing candidates via `compute_evidence_score()`, enforces `max_candidates_per_s1`, measures recall impact. |
| **6** | Candidate Materialization | `to_csv('output/candidate_pairs.tsv')` | Explicitly materializes `candidate_pairs.tsv` satisfying the contract that final matches are always a subset of candidates. |
| **7** | Training Pair Construction | `construct_training_pairs()` (`src/training_pairs.py`) | Retains 100% of surviving true matches, selects evidence-ranked hard negatives, and computes dynamic `scale_pos_weight`. |
| **8** | Entity-Level Split | `split_candidate_pairs()` / `split_by_entity_id()` (`src/split.py`) | Strict partition on Source 1 ID ensuring reference entity isolation (`Train S1 ∩ Val S1 = ∅`). |
| **9** | Transductive TF-IDF | `TransductiveTFIDF` (`src/tfidf_model.py`) | Fits name and address vectorizers on unlabeled raw text union across train and test without label leakage. |
| **10**| Feature Engineering | `build_vectorized_features()` (`src/features.py`) & `build_feature_table()` (`src/build_features.py`) | Computes 57 finite pairwise features across Groups A–G + TF-IDF with directional missingness indicators. |
| **11**| Collision Diagnostics | `analyze_collisions()` (`src/collision.py`) | Evaluates many-to-many associations and collision frequency distributions without forcing artificial 1-to-1 matching. |
| **12**| Configuration Auditing | `get_config_fingerprint()` (`src/config.py`) | Deterministic MD5 hash verifying train and test configuration synchronization. |

---

## B. Evidence Flow Verification

We explicitly verified that blocking channel metadata generated during multi-key blocking survives all the way into the final feature matrix:
- **Columns Verified in Candidate Table:**
  `['blocked_exact_name', 'blocked_selective_token', 'blocked_rare_token', 'blocked_token_pair', 'blocked_char_ngram', 'shared_ngram_count', 'num_blocking_keys']`
- **Validation Feature Matrix Check:**
  `feature_table['num_blocking_keys']` was verified to be populated with real, non-zero values (total sum in validation set: 12,578 keys across 4,925 pairs).
- **Result:** Evidence is never regenerated as a placeholder; it flows directly from `ProductionBlockingEngine` through `candidate_safety` into `build_feature_table`.

---

## C. Candidate Safety Cap Verification

Tested with `max_candidates_per_s1 = 300` on 500 Source 1 records and 11,776 target records:
- **Candidate Pairs Before Cap:** 91,050
- **Candidate Pairs After Cap:** 76,372 (14,678 noisy candidate pairs safely eliminated)
- **Overflowing S1 Entities:** 117 entities had >300 candidates before capping
- **True Pairs Before Cap:** 1,619 (out of 1,778 ground truth true pairs)
- **True Pairs After Cap:** 1,619
- **Candidate-Cap Recall Retention:** **100.00%** (zero true positives lost by evidence ranking!)
- **Combined Final Candidate Recall:** **91.06%**

---

## D. Training Pair Construction & Hard Negatives

Evaluated with target negative ratio of 15:1:
- **Surviving Positive Pairs:** 1,619
- **Selected Negative Pairs:** 24,285
  - *Hard Negatives (Top 70% ranked by evidence score):* 16,999
  - *Random Negatives (Remaining 30% for diversity):* 7,286
- **Dynamic `scale_pos_weight`:** **15.00**
- Hard negatives were verified to be deterministically ordered using `compute_evidence_score()`.

---

## E. Entity-Level Split Verification

- **Train S1 Entities:** 400
- **Validation S1 Entities:** 100
- **Entity Overlap Check:** `len(train_s1.intersection(val_s1)) == 0` (Strictly verified)
- **Train Candidate Pairs:** 20,979 (Positives: 1,293)
- **Validation Candidate Pairs:** 4,925 (Positives: 326)

---

## F. Transductive Unlabeled TF-IDF

- **Module:** `src/tfidf_model.py` (`TransductiveTFIDF`)
- **Fitted on:** Unlabeled union of raw normalized names and addresses across train and test partitions.
- **Labels Passed:** Strictly None (no ground truth, no matched IDs, no target labels).
- **Vocabularies Fitted:** Name vocabulary: 10,000 terms, Address vocabulary: 10,000 terms.
- **Integration:** `src/build_features.py::fit_tfidf_models()` delegates to `TransductiveTFIDF`, ensuring a single authoritative implementation.

---

## G. Feature Integrity & Numerical Safety

- **Total Features:** 57 numerical features across Groups A–G + TF-IDF
- **Total NaNs:** 0
- **Total Infs:** 0
- **Non-Finite Values:** 0
- Directional missingness indicators (`s1_name_missing`, `candidate_address_missing`, etc.) properly represent missing fields without generating spurious 1.0 similarities.

---

## H. Central Configuration & Fingerprint

- **Configuration Module:** `src/config.py`
- **Initial Fingerprint:** `af143f61867ee2fcd7238f76ea4eb09f`
- **Sensitivity Test:** Modifying any parameter (e.g. `ngram_size` from 3 to 4) immediately changes the fingerprint (`11eb6aaf2e8e67da2388d1acd564bcf0`).
- Both train and test inference can verify this fingerprint at startup to prevent silent configuration drift.

---

## I. Diagnostic Collision Analysis

Ran on the post-cap candidate set (76,372 pairs across 500 S1 entities):
- **Unique S1 Entities:** 500
- **Unique Target Candidates:** 9,593
- **Candidates Matched to >1 S1:** 8,555 (89.2% of target candidates were retrieved as potential candidates for multiple S1 queries)
- **S2 Colliding Candidates:** 4,223
- **S3 Colliding Candidates:** 4,332
- **Distribution:** Confirmed purely diagnostic. No artificial 1-to-1 matching constraints or Hungarian algorithms were applied, honoring the many-to-many challenge requirement.

---

## J. Performance & Compute Budget

- **Total Dry-Run Wall Clock Time:** 141.7s
- **Peak Memory:** 335.5 MB (traced via Python tracemalloc)
- Operations are memory-safe and easily fit within typical hardware limits.

---

## K. Repository Duplicate Audit

| Component | Found Implementations | Resolution |
| :--- | :--- | :---: |
| **Blocking** | `src/blocking.py` (`ProductionBlockingEngine`), `src/test_blocking_scale.py` (`BlockingEngine`) | **MERGED:** `src/blocking.py` is the single authoritative production blocking engine; experimental script retained for benchmark history. |
| **Candidate Cap** | `src/candidate_safety.py` (`apply_candidate_safety_cap`) | **AUTHORITATIVE:** Single implementation. |
| **Training Pairs** | `src/training_pairs.py` (`construct_training_pairs`) | **AUTHORITATIVE:** Single implementation. |
| **Split** | `src/split.py` (`split_candidate_pairs`) | **AUTHORITATIVE:** Single implementation. |
| **TF-IDF** | `src/tfidf_model.py` (`TransductiveTFIDF`), `src/build_features.py` (`fit_tfidf_models`) | **ALIGNED:** `build_features.py` now delegates directly to `TransductiveTFIDF`. |
| **Collisions** | `src/collision.py` (`analyze_collisions`) | **AUTHORITATIVE:** Single implementation. |

---

## L. Issues Found & Remediated During Audit

1. **Issue:** Duplicate TF-IDF vectorizer logic in `build_features.py`.
   - **Severity:** Low.
   - **File:** `src/build_features.py`.
   - **Fix:** Replaced ad-hoc fitting in `build_features.py` to delegate to `TransductiveTFIDF`.
   - **Verification:** Unit tests and dry run verified identical interface.

2. **Issue:** Hard-negative sorting in `training_pairs.py` was sorting on tuple columns rather than composite evidence score.
   - **Severity:** Medium.
   - **File:** `src/training_pairs.py`.
   - **Fix:** Updated to use `compute_evidence_score()` with tie-breaking on `entity_id_s1` and `entity_id_cand`.
   - **Verification:** Determinism verified in `test_architecture.py`.

---

## M. Final Status

All unit tests pass (17/17 passed):
```bash
python -m unittest discover -s src -p "test_*.py" -> OK (17 tests)
python src/test_normalization.py -> Tests completed successfully.
python -m compileall src -> Clean compilation, 0 errors.
```

> **PHASE 4 ARCHITECTURE INTEGRATION VERIFIED — READY FOR PHASE 4B**
