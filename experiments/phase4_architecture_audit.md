# Phase 4 Architecture Audit & Alignment

This document audits all current modules, scripts, and components against the final production architecture specification for the Amazon ML Challenge 2026 Business Entity Resolution project.

## Current Pipeline vs. Final Architecture

| Component / File | Current State | Final Architecture Requirement | Status | Rationale |
| :--- | :--- | :--- | :---: | :--- |
| **`config.py`** | Does not exist; magic numbers scattered in test files. | Central configuration file containing all thresholds, caps, seeds, and versioning fingerprint. | **ADD** | Mandatory for reproducibility and train/test drift detection. |
| **`src/normalize.py`** | Conservative Unicode-safe normalization preserving letters, marks, digits. | Unicode NFKC, casefolding, punctuation replacement, whitespace collapse. No aggressive stripping. | **KEEP** | Completely aligned with Phase 2 specification; passes all tests. |
| **`src/blocking.py`** | Basic naive blocking functions (exact, country, token, char-ngram) using pandas merges. | Production multi-key blocking engine with inverted indexes, streaming, evidence preservation, and safety caps. | **MODIFY** | Upgrade to export the production `BlockingEngine` and candidate generator, preserving blocking evidence metadata per channel. |
| **`src/candidate_safety.py`** | Logic for candidate capping and evidence ranking is missing. | Rank overflowing candidates by blocking evidence (exact match, num keys, shared n-grams, tokens) up to `max_candidates_per_s1`. Measure recall before/after cap. | **ADD** | Mandatory safety mechanism against candidate explosion. |
| **`src/training_pairs.py`** | Does not exist; training pairs currently just take all candidate pairs. | Construct training pairs: retain all positives, select hard negatives (high similarity, shared blocking keys, same country), deterministic sampling. | **ADD** | Prevents trivial negatives from dominating the ~114:1 imbalance. |
| **`src/split.py`** | Does not exist. | Entity-level split on Source 1 ID. Never split individual candidate pairs across train/validation. | **ADD** | Mandatory to prevent data leakage from repeated S1 reference entities. |
| **`src/features.py`** | 57 features across Groups A-G + TF-IDF. | Name, address, country (open-set), directional missingness, structural, blocking evidence, source indicators. | **KEEP / MODIFY** | Retain all 57 features; ensure blocking evidence flags reflect actual channels from candidate generator. |
| **`src/tfidf.py` / TF-IDF logic | TF-IDF is currently fit only on training sample names/addresses. | Fit TF-IDF vocabulary on UNLABELED union of raw train + test text across S1, S2, S3 to prevent OOV drift (e.g. France in test). | **MODIFY** | Extract into dedicated transductive unlabelled helper and ensure consistent vectorizer persistence. |
| **`src/build_features.py`** | Feature matrix creation with ground-truth label separation. | Ingestion of candidate pairs, feature extraction, clean table output without label leakage. | **KEEP / MODIFY** | Align with updated blocking metadata format and candidate cap workflow. |
| **`src/collision.py`** | Collision analysis logic is absent. | Diagnostic tool to measure multi-S1 matches per S2/S3 entity and assess false merge risks. | **ADD** | Required diagnostic for many-to-many match structures without imposing artificial 1-to-1 constraints. |
| **`src/test_blocking_scale.py`** | Inverted-index blocking stress test script. | Standalone experimental script. | **KEEP** | Preserved for historical/experimental reference. |
| **`src/test_blocking_refined.py`** | Phase 3B benchmark script. | Standalone experimental script. | **KEEP** | Preserved for historical benchmark reference. |
| **`src/test_feature_sanity.py`** | 10 unit tests for feature edge cases. | Sanity tests for feature computation. | **KEEP** | Passes all 10 tests. |
| **`src/test_architecture.py`** | Does not exist. | Tests for candidate cap, evidence ranking, entity split, hard negative sampling, and config hashing. | **ADD** | Validates new architectural components. |

## Actions Plan
1. Create `src/config.py` centralizing all parameters, thresholds, and configuration hashing.
2. Upgrade `src/blocking.py` to house the production Inverted-Index Blocking Engine that tracks per-channel evidence (`blocked_exact_name`, `blocked_selective_token`, `blocked_rare_token`, `blocked_token_pair`, `blocked_char_ngram`, `shared_ngram_count`, `num_blocking_keys`).
3. Create `src/candidate_safety.py` implementing deterministic evidence-based ranking and candidate safety capping per Source 1 entity.
4. Create `src/training_pairs.py` to construct positive and hard-negative pairs with deterministic sampling.
5. Create `src/split.py` to enforce strict Source 1 entity-level train/validation splitting.
6. Create `src/collision.py` for many-to-many collision diagnostics.
7. Create `src/tfidf_model.py` for transductive unlabeled TF-IDF vectorizer fitting across train and test.
8. Update `src/features.py` and `src/build_features.py` to integrate seamlessly with the evidence-preserving blocking and safety cap pipeline.
9. Implement comprehensive unit tests in `src/test_architecture.py` and run full test suites.
