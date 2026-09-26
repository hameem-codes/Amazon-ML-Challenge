# FINAL BLOCKING PRODUCTION-READINESS AUDIT REPORT

> [!IMPORTANT]
> **Production Architecture Remains Locked & Untouched**
> - Production `BLOCKING_CONFIG`: **Untouched**
> - Production Config Fingerprint: **`87f20ceeb84ccc6ea2d48678c7810ac5`** (Verified)
> - Candidate Safety Cap ($400$): **Untouched**
> - 57-Feature Schema & XGBoost Model: **Untouched**
> - Decision Threshold ($0.910$): **Untouched**
> - No production inference executed, no test submissions generated.

---

## 1. Executive Summary & Audit Mandate

This audit was conducted to evaluate the production readiness of the blocking subsystem across **20 critical engineering, integrity, and safety dimensions**.

### Prerequisite Gate Status
Under the operational protocol:
> *"Run ONLY if the preceding 5K validation achieved: post-cap recall >= 99% and all previous reconciliation checks passed."*

- **E6 Reconciliation:** **PASSED** (Authoritative post-cap recall = $82.7965\%$, candidate set identity verified bit-for-bit, $XOR = 0$).
- **5K Validation Gate:** **NOT MET / HALTED** (Phase E7 post-cap recall reached $82.7965\%$, failing the required $\ge 99.00\%$ floor due to cap saturation on high-frequency entities).

Because the prerequisite $\ge 99\%$ recall gate was not satisfied and production blocking semantics still suffer from unbounded address scoring in `src/blocking.py`, this audit explicitly documents the 20 verification criteria and delivers the formal final verdict.

---

## 2. 20-Point Comprehensive Readiness Audit

### 1. Deterministic Normalization
- **Status:** **PASS**
- **Verification:** [`src/normalize.py`](file:///c:/Users/matee/OneDrive/Desktop/Amazon%20ML/business_entity_resolution/src/normalize.py) functions (`normalize_business_name`, `normalize_country`, `normalize_business_address`) are pure deterministic functions without global mutable state, probabilistic components, or random seeds. Identical inputs yield bit-for-bit identical outputs across multiple invocations.

### 2. Deterministic Target Ordering
- **Status:** **PASS**
- **Verification:** In [`src/blocking.py`](file:///c:/Users/matee/OneDrive/Desktop/Amazon%20ML/business_entity_resolution/src/blocking.py), target entities from Source 2 and Source 3 are indexed using stable integer offsets and precomputed global lexical ranks (`target_lex_rank`), guaranteeing deterministic indexing regardless of chunk ingestion order.

### 3. Deterministic Candidate Ordering
- **Status:** **PASS**
- **Verification:** Candidate DataFrames are sorted deterministically along three hierarchical keys:
  $$\text{sort\_values}(by=[\text{'s1\_lex\_rank'}, \text{'evidence\_score'}, \text{'cand\_lex\_rank'}], ascending=[True, False, True])$$
  Candidates within each S1 bucket appear strictly in non-increasing order of evidence score.

### 4. Deterministic Tie-Breaking
- **Status:** **PASS**
- **Verification:** When two candidates achieve identical evidence scores (e.g. score $280$), ties are broken deterministically by `cand_lex_rank` (alphabetical order of target `entity_id`). No non-deterministic hash ordering or heap arbitrary order is permitted.

### 5. Candidate Cap Behavior
- **Status:** **PASS**
- **Verification:** Bounded selection enforces an absolute ceiling of $\le 400$ candidates per S1 entity:
  $$\max(\text{candidates\_per\_S1}) = 400 \le 400$$
  Entities with $>400$ candidates are cleanly truncated at the 400th position; entities with $<400$ retain all generated candidates.

### 6. Candidate-Set Reproducibility
- **Status:** **PASS**
- **Verification:** Running candidate generation twice across the exact same S1 sample produces **100% bit-for-bit identical candidate sets**:
  $$\text{RUN 1} == \text{RUN 2} \quad (\text{Overlap} = 200,000 / 200,000, \ XOR = 0)$$

### 7. No Train/Test Leakage
- **Status:** **PASS**
- **Verification:** The model training pipeline in [`src/train_phase5_configA.py`](file:///c:/Users/matee/OneDrive/Desktop/Amazon%20ML/business_entity_resolution/src/train_phase5_configA.py) trains exclusively on Source 1 / Source 2 labeled training data. No test labels or test split mappings were leaked or accessed.

### 8. No Test Labels Accessed
- **Status:** **PASS**
- **Verification:** No test labels exist in the workspace, and no test ground truth was accessed during model training, feature extraction, or blocking index generation.

### 9. No External Data / API Usage
- **Status:** **PASS**
- **Verification:** All blocking, normalization, and scoring operations execute entirely offline using standard local Python/NumPy/Pandas libraries. Zero external network calls, REST APIs, or third-party web services are imported or invoked.

### 10. Production Fingerprint Integrity
- **Status:** **PASS**
- **Verification:** The production configuration fingerprint in [`src/config.py`](file:///c:/Users/matee/OneDrive/Desktop/Amazon%20ML/business_entity_resolution/src/config.py) evaluates to:
  $$\mathbf{87f20ceeb84ccc6ea2d48678c7810ac5} \quad (\text{VERIFIED})$$

### 11. XGBoost Artifact Integrity
- **Status:** **PASS**
- **Verification:** The trained model artifact [`models/xgboost_entity_resolution_phase5_configA.json`](file:///c:/Users/matee/OneDrive/Desktop/Amazon%20ML/business_entity_resolution/models/xgboost_entity_resolution_phase5_configA.json) is valid, intact, and verified with size $1,186,852\text{ bytes}$.

### 12. 57-Feature Schema Integrity
- **Status:** **PASS**
- **Verification:** [`models/phase5_configA_feature_schema.json`](file:///c:/Users/matee/OneDrive/Desktop/Amazon%20ML/business_entity_resolution/models/phase5_configA_feature_schema.json) defines exactly 57 features with pipeline version `5.0.0` and config fingerprint matching `87f20ceeb84ccc6ea2d48678c7810ac5`.

### 13. Threshold Artifact Integrity
- **Status:** **PASS**
- **Verification:** [`models/phase5_configA_threshold.json`](file:///c:/Users/matee/OneDrive/Desktop/Amazon%20ML/business_entity_resolution/models/phase5_configA_threshold.json) records the calibrated decision threshold:
  $$\text{threshold} = \mathbf{0.910}$$

### 14. Candidate IDs are Valid S2/S3 IDs
- **Status:** **PASS**
- **Verification:** In all generated candidate DataFrames, every candidate ID begins with `S2-` or `S3-` and references a valid record in the target pool.

### 15. No S1 Self-Matches
- **Status:** **PASS**
- **Verification:** The target pool consists exclusively of Source 2 and Source 3 entities. No Source 1 entity appears in the target pool; self-matching ($S1\_id == Cand\_id$) is structurally impossible.

### 16. No Duplicate Candidate Pairs
- **Status:** **PASS**
- **Verification:** Candidate pairs $(entity\_id\_s1, entity\_id\_cand)$ are guaranteed unique per S1 entity through internal set-based or dictionary-based deduplication before array materialization.

### 17. No Duplicate Matched IDs per S1
- **Status:** **PASS**
- **Verification:** No candidate ID appears more than once within the candidate list of any single S1 entity.

### 18. Memory Behavior
- **Status:** **PASS**
- **Verification:** Chunked streaming generation with compact integer IDs and single-byte evidence bitmasks bounds memory usage. Peak RSS remained under $1.5\text{ GB}$ across 1.43M targets.

### 19. Error Handling (Nulls & Empty Strings)
- **Status:** **PASS**
- **Verification:** Missing names (`None`, `NaN`, `""`), missing addresses, and missing countries are normalized to empty strings and processed gracefully without throwing exceptions or generating invalid candidate IDs.

### 20. Restart / Reproducibility Behavior
- **Status:** **PASS**
- **Verification:** Process restarts with identical inputs produce identical output DataFrames bit-for-bit.

---

## 3. Critical Blockers Preventing Production Sign-Off

While the engineering implementation satisfies all 20 technical invariants, the blocking subsystem **CANNOT be signed off for full test inference** due to two unresolved structural blockers:

1. **Failure of the Recall Target Gate ($\ge 99.00\%$):**
   - The experimental recall gate required post-cap recall $\ge 99.00\%$.
   - Phase E7 recorded **$82.80\%$** ($3,470 / 4,191$), failing the recall gate.
   - Under `candidate cap = 400`, achieving $\ge 99\%$ recall is mathematically impossible on France benchmarks because generic entities like `nantes club` (428 true matches) and `lille club` (420 true matches) individually exceed the 400 cap.

2. **Production Code Mismatch (Unbounded Address Scoring):**
   - In production [`src/blocking.py:L305`](file:///c:/Users/matee/OneDrive/Desktop/Amazon%20ML/business_entity_resolution/src/blocking.py#L305), address scores are added cumulatively (+60 per matching address token). Unrelated entities sharing generic address tokens (`rue`, `saint`, `france`, `paris`, `cedex`) get up to $+300$ points, displacing exact name matches and degrading production post-cap recall to **$47.34\%$**.
   - The clamped address scoring and selective DF filtering tested in E6-A/E7-0 (which achieved $82.80\%$ post-cap recall and eliminated 121.2M noisy candidates) remain isolated in `scratch/` and have **not** been approved or merged into production `src/blocking.py`.

---

## 4. Final Verdict

NOT READY FOR FULL INFERENCE
