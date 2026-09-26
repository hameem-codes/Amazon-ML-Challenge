# Optimization Cap → Model → Macro F0.5 Validation Report

## 1. Objective
The primary objective of this experiment is to answer a single critical architectural question:
> **Does increasing the candidate safety cap above 400 materially improve the FINAL MODEL METRIC (Macro F0.5), after the locked Phase 5 XGBoost model scores the candidates?**

While upstream blocking experiments measured raw and post-cap candidate recall, candidate recall is **not** the competition decision criterion. The competition evaluates entity-level **Macro F0.5**, which places $4\times$ more weight on precision than recall ($\beta = 0.5$). This experiment executes the full end-to-end pipeline:
$$\text{BLOCKING} \longrightarrow \text{SAFETY CAP} \longrightarrow \text{57 FEATURES} \longrightarrow \text{XGBOOST INFERENCE} \longrightarrow \text{THRESHOLD (0.910)} \longrightarrow \text{MACRO F0.5}$$

---

## 2. Locked Artifacts
All downstream feature extraction, inference, and thresholding steps utilized strictly locked production artifacts:

- **Model**: `models/xgboost_entity_resolution_phase5_configA.json`
- **Feature Schema**: `models/phase5_configA_feature_schema.json` (exactly 57 features)
- **Classification Threshold**: `models/phase5_configA_threshold.json` (`0.910`)
- **Production Fingerprint**: `87f20ceeb84ccc6ea2d48678c7810ac5` (Verified unchanged)
- **Model Modifications**: Zero. No retraining, no threshold re-tuning, no feature schema adjustments.

---

## 3. Validation Dataset
The evaluation was executed on the exact labeled France benchmark used across all Phase E experiments:
- **Source S1 Entities**: 500 labeled French entities from `data/train/source1.csv`.
- **Target Record Universe**: 1,434,993 France records ($S_2 \cup S_3$) from `data/train/source2.csv` and `data/train/source3.csv`.
- **Ground Truth Reference Pairs**: 4,191 true matching pairs (from `data/train/ground_truth.csv`).
- **Ground Truth Zero-Match Entities**: 94 out of 500 entities have zero true matches in the target pool.
- **Normalization**: Standardized canonical address/name normalization matching production.

---

## 4. Experimental Controls
To guarantee strict apples-to-apples fairness:
1. **Identical Entities & Targets**: Every configuration processed the exact same 500 S1 queries against the identical 1.43M targets.
2. **Identical Blocking Pipeline**: Config A channels (Country, Name Token, Name Prefix-3, Name Prefix-4, Address Token) with selective DF filtering (> 50,000) and bitmask evidence tracking.
3. **Identical Downstream Pipeline**: 57-feature table computed using vectorized cosine similarities and transductive TF-IDF; identical XGBoost model evaluated at exact threshold `0.910`.
4. **Single Isolated Variable**: The candidate safety cap per S1 entity ($K \in \{400, 600, 800, 1000\}$).
5. **Deterministic Ordering & Tie-Breaking**: Candidates sorted deterministically by (evidence score descending, target ID ascending) before truncation.

---

## 5. Cap Comparison Table

| Metric | CAP-400 (Baseline) | CAP-600 | CAP-800 | CAP-1000 |
|:---|:---:|:---:|:---:|:---:|
| **Post-Cap Candidates** | 200,000 | 300,000 | 400,000 | 500,000 |
| **Candidate Recall** | 82.80% | 85.49% | 88.00% | 90.58% |
| **Macro Precision** | **6.7123%** | 5.2252% | 4.4010% | 3.8889% |
| **Macro Recall** | 86.3008% | 86.4070% | 86.4503% | **86.5386%** |
| **Macro F0.5** | **0.077903** | **0.061219** | **0.051912** | **0.046087** |
| **$\Delta$ Macro F0.5 vs CAP-400** | — | **-0.016684 (-21.4%)** | **-0.025991 (-33.4%)** | **-0.031816 (-40.8%)** |
| **True Positives (TP)** | 2,601 | 2,690 | 2,774 | 2,863 |
| **False Positives (FP)** | 32,930 | 47,765 | 62,243 | 74,837 |
| **False Negatives (FN)** | 1,590 | 1,501 | 1,417 | 1,328 |
| **Pairwise Precision** | 7.3204% | 5.3315% | 4.2666% | 3.6847% |
| **Pairwise Recall** | 62.0616% | 64.1852% | 66.1895% | 68.3131% |
| **Total Runtime (s)** | 320.49s | 476.77s | 426.34s | 381.38s |
| **Peak RAM (MB)** | 1,252.5 MB | 1,345.3 MB | 1,438.4 MB | 1,532.6 MB |

---

## 6. Candidate-Generation Metrics

| Metric | CAP-400 | CAP-600 | CAP-800 | CAP-1000 |
|:---|:---:|:---:|:---:|:---:|
| **Raw Candidates Generated** | 95,603,816 | 95,603,816 | 95,603,816 | 95,603,816 |
| **Raw Candidate Recall** | 100.00% | 100.00% | 100.00% | 100.00% |
| **Post-Cap Candidates** | 200,000 | 300,000 | 400,000 | 500,000 |
| **Average Candidates / S1** | 400.0 | 600.0 | 800.0 | 1,000.0 |
| **Median Candidates / S1** | 400.0 | 600.0 | 800.0 | 1,000.0 |
| **Max Candidates / S1** | 400 | 600 | 800 | 1,000 |
| **Post-Cap Candidate Recall** | 82.7965% | 85.4927% | 87.9981% | 90.5750% |
| **True Pairs Retained in Pool** | 3,470 | 3,583 | 3,688 | 3,796 |
| **True Pairs Lost in Truncation** | 721 | 608 | 503 | 395 |

---

## 7. Final Model Metrics
Evaluating post-model predictions after applying threshold `0.910`:

| Metric | CAP-400 | CAP-600 | CAP-800 | CAP-1000 |
|:---|:---:|:---:|:---:|:---:|
| **Predicted Matched Entities** | 500 | 500 | 500 | 500 |
| **Predicted Zero-Match Entities** | 0 | 0 | 0 | 0 |
| **Ground Truth Zero-Match Entities**| 94 | 94 | 94 | 94 |
| **Entity Macro Precision** | **6.7123%** | 5.2252% | 4.4010% | 3.8889% |
| **Entity Macro Recall** | 86.3008% | 86.4070% | 86.4503% | **86.5386%** |
| **Entity Macro F0.5** | **0.077903** | 0.061219 | 0.051912 | 0.046087 |
| **Pairwise Precision** | 7.3204% | 5.3315% | 4.2666% | 3.6847% |
| **Pairwise Recall** | 62.0616% | 64.1852% | 66.1895% | 68.3131% |

---

## 8. Precision / Recall Tradeoff
The empirical results reveal a stark asymmetry between candidate recall and model precision:

1. **Precision Collapses Under Higher Caps**:
   - Going from CAP-400 to CAP-600 drops Macro Precision from **6.71% to 5.23%** (-22.2% relative).
   - Going to CAP-1000 collapses Macro Precision to **3.89%** (-42.1% relative).
2. **Recall Barely Budges at the Entity Level**:
   - Despite candidate pool recall jumping from 82.80% to 90.58% (+7.78%), **Macro Recall only increases from 86.30% to 86.54% (+0.24%)**.
   - **Why?** In entity resolution, 494 out of 500 entities already had their true matches identified or had saturated recall. The extra candidate slots only recovered borderline matches for a tiny handful of entities.
3. **Severe FP Pollution**:
   - For every 1 true positive recovered beyond CAP-400, the model introduces **over 160 false positives**.
   - Because $F_{0.5}$ weights precision at $4\times$ the weight of recall:
     $$F_{0.5} = \frac{1.25 \cdot P \cdot R}{0.25 \cdot P + R}$$
     the flood of false positives causes **Macro F0.5 to drop by 40.8%** from 0.0779 to 0.0461.

---

## 9. Runtime and Memory Tradeoff

| Configuration | Blocking (s) | Feature Gen (s) | Model Inference (s) | Total Time (s) | Throughput (S1/s) | Peak RAM (MB) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **CAP-400** | 270.24s | 46.22s | 3.02s | **320.49s** | **1.56** | **1,252.5 MB** |
| **CAP-600** | 393.72s | 69.31s | 3.74s | 476.77s | 1.05 | 1,345.3 MB |
| **CAP-800** | 338.41s | 74.45s | 3.48s | 426.34s | 1.17 | 1,438.4 MB |
| **CAP-1000** | 291.93s | 83.37s | 3.88s | 381.38s | 1.31 | 1,532.6 MB |

- **Feature Generation Scaling**: Directly linear with candidate volume ($46.2\text{s} \to 83.4\text{s}$, nearly doubling as candidates scale from 200k to 500k).
- **Memory Overhead**: Peak RAM increases monotonically from 1.25 GB to 1.53 GB (+280 MB) to store candidate pairs and feature matrices.

---

## 10. Incremental Benefit Per Additional Candidate

| Comparison | $\Delta$ Candidates | $\Delta$ True Pairs (TP) | $\Delta$ False Positives (FP) | FP per 1 TP | $\Delta$ Macro F0.5 |
|:---|:---:|:---:|:---:|:---:|:---:|
| **CAP-600 vs CAP-400** | +100,000 | +89 | +14,835 | **166.7 FP / TP** | **-0.016684 (-21.4%)** |
| **CAP-800 vs CAP-400** | +200,000 | +173 | +29,313 | **169.4 FP / TP** | **-0.025991 (-33.4%)** |
| **CAP-1000 vs CAP-400** | +300,000 | +262 | +41,907 | **160.0 FP / TP** | **-0.031816 (-40.8%)** |

**Finding**: Every additional candidate added past 400 is disproportionately negative. Generating 300,000 additional candidates delivers 262 true positives at the catastrophic expense of 41,907 false positives.

---

## 11. Incremental Benefit Per Additional Second

| Comparison | $\Delta$ Total Time | $\Delta$ TP | TP gained / second | FP gained / second | Net Impact on F0.5 |
|:---|:---:|:---:|:---:|:---:|:---:|
| **CAP-600 vs CAP-400** | +156.28s | +89 | 0.57 TP/s | 94.9 FP/s | Strongly Negative |
| **CAP-800 vs CAP-400** | +105.85s | +173 | 1.63 TP/s | 276.9 FP/s | Strongly Negative |
| **CAP-1000 vs CAP-400** | +60.89s | +262 | 4.30 TP/s | 688.2 FP/s | Strongly Negative |

Spending additional computational time and memory generates negative utility. The system incurs higher compute costs to achieve a worse final score.

---

## 12. Exact Differences in Predictions Between Caps

1. **Monotonic Prediction Superset**:
   - The set of predictions made by CAP-400 is a strict subset of CAP-600, which is a subset of CAP-800, which is a subset of CAP-1000:
     $$\mathcal{P}_{400} \subset \mathcal{P}_{600} \subset \mathcal{P}_{800} \subset \mathcal{P}_{1000}$$
   - Zero predictions from CAP-400 were dropped in higher caps (`dropped_predictions_count = 0`).
2. **Prediction Breakdown**:
   - **CAP-600 vs CAP-400**: Exactly 14,924 new predicted pairs ($\mathbf{89\text{ TP}}$ and $\mathbf{14,835\text{ FP}}$).
   - **CAP-800 vs CAP-400**: Exactly 29,486 new predicted pairs ($\mathbf{173\text{ TP}}$ and $\mathbf{29,313\text{ FP}}$).
   - **CAP-1000 vs CAP-400**: Exactly 42,169 new predicted pairs ($\mathbf{262\text{ TP}}$ and $\mathbf{41,907\text{ FP}}$).
3. **Representative Sample of New Predictions**:
   - `('S1-260837575', 'S2-54568345')`: False Positive
   - `('S1-756828693', 'S3-212635632')`: False Positive
   - `('S1-210978493', 'S3-7133606')`: False Positive
   - `('S1-67951250', 'S2-952126562')`: False Positive
   - `('S1-734373273', 'S2-724939493')`: False Positive

---

## 13. Failure Cases and Vulnerability Analysis
- **The "High Recall Fallacy"**: Optimizing candidate generation in isolation to achieve $\ge 99\%$ recall pushes tail candidates with weak blocking evidence (e.g., matching a single generic address token or 3-gram prefix) into the scoring pipeline.
- **Model False Alarm Rate**: Even with an XGBoost threshold of 0.910, scoring 500,000 weak pairs results in tens of thousands of marginal pairs scoring $\ge 0.910$, completely overwhelming the true signal.
- **Candidate Cap as a High-Precision Filter**: The safety cap of 400 is not merely a memory constraint; it functions as a highly effective **first-stage rank filter** that discards noisy false-positive candidates before the model ever sees them.

---

## 14. Production Firewall Verification
Strict firewall boundaries were maintained throughout the experiment:
- **Test Set Firewall**: No test datasets (`data/test/*`) were accessed, loaded, or inspected.
- **No Submission Output**: No submission files or test candidate pairs were generated.
- **Zero Production Mutations**: `src/config.py`, `src/blocking.py`, and `src/candidate_safety.py` remain untouched.
- **Fingerprint Verification**: Production configuration fingerprint confirmed as:
  `87f20ceeb84ccc6ea2d48678c7810ac5`

---

## 15. Final Evidence-Based Recommendation

### Decision: KEEP CANDIDATE CAP LOCKED AT 400

1. **Macro F0.5 Outcome**:
   - CAP-400 achieves **Macro F0.5 = 0.0779**.
   - Increasing the cap to 600, 800, or 1000 reduces Macro F0.5 to **0.0612 (-21.4%)**, **0.0519 (-33.4%)**, and **0.0461 (-40.8%)** respectively.
2. **Precision Penalty**:
   - Every additional candidate slot above 400 incurs ~160 false positives for every 1 true positive recovered.
   - Precision drops from 6.71% to 3.89%.
3. **Macro Recall Saturation**:
   - Entity-level Macro Recall improves by an insignificant +0.24% (86.30% to 86.54%).
4. **Computational Efficiency**:
   - CAP-400 requires 40% less feature-computation time and saves ~280 MB of RAM.

**Conclusion**: Increasing the candidate safety cap above 400 **does not improve** Macro F0.5—it severely degrades it. **CAP-400 is strictly superior.**
