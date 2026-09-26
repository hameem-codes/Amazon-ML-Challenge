# Phase 7 — Composite Pre-Ranker @ Fixed Cap 400 Validation Report

## 1. Executive Summary & Core Objective
The objective of Phase 7 is to evaluate whether a multi-component **Composite Pre-Ranker** can improve the competition decision metric (**Macro F0.5**) when the candidate safety cap is strictly fixed at **400 candidates per S1 entity**.

While previous Phase E experiments focused on raw blocking and frequency filtering, this experiment tests whether re-ranking the exact same raw candidate pool (95.6M postings across 500 France S1 queries) using:
$$\text{composite\_score} = 0.50 \cdot \text{name\_score} + 0.20 \cdot \text{address\_score} + 0.15 \cdot \text{rarity\_score} + 0.10 \cdot \text{country\_score} + 0.05 \cdot \text{structural\_score}$$
can improve downstream model discrimination over the legacy integer evidence ranker.

### Empirical Verdict
**REJECT COMPOSITE PRE-RANKER. RETAIN LEGACY CAP-400 EVIDENCE RANKER.**
- Although candidate-level recall slightly increased from **82.80% to 83.63%** (+35 true pairs captured in candidate slots), downstream model precision **collapsed from 6.71% to 3.23%** (-51.9% relative).
- False positives more than doubled from **32,930 to 75,428 (+42,498 FP)**, while true positives only rose from 2,601 to 2,680 (+79 TP).
- As a direct consequence of the $4\times$ weighting on precision in $F_{0.5}$, **Macro F0.5 plummeted by -51.1% from 0.077903 to 0.038058**.

---

## 2. Locked Artifacts & Contracts
All downstream components and model artifacts were locked and executed without modification:

- **Model**: `models/xgboost_entity_resolution_phase5_configA.json` (Phase 5 locked model)
- **Feature Schema**: `models/phase5_configA_feature_schema.json` (exactly 57 features)
- **Classification Threshold**: `models/phase5_configA_threshold.json` (`0.910`)
- **Production Fingerprint**: `87f20ceeb84ccc6ea2d48678c7810ac5` (Verified unchanged)
- **Validation Population**: 500 France S1 entities evaluated against 1,434,993 France targets ($S_2 \cup S_3$) with 4,191 reference true pairs.
- **Firewall Integrity**: No test datasets or labels loaded; no submission files generated.

---

## 3. Experimental Setup & Controls
To ensure an exact apples-to-apples comparison:
1. **Identical Raw Candidate Pool**: The identical Config A inverted-index channels with selective DF cap 50,000 generated 95,603,816 postings across the 500 S1 queries with 100% raw recall.
2. **Single Isolated Variable**: Candidate ranking before applying cap 400:
   - **Baseline**: Legacy integer evidence ranker (Channel scores: Name Token 70, Address Token 60, Prefix-4 50, Prefix-3 30, ties broken by `target_lex_rank` ASC).
   - **Phase 7 Experimental**: Composite pre-ranker normalizing components to $[0, 1]$:
     - $\text{name\_score} \in [0, 1]$: Exact match (1.0) or weighted token/prefix overlap.
     - $\text{address\_score} \in [0, 1]$: Bounded non-cumulative token overlap ratio.
     - $\text{rarity\_score} \in [0, 1]$: Maximum token rarity ($1.0 - \frac{\log(\text{DF})}{\log(N)}$).
     - $\text{country\_score} \in [0, 1]$: Exact country match (1.0).
     - $\text{structural\_score} \in [0, 1]$: Length compatibility ratio.
     - Ordering: `composite_score` DESC, ties broken by `target_lex_rank` ASC.
3. **Identical Downstream Inference**: Identical transductive TF-IDF embeddings, identical 57-feature table generation in 25,000-pair batches, and identical XGBoost inference at threshold 0.910.

---

## 4. Primary Results: Comparison Table

| Metric | Baseline CAP-400 (Legacy Ranker) | Composite Pre-Ranker @ Cap 400 | Incremental Change ($\Delta$) | Relative Change (%) |
|:---|:---:|:---:|:---:|:---:|
| **Candidate Cap** | 400 | 400 | 0 | 0.0% |
| **Raw Candidate Count** | 95,603,816 | 95,603,816 | 0 | 0.0% |
| **Raw Candidate Recall** | 100.00% | 100.00% | 0.0% | 0.0% |
| **Post-Ranker Candidates** | 200,000 | 200,000 | 0 | 0.0% |
| **Post-Ranker Recall** | 82.80% (82.7965%) | **83.63% (83.6316%)** | **+0.83%** | +1.0% |
| **True Pairs Retained** | 3,470 | **3,505** | **+35** | +1.0% |
| **True Pairs Lost** | 721 | **686** | **-35** | -4.9% |
| **Entity Macro Precision** | **6.7123%** | 3.2298% | **-3.4825%** | **-51.9%** |
| **Entity Macro Recall** | **86.3008%** | 86.0956% | -0.2052% | -0.2% |
| **Entity Macro F0.5** | **0.077903** | 0.038058 | **-0.039845** | **-51.1%** |
| **True Positives (TP)** | 2,601 | 2,680 | **+79** | +3.0% |
| **False Positives (FP)** | **32,930** | 75,428 | **+42,498** | **+129.1%** |
| **False Negatives (FN)** | 1,590 | 1,511 | -79 | -5.0% |
| **Pairwise Precision** | **7.3204%** | 3.4311% | -3.8893% | -53.1% |
| **Pairwise Recall** | 62.0616% | 63.9466% | +1.8850% | +3.0% |
| **Pairwise F0.5** | **0.088884** | 0.042322 | -0.046562 | -52.4% |
| **Matched S1 Entities** | 500 | 500 | 0 | 0.0% |
| **Zero-Match S1 Entities**| 0 | 0 | 0 | 0.0% |
| **Blocking + Ranking Time** | 270.24s | 921.45s | +651.21s | +240.9% |
| **Total Pipeline Runtime** | **320.49s** | 972.70s | +652.21s | +203.5% |
| **Peak RAM (MB)** | **1,252.5 MB** | 1,264.5 MB | +12.0 MB | +1.0% |

---

## 5. In-Depth Metric Analysis

### 5.1 The Candidate Recall Illusion
At the candidate generation stage, the Composite Pre-Ranker appeared to succeed:
- True pairs retained inside the 400 candidate slots increased from **3,470 to 3,505 (+35 pairs)**.
- Candidate pool recall improved from **82.80% to 83.63% (+0.83%)**.

### 5.2 The Downstream Collapse
However, scoring the candidates with the locked Phase 5 XGBoost model completely reversed this advantage:
- **Massive False Positive Inflation**: The number of predicted pairs scoring $\ge 0.910$ exploded from **35,531 to 78,108 (+42,577 total predictions)**.
- **FP / TP Asymmetry**: For the 79 true positive pairs recovered, the model accepted **42,498 additional false positives**—an alarming ratio of **538 False Positives for every 1 True Positive recovered**.
- **Macro Precision Collapse**: Entity Macro Precision was cut in half, dropping from **6.71% to 3.23%**.
- **Macro Recall Stagnation**: Entity Macro Recall did not improve; it slightly decreased from **86.30% to 86.10% (-0.21%)**.
- **Macro F0.5 Penalty**: Because $F_{0.5}$ weights precision at $4\times$ the weight of recall:
  $$F_{0.5} = \frac{1.25 \cdot P \cdot R}{0.25 \cdot P + R}$$
  the precision collapse drove **Macro F0.5 down by -51.1% (0.0779 $\to$ 0.0381)**.

---

## 6. Exact Prediction Differences Versus Baseline

An exact set difference between the final predicted pairs of Baseline CAP-400 and Composite Pre-Ranker CAP-400 reveals:

```
Total Predictions (Baseline CAP-400):       35,531  (2,601 TP + 32,930 FP)
Total Predictions (Composite Pre-Ranker):   78,108  (2,680 TP + 75,428 FP)

New Predictions (Composite - Baseline):     50,601  (  224 TP + 50,377 FP) -> 99.56% False Positives
Dropped Predictions (Baseline - Composite):  8,024  (  145 TP +  7,879 FP) -> 98.19% False Positives
Net Change:                                +42,577  (  +79 TP + 42,498 FP)
```

### Analysis of New Predictions:
The composite pre-ranker favored candidates with high token rarity (low document frequency) and matched string lengths. In practice, many unrelated business entities in France share a rare surname or unique localized street token while having completely different business types. Because the XGBoost model assigns substantial weight to TF-IDF cosine similarity and rare token flags, presenting these rare-token candidates to the model caused the model to cross the 0.910 decision boundary en masse, producing 50,377 new false positive errors.

---

## 7. Computational Runtime & Memory Tradeoff
- **Candidate Generation Overhead**: The composite scoring function calculates floating-point scores across all candidate postings in memory. This increased blocking runtime from **270s to 921s** (+3.4x slower).
- **Throughput**: Overall throughput fell from **1.56 S1/sec to 0.51 S1/sec**.
- **Memory Footprint**: Peak RAM remained modest at 1,264.5 MB (+12 MB vs baseline).

---

## 8. Failure Cases and Root Cause Analysis
1. **Adversarial Candidate Selection**: The composite pre-ranker actively selected "hard negative" candidates that possess high token rarity and similar string lengths. These are precisely the candidates that are most likely to deceive the downstream classifier into false alarms.
2. **Channel Correlation in Legacy Ranker**: The legacy integer evidence score heavily prioritized candidates matching multiple distinct channels (Country + Name Token + Prefix-4 + Address Token = 210 points). These multi-channel candidates provide robust corroborating evidence that XGBoost correctly classifies with high precision.
3. **Decoupled Optimization Danger**: Optimizing upstream candidate selection using heuristic multi-criteria scores without joint calibration against the downstream classifier's decision boundary degrades competition metrics.

---

## 9. Verification of Technical Controls
- **Deterministic Ordering**: Maintained via `composite_score DESC`, ties broken by `target_lex_rank ASC`.
- **Target Invariants**: All candidate IDs belong strictly to $S_2$ and $S_3$; no self-matches; no duplicate pairs.
- **Model Invariants**: Exact 57 features, zero retraining, threshold fixed at 0.910, config fingerprint verified as `87f20ceeb84ccc6ea2d48678c7810ac5`.

---

## 10. Final Recommendation
1. **DO NOT ADOPT the Composite Pre-Ranker**: The composite pre-ranker causes a severe 51.1% drop in Macro F0.5.
2. **LOCK the Legacy Integer Evidence Ranker**: The legacy channel-based evidence ranker produces superior downstream precision and higher Macro F0.5 (0.0779 vs 0.0381).
3. **MAINTAIN Cap = 400**: Cap 400 with the legacy ranker remains the optimal configuration for maximum Macro F0.5.
