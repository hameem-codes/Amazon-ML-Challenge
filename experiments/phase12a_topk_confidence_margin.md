# Phase 12A — Top-K, Absolute Confidence & Score-Margin Diagnostic Report

**Date**: 2026-09-26  
**Status**: COMPLETE — EXPERIMENT REPORT  
**Production Fingerprint**: `87f20ceeb84ccc6ea2d48678c7810ac5` (VERIFIED & UNCHANGED)  
**Pipeline Contract**: Config A Blocking → Legacy Evidence Ranking → CAP 400 → 57 Locked Features → Phase 5 XGBoost  
**Validation Benchmark**: 500-France Validation Benchmark (1,434,993 France targets, 4,191 reference true pairs)  
**Evaluated Data**: `scratch/cap400_scored_candidates.pkl` (200,000 scored candidate pairs)  
**Baseline**: Phase 10 Adaptive Gap Rule (`threshold >= 0.99` + `gap <= 0.0001` stopping, Macro F0.5 = `0.317727`)  

---

## Executive Summary

Phase 12A conducted an exhaustive exploration of 192 post-model decision configurations across three structural levers:
1. **Top-K truncation**: $K \in \{1, 2, 3, 5\}$
2. **Absolute probability floors**: $\tau_{\text{abs}} \in \{0.90, 0.92, 0.94, 0.96, 0.98, 0.99\}$
3. **Score-margin ambiguity gates**: $\tau_{\text{gap}} = p_1 - p_2 \in \{0.00, 0.0001, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10\}$

### Key Findings
1. **Pathological K-Collapse**: While fixed $K=2$ achieved the nominal highest Macro F0.5 (`0.341333` vs baseline `0.317727`), it achieved this solely by artificially restricting every query to at most 2 matches. This mechanically **destroyed 76.2% of legitimate multi-match true positives** (reducing multi-match TP from `1,386` to `330`), collapsing overall true positives from `1,497` to `429`.
2. **Catastrophic Recall Destruction under Margin Gates**: Imposing any positive score-margin gate ($\tau_{\text{gap}} \ge 0.0001$) caused a complete collapse of recall. In this real-world entity resolution task, multi-matches naturally produce near-identical model probabilities ($p_1 \approx p_2 \approx 0.9999$). The median top-1 vs top-2 probability gap is only `0.000020` (0.002%). Demanding a margin $\ge 0.0001$ suppressed 86.8% of queries, and $\ge 0.005$ suppressed 100% of queries, reducing TPs to 0.
3. **Verdict**: **No configuration satisfies the criteria to replace the locked Phase 10 baseline**. The Phase 10 adaptive gap stopping rule (`threshold >= 0.99` + `gap <= 0.0001`) remains the superior, balanced decision layer, preserving 70.270% recall and 1,497 true positives without imposing a rigid cardinality cap.

---

## Current Baseline Verification

| Metric | Phase 10 Baseline Expected | Phase 12A Verified Baseline | Status |
|---|---|---|---|
| **Macro F0.5** | **0.317727** | **0.317727** | Exact Match |
| **Macro Precision** | 32.602% | 32.602% | Exact Match |
| **Macro Recall** | 70.270% | 70.270% | Exact Match |
| **Pairwise Precision** | 27.258% | 27.258% | Exact Match |
| **Pairwise Recall** | 35.720% | 35.720% | Exact Match |
| **True Positives (TP)** | 1,497 | 1,497 | Exact Match |
| **False Positives (FP)** | 3,995 | 3,995 | Exact Match |
| **False Negatives (FN)** | 2,694 | 2,694 | Exact Match |
| **Total Predicted Pairs** | 5,492 | 5,492 | Exact Match |
| **Mean Predictions / S1** | 10.98 | 10.98 | Exact Match |
| **Median Predictions / S1** | 3.0 | 3.0 | Exact Match |
| **Singleton TP Retained** | 111 / 134 (82.8%) | 111 / 134 (82.8%) | Exact Match |
| **Multi-Match TP Retained** | 1,386 / 4,057 (34.2%) | 1,386 / 4,057 (34.2%) | Exact Match |

---

## Step 1 — Top-K Only ($p \ge 0.99$)

For each S1 entity, candidate predictions with $p \ge 0.99$ were sorted by `(-probability, target_lex_rank)` and truncated at $K \in \{1, 2, 3, 5\}$:

| Configuration | Macro F0.5 | Macro Prec | Macro Rec | Pairwise Prec | Pairwise Rec | TP | FP | FN | Pred Pairs | Mean Preds | Zero-Match with Preds | Sing TP Ret (%) | Multi TP Ret (%) | Total TP Lost |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Baseline (Phase 10)** | **0.317727** | **32.602%** | **70.270%** | **27.258%** | **35.720%** | **1,497** | **3,995** | **2,694** | **5,492** | **10.98** | **94 / 94** | **100.0% (111)** | **100.0% (1,386)** | **0** |
| **Top-1** ($p \ge 0.99$) | 0.308465 | 46.600% | 39.611% | 46.600% | 5.559% | 233 | 267 | 3,958 | 500 | 1.00 | 94 / 94 | 52.2% (58) | 12.6% (175) | 1,264 |
| **Top-2** ($p \ge 0.99$) | 0.341333 | 42.900% | 55.329% | 42.900% | 10.236% | 429 | 571 | 3,762 | 1,000 | 2.00 | 94 / 94 | 89.2% (99) | 23.8% (330) | 1,068 |
| **Top-3** ($p \ge 0.99$) | 0.312130 | 37.000% | 61.887% | 37.000% | 13.243% | 555 | 945 | 3,636 | 1,500 | 3.00 | 94 / 94 | 97.3% (108) | 32.2% (447) | 942 |
| **Top-5** ($p \ge 0.99$) | 0.259144 | 28.953% | 69.010% | 29.014% | 17.204% | 721 | 1,764 | 3,470 | 2,485 | 4.97 | 94 / 94 | 105.4% (117) | 43.6% (604) | 776 |

### Analysis of Step 1
- **Top-1**: Fails catastrophically on multi-match entities, retaining only 12.6% of multi-match TPs and dropping Macro Recall to 39.611%.
- **Top-2**: Yields a nominal peak Macro F0.5 of 0.341333, but sacrifices 1,068 legitimate true positives (76.2% multi-match TP loss).
- **Top-5**: Reduces Macro F0.5 to 0.259144 due to accumulating 1,764 false positives while still missing 56.4% of multi-match TPs.

---

## Step 2 — Top-K + Absolute Confidence Threshold Sweep

We evaluated the cross-product of $K \in \{1, 2, 3, 5\}$ with $\tau_{\text{abs}} \in \{0.90, 0.92, 0.94, 0.96, 0.98, 0.99\}$:

| K | $\tau_{\text{abs}}$ | Macro F0.5 | Macro Prec | Macro Rec | TP | FP | Pred Pairs | Multi TP Ret (%) |
|---|---|---|---|---|---|---|---|---|
| **1** | 0.90 | 0.308465 | 46.600% | 39.611% | 233 | 267 | 500 | 12.6% |
| **1** | 0.92 | 0.308465 | 46.600% | 39.611% | 233 | 267 | 500 | 12.6% |
| **1** | 0.94 | 0.308465 | 46.600% | 39.611% | 233 | 267 | 500 | 12.6% |
| **1** | 0.96 | 0.308465 | 46.600% | 39.611% | 233 | 267 | 500 | 12.6% |
| **1** | 0.98 | 0.308465 | 46.600% | 39.611% | 233 | 267 | 500 | 12.6% |
| **1** | 0.99 | 0.308465 | 46.600% | 39.611% | 233 | 267 | 500 | 12.6% |
| **2** | 0.90 | 0.341333 | 42.900% | 55.329% | 429 | 571 | 1,000 | 23.8% |
| **2** | 0.92 | 0.341333 | 42.900% | 55.329% | 429 | 571 | 1,000 | 23.8% |
| **2** | 0.94 | 0.341333 | 42.900% | 55.329% | 429 | 571 | 1,000 | 23.8% |
| **2** | 0.96 | 0.341333 | 42.900% | 55.329% | 429 | 571 | 1,000 | 23.8% |
| **2** | 0.98 | 0.341333 | 42.900% | 55.329% | 429 | 571 | 1,000 | 23.8% |
| **2** | 0.99 | 0.341333 | 42.900% | 55.329% | 429 | 571 | 1,000 | 23.8% |
| **3** | 0.90 | 0.312130 | 37.000% | 61.887% | 555 | 945 | 1,500 | 32.2% |
| **3** | 0.92 | 0.312130 | 37.000% | 61.887% | 555 | 945 | 1,500 | 32.2% |
| **3** | 0.94 | 0.312130 | 37.000% | 61.887% | 555 | 945 | 1,500 | 32.2% |
| **3** | 0.96 | 0.312130 | 37.000% | 61.887% | 555 | 945 | 1,500 | 32.2% |
| **3** | 0.98 | 0.312130 | 37.000% | 61.887% | 555 | 945 | 1,500 | 32.2% |
| **3** | 0.99 | 0.312130 | 37.000% | 61.887% | 555 | 945 | 1,500 | 32.2% |
| **5** | 0.90 | 0.258222 | 28.870% | 69.010% | 721 | 1,777 | 2,498 | 43.6% |
| **5** | 0.92 | 0.258222 | 28.870% | 69.010% | 721 | 1,777 | 2,498 | 43.6% |
| **5** | 0.94 | 0.258222 | 28.870% | 69.010% | 721 | 1,775 | 2,496 | 43.6% |
| **5** | 0.96 | 0.258222 | 28.870% | 69.010% | 721 | 1,774 | 2,495 | 43.6% |
| **5** | 0.98 | 0.258515 | 28.897% | 69.010% | 721 | 1,771 | 2,492 | 43.6% |
| **5** | 0.99 | 0.259144 | 28.953% | 69.010% | 721 | 1,764 | 2,485 | 43.6% |

### Analysis of Step 2
Because every S1 entity has at least 3 candidates with model probability $\ge 0.99$, changing $\tau_{\text{abs}}$ between 0.90 and 0.99 has virtually zero impact on Top-1, Top-2, or Top-3 predictions (the metrics are 100% invariant across $\tau_{\text{abs}}$). Only at $K=5$ does threshold 0.99 slightly prune 13 low-confidence false positives.

---

## Step 3 — Top-K + Confidence + Score-Margin Gates

We tested emission gating based on top-candidate margin:
An S1 entity emits predictions if and only if $p_1 \ge \tau_{\text{abs}}$ AND $(p_1 - p_2) \ge \tau_{\text{gap}}$.

### Impact of Margin Threshold $\tau_{\text{gap}}$ (Shown at $K=2, \tau_{\text{abs}}=0.99$)

| Margin $\tau_{\text{gap}}$ | Macro F0.5 | Macro Precision | Macro Recall | TP | FP | Total Pred Pairs | Active Entities | Verdict |
|---|---|---|---|---|---|---|---|---|
| **0.0000** | **0.341333** | 42.900% | 55.329% | 429 | 571 | 1,000 | 500 / 500 (100.0%) | Control |
| **0.0001** | **0.176543** | 90.100% | 23.278% | 33 | 99 | 132 | 66 / 500 (13.2%) | **Severe Collapse** |
| **0.0010** | **0.191333** | 99.700% | 19.400% | 3 | 3 | 6 | 3 / 500 (0.6%) | **Near-Total Collapse** |
| **0.0050** | **0.188000** | 100.000% | 18.800% | 0 | 0 | 0 | 0 / 500 (0.0%) | **Total Suppression** |
| **0.0100** | **0.188000** | 100.000% | 18.800% | 0 | 0 | 0 | 0 / 500 (0.0%) | **Total Suppression** |
| **0.0200** | **0.188000** | 100.000% | 18.800% | 0 | 0 | 0 | 0 / 500 (0.0%) | **Total Suppression** |
| **0.0500** | **0.188000** | 100.000% | 18.800% | 0 | 0 | 0 | 0 / 500 (0.0%) | **Total Suppression** |
| **0.1000** | **0.188000** | 100.000% | 18.800% | 0 | 0 | 0 | 0 / 500 (0.0%) | **Total Suppression** |

*(Note: When 0 predictions are emitted across all queries, Macro Recall is 18.8% because the 94 GT zero-match entities achieve Precision=1, Recall=1, F0.5=1 when empty).*

### Why Score-Margin Ambiguity Gating Fails Fundamentally
In competitive entity resolution with high-multiplicity targets (e.g. company registries where subsidiaries and branches share near-identical names and addresses), the model generates tight probability clusters for true matches:
$$\text{Candidate 1}: p = 0.999993 \quad \text{Candidate 2}: p = 0.999988 \implies \Delta p = 0.000005$$
Demanding that $p_1 - p_2 \ge \tau_{\text{gap}}$ misinterprets legitimate multi-match agreement as "classifier ambiguity" and suppresses almost all genuine entities. Margin gating is fatally mismatched to multi-match entity resolution.

---

## Step 4 & 5 — Subgroup Analysis for Top 10 Configurations

For the top configurations by Macro F0.5, we broke down performance across the three ground-truth subgroups:

| Rank | Configuration | Overall F0.5 | Overall Prec | Overall Rec | Overall TP | Zero-Match FP | Singleton TP (Ret %) | Multi-Match TP (Ret %) |
|---|---|---|---|---|---|---|---|---|
| **Base** | **Phase 10 (Gap <= 0.0001)** | **0.317727** | **32.602%** | **70.270%** | **1,497** | **395** | **111 / 134 (100.0%)** | **1,386 / 4,057 (100.0%)** |
| 1–6 | $K=2, \tau_{\text{gap}}=0.0, \tau_{\text{abs}} \in [0.90..0.99]$ | 0.341333 | 42.900% | 55.329% | 429 | 188 | 99 / 134 (89.2%) | **330 / 4,057 (23.8%)** |
| 7–12 | $K=3, \tau_{\text{gap}}=0.0, \tau_{\text{abs}} \in [0.90..0.99]$ | 0.312130 | 37.000% | 61.887% | 555 | 282 | 108 / 134 (97.3%) | **447 / 4,057 (32.2%)** |
| 13–18 | $K=1, \tau_{\text{gap}}=0.0, \tau_{\text{abs}} \in [0.90..0.99]$ | 0.308465 | 46.600% | 39.611% | 233 | 94 | 58 / 134 (52.2%) | **175 / 4,057 (12.6%)** |
| 19 | $K=5, \tau_{\text{gap}}=0.0, \tau_{\text{abs}}=0.99$ | 0.259144 | 28.953% | 69.010% | 721 | 470 | 117 / 134 (105.4%) | **604 / 4,057 (43.6%)** |

### Subgroup Insights
1. **Multi-Match Degradation in $K=2$**:
   Multi-match entities represent 54.4% of the query entities and 96.8% of all true match pairs in the benchmark. Capping predictions at $K=2$ destroys **1,056 true positives** (76.2% loss), leaving only 330 TPs.
2. **False Positive Reduction vs Recall**:
   While $K=2$ reduces zero-match false positives from 395 to 188, it does not leave a single zero-match entity empty (all 94 receive exactly 2 false predictions). The higher F0.5 is an artifact of truncation on precision, not genuine zero-match identification.

---

## Step 6 — Selection Criteria Assessment

Responding directly to the prompt's evaluation criteria:

### A. Highest Macro F0.5
Fixed $K=2$ ($\tau_{\text{abs}} \in [0.90, 0.99], \tau_{\text{gap}} = 0.0$) achieves Macro F0.5 = `0.341333`.

### B. Best Precision/Recall Balance
The Phase 10 adaptive gap rule (`threshold >= 0.99`, `gap <= 0.0001`) provides the best balance: **32.602% precision** with **70.270% recall**, yielding **1,497 true positives**.

### C. Best Multi-Match TP Retention
The Phase 10 adaptive gap rule achieves **1,386 multi-match true positives** (100.0% retention vs baseline), outperforming $K=2$ by **4.2x** (1,386 vs 330) and $K=3$ by **3.1x** (1,386 vs 447).

### D. Lowest FP Count with Reasonable Recall
$K=3$ achieves 945 FPs with 61.887% recall, but drops multi-match retention to 32.2%. Any configuration with margin gates collapses recall to $< 24\%$.

### E. Does Any Configuration Clearly Beat the Current 0.317727 Baseline?
**NO.**
The prompt states:
> *"A configuration that gets high F0.5 by simply predicting Top-1/Top-2 but destroys multi-match recall should NOT automatically be selected."*

Because $K=2$ relies entirely on a pathological cardinality collapse that sacrifices over 76% of legitimate multi-match pairs, and score-margin gating triggers near-total prediction suppression, **no tested configuration satisfies the safety and retention criteria to beat the Phase 10 baseline**.

---

## Step 7 — Final Decision & Architectural Guidance

1. **Rejection of Top-K Truncation**: Rigidly capping predictions at $K=1$ or $K=2$ severely penalizes multi-match entities which constitute the majority of ground-truth entity connections.
2. **Rejection of Score-Margin Gates**: Score-margin gating ($\tau_{\text{gap}} \ge 0.0001$) is fundamentally unviable for entity resolution problems with multi-match target pools.
3. **Locking the Phase 10 Adaptive Rule**: The Phase 10 rule (`threshold >= 0.99` with consecutive `gap <= 0.0001` stopping) remains the optimal and safest decision layer. It naturally adapts: stopping after 1–2 predictions when confidence drops, while seamlessly expanding to capture multi-match clusters when candidate probabilities remain tied.

---

```
PHASE 12A TOP-K CONFIDENCE MARGIN EXPERIMENT COMPLETE
```
