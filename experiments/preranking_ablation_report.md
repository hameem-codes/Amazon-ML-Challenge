# Phase — Pre-Ranking Ablation Experiment Report

**Date**: 2026-09-26  
**Status**: COMPLETE — EXPERIMENT REPORT  
**Production Fingerprint**: `87f20ceeb84ccc6ea2d48678c7810ac5` (VERIFIED & UNCHANGED)  
**Pipeline Contract**: Config A Blocking → Pre-Ranking Ablation (Stages 0–9) → Fixed Cap 400 → Phase 5 XGBoost Model → 57 Features → Threshold 0.99 → Adaptive Gap Rule (0.0001)  
**Validation Benchmark**: 500-France Stress Benchmark (1,434,993 France targets, 4,191 reference true pairs)  
**Output Files**:
- `experiments/preranking_ablation_report.md`
- `experiments/preranking_ablation_results.json`

---

## Executive Summary

This isolated ablation experiment evaluated whether augmenting the legacy evidence ranking with classical similarity signals can improve the candidate ordering of the existing Config A raw candidate pool so that more true pairs survive the fixed 400-candidate cap.

### Key Findings
1. **Recall@400 Breakthrough in Stage 1**: Augmenting the legacy evidence ranking with **Name Token Jaccard** (Stage 1) dramatically improved candidate Recall@400 from **82.7965% to 97.4708%**, capturing **+615 additional true pairs** (4,085 vs 3,470 retained out of 4,191).
2. **Subsequent Signals Provide Diminishing or Negative Returns**: Name Jaro-Winkler and Name TF-IDF (Stages 2–3) yielded zero additional gain over Name Jaccard. Adding address signals (Stages 4–5) actually degraded Recall@400 from 97.4708% down to 94.1541% (-139 pairs) because address variations heavily penalize legitimate matches.
3. **Downstream Macro F0.5 Collapse**: When the superior Stage 1 candidate pool was fed into the locked downstream Phase 5 XGBoost model with the adaptive gap stopping rule, **Macro F0.5 collapsed from 0.317727 to 0.264236** (-0.053491). While true positives increased by +413 (1,910 vs 1,497), false positives surged by **+105.8% (8,223 vs 3,995)**, severely eroding Macro Precision from 32.602% to 26.432%.
4. **Final Recommendation: NO-GO**: In accordance with the experiment's success criteria (which require that end-to-end Macro F0.5 must not fall below 0.3177), this ranking function must **NOT** be adopted into production. The production pipeline must retain the current legacy ranking.

---

## Stage 0 — Baseline Verification

Stage 0 reproduced the baseline legacy evidence ranking with exact fidelity:

| Metric | Expected Baseline | Stage 0 Observed | Status |
|---|---|---|---|
| **Total Reference True Pairs** | 4,191 | 4,191 | Exact Match |
| **Retained True Pairs @ 400** | ~3,470 | **3,470** | **Exact Match (100.0%)** |
| **Recall@400** | ~82.7965% | **82.7965%** | **Exact Match (100.0%)** |
| **Candidates per S1** | 400.00 | 400.00 | Exact Match |
| **S1 Entities Hitting Cap 400** | 500 / 500 | 500 / 500 | Exact Match |

---

## Numerical Scale Inspection

As required by the experiment protocol, the numerical scale of `legacy_score` versus the $[0, 1]$ similarity signals was inspected:

- **Legacy Evidence Score Range**: $130.0$ to $570.0$ (mean $\approx 200.0$ to $215.0$ at the cap boundary). The discrete step between evidence tiers is $\Delta \ge 10.0$ (typically $30, 50, 60,$ or $70$ points).
- **Similarity Signal Sum Range**: Maximum possible sum across all 9 added features is $\le 12.0$ (typical observed sum $\approx 1.0$ to $2.5$ for individual signals).
- **Ordering Effect**: Because the gap between legacy tiers exceeds the maximum similarity boost, the added signals do not cause candidates to cross major evidence tiers (e.g. an address-only candidate with score 60 cannot overtake a candidate with score 200). Instead, the signals act as a **deterministic intra-tier tie-breaker** at the 400-cap cutoff boundary, replacing arbitrary target lexical ID ordering with semantic name alignment.

---

## Mandatory Pre-Ranking Ablation Output Table

All 10 stages were evaluated sequentially on the identical raw candidate pool across all 500 S1 queries:

| Stage | Signals | Recall@400 | True Retained | Delta vs Previous | Delta vs Baseline | Runtime | Peak RAM |
|------:|:--------------------|-----------:|--------------:|------------------:|------------------:|--------:|---------:|
| **0** | **Legacy (Control)**| **82.7965%**| **3,470** | — | — | **1.07s** | **2,927.5 MB** |
| **1** | **+ Name Jaccard** | **97.4708%**| **4,085** | **+615** | **+615** | **1.13s** | **2,927.5 MB** |
| 2 | + Name JW | 97.4708% | 4,085 | +0 | +615 | 1.28s | 2,927.5 MB |
| 3 | + Name TF-IDF | 97.4708% | 4,085 | +0 | +615 | 1.32s | 2,927.5 MB |
| 4 | + Address Jaccard | 97.2799% | 4,077 | -8 | +607 | 1.33s | 2,927.5 MB |
| 5 | + Address TF-IDF | 94.1541% | 3,946 | -131 | +476 | 1.34s | 2,927.5 MB |
| 6 | + Country | 94.1541% | 3,946 | +0 | +476 | 1.33s | 2,927.5 MB |
| 7 | + Blocking Count | 94.1541% | 3,946 | +0 | +476 | 1.34s | 2,927.5 MB |
| 8 | + Prefix | 94.1541% | 3,946 | +0 | +476 | 1.32s | 2,927.5 MB |
| 9 | + Length | 97.2083% | 4,074 | +128 | +604 | 1.33s | 2,927.5 MB |

### Signal Analysis
- **Stage 1 (Name Token Jaccard)**: Responsible for **100% of the maximum candidate recall gain** (+615 true pairs). By prioritizing candidates that share exact token overlap within the cutoff score tier, it cleanly promotes legitimate name variations into the top 400.
- **Stages 2–3 (Name JW & Name TF-IDF)**: Add zero new pairs over Name Jaccard.
- **Stages 4–5 (Address Signals)**: Degrade recall by 139 pairs. Address tokens frequently differ due to suite numbers, street abbreviations, or postal formats. Rewarding address similarity penalizes legitimate matches whose addresses differ slightly.
- **Stage 9 (Name Length Similarity)**: Partially recovers lost recall (+128 pairs over Stage 8) by penalizing spurious matches with drastically different string lengths.

---

## Recall Scaling Across Safety Caps (Best Stage: Stage 1)

For the best pre-ranking configuration (Stage 1: Legacy + Name Jaccard), recall was measured across different candidate caps:

| Candidate Cap | True Retained | True Lost | Recall | Delta vs Cap 400 |
|---|---|---|---|---|
| **Cap 300** | 3,882 | 309 | **92.6271%** | -4.8437% (-203 pairs) |
| **Cap 400** | **4,085** | **106** | **97.4708%** | **Baseline (+0 pairs)** |
| **Cap 500** | 4,127 | 64 | **98.4729%** | +1.0021% (+42 pairs) |

---

## Downstream Pipeline Validation

The candidate set from **Stage 1** (the strongest stage by Recall@400) was evaluated through the locked downstream production pipeline:
- Locked Phase 5 XGBoost Model (`models/xgboost_entity_resolution_phase5_configA.json`)
- 57 locked features (`models/phase5_configA_feature_schema.json`)
- Probability threshold: $p \ge 0.99$
- Adaptive consecutive probability gap rule: $\le 0.0001$

### End-to-End Metrics Comparison

| Metric | Phase 10 Baseline (Legacy Ranking) | Stage 1 Candidate Pool | Impact / Delta | Status |
|---|---|---|---|---|
| **Candidate Recall@400** | 82.7965% | **97.4708%** | **+14.6743% (+615 true pairs)** | Massive Improvement |
| **Macro F0.5** | **0.317727** | **0.264236** | **-0.053491 (-5.35 pts)** | **SEVERE DROP** |
| **Macro Precision** | **32.602%** | **26.432%** | **-6.170%** | Substantial Loss |
| **Macro Recall** | 70.270% | 71.215% | +0.945% | Modest Gain |
| **Pairwise Precision** | 27.258% | 18.849% | -8.409% | Severe Loss |
| **Pairwise Recall** | 35.720% | 45.574% | +9.854% | Gain |
| **True Positives (TP)** | 1,497 | 1,910 | +413 | Gain |
| **False Positives (FP)** | **3,995** | **8,223** | **+4,228 (+105.8%)** | **CATASTROPHIC SURGE** |
| **False Negatives (FN)** | 2,694 | 2,281 | -413 | Reduction |
| **Total Predicted Pairs** | 5,492 | 10,133 | +4,641 (+84.5%) | Large Multiplicity Inflation |
| **Singleton TP** | 111 / 134 | 112 / 134 | +1 TP | Stable |
| **Multi-Match TP** | 1,386 / 4,057 | 1,798 / 4,057 | +412 TP | Gain |
| **Zero-Match With Preds** | 94 / 94 | 94 / 94 | 0 (Unchanged) | Unchanged |

### Root Cause Analysis: The False-Positive Multiplicity Trap
Why did Macro F0.5 drop from 0.317727 to 0.264236 despite candidate recall jumping from 82.8% to 97.5%?

1. **Model Distribution Shift**: The Phase 5 XGBoost model was trained on candidate pairs generated by the legacy evidence ranker. When Name Jaccard is used to pre-rank candidates into the top 400, it populates the candidate pool with dozens of distinct companies that share generic business name tokens (e.g. franchises, chains, or common French descriptors like `boulangerie`, `societe`, `france`, `centre`).
2. **High-Confidence Confusion**: The downstream model assigns extremely high probabilities ($p \ge 0.999$) to these lookalike candidates because of their strong token alignment.
3. **Adaptive Gap Rule Admission**: Because the probabilities for these false matches are saturated ($p \approx 0.9999$), the consecutive probability gap between them is tiny ($\le 0.0001$). As a consequence, the adaptive gap stopping rule admits all of them.
4. **Metric Sensitivity**: The competition metric is **Macro F0.5**, which squares the penalty on precision ($\beta = 0.5$). Gaining +413 true positives at the cost of **+4,228 false positives** (+10.2 false positives per additional true positive) heavily punishes precision, dragging Macro F0.5 down by over 5 full percentage points.

---

## Required Final Report Answers

1. **Baseline Recall@400**: `82.7965%` (3,470 / 4,191 true pairs)
2. **Best Ablation Recall@400**: `97.4708%` (4,085 / 4,191 true pairs)
3. **Improvement in Percentage Points**: `+14.6743%` (+615 true pairs)
4. **Signals Responsible for Improvement**: `Name Token Jaccard` (Stage 1 accounted for 100% of the gain).
5. **Recall Across Caps for Best Ranking (Stage 1)**:
   - Recall@300: `92.6271%` (3,882 pairs)
   - Recall@400: `97.4708%` (4,085 pairs)
   - Recall@500: `98.4729%` (4,127 pairs)
6. **Runtime Comparison**: Stage 0 ranking took `1.07s`; Stage 1 ranking took `1.13s` (+0.06s).
7. **Peak RAM Comparison**: Peak RAM remained stable at `2,927.5 MB`.
8. **End-to-End Macro F0.5 for Best Ranking**: `0.264236` (vs baseline `0.317727`, -0.053491).
9. **Baseline vs New TP/FP/FN**:
   - Baseline: TP = 1,497, FP = 3,995, FN = 2,694
   - Stage 1:  TP = 1,910, FP = 8,223, FN = 2,281
10. **Clear Recommendation**:
    > **NO-GO: Keep the current legacy ranking.**

While Name Token Jaccard produces an impressive raw candidate recall gain (+14.67%), feeding those candidates into the locked downstream pipeline causes false positives to explode by +105.8%, resulting in an unacceptable drop in Macro F0.5. The production pipeline must remain locked to the legacy evidence ranking.
