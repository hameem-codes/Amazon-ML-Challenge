# Phase 10 — Per-S1 Adaptive Prediction Selection Report

## 1. Executive Summary & Objective
In Phase 9, error auditing revealed that **85.91% of all false positives at threshold 0.99** originate from entities that already possess valid true matches, driven by the model emitting an average of **38.92 predictions per S1 entity** (median 23, max 299).

The objective of Phase 10 is to determine whether **per-S1 adaptive prediction selection** can curb this massive multiplicity and improve the final competition metric (**Macro F0.5**) without altering the underlying model, blocking, or feature pipeline:
$$\text{Candidate Pool} \longrightarrow \text{Probabilities } p \ge 0.99 \longrightarrow \text{Per-S1 Selection Rule} \longrightarrow \text{Final Predictions}$$

### Key Empirical Findings:
1. **Dramatic Macro F0.5 Improvements**:
   - **Top-K 2**: Achieves the highest overall Macro F0.5 of **0.341333** (**+172.0% improvement** over baseline 0.125500), boosting Macro Precision from 11.13% to **42.90%**.
   - **Score-Gap Stopping (Gap $\le$ 0.0001)**: Achieves **Macro F0.5 = 0.317727** (**+153.2% improvement**) while preserving **1,497 True Positives** (68.2% of all TPs) and **70.27% Macro Recall**, cutting false positives by **76.9%** (from 17,263 down to 3,995).
   - **Strong Name Filter (C2)**: Achieves **Macro F0.5 = 0.159197** (+26.9% improvement) while preserving **100% of True Positives (2,195/2,195)** and eliminating 2,567 false positives.
2. **Zero-Match Blind Spot Persists**:
   - Pure top-K and gap rules still allocate predictions to zero-match entities (all 94 zero-match entities receive predictions under Top-K rules because every S1 has candidates $\ge 0.99$).
   - A dedicated zero-match gate or evidence filter is required to leave zero-match entities empty.

---

## 2. Locked Pipeline & Experimental Invariants
All diagnostic evaluations were conducted strictly read-only on the locked Phase 5 architecture:
- **Blocking**: Config A (Country, Name Token, Prefix-3, Prefix-4, Address Token with selective address DF cap 50,000).
- **Ranking**: Legacy integer evidence ranker (`cand_scores` with channel weights 70/60/50/30).
- **Candidate Cap**: Exactly 400 candidates per S1 entity.
- **Features**: Locked 57-feature schema with transductive TF-IDF embeddings.
- **Model**: `models/xgboost_entity_resolution_phase5_configA.json`.
- **Benchmark**: 500 France S1 entities against 1,434,993 France targets (4,191 reference true pairs).
- **Production Fingerprint**: Verified unchanged as `87f20ceeb84ccc6ea2d48678c7810ac5`.

---

## 3. Baseline Reproduction (@ Threshold 0.99)
Before testing adaptive rules, the baseline performance at threshold 0.99 was reproduced exactly:
- **Macro F0.5**: `0.125500`
- **Macro Precision**: `11.125%`
- **Macro Recall**: `83.142%`
- **True Positives (TP)**: `2,195`
- **False Positives (FP)**: `17,263`
- **False Negatives (FN)**: `1,996`
- **Total Predictions**: `19,458`
- **Mean Predictions / S1**: `38.92` (Median: `23.00`)

---

## 4. Master Comparison Table

| Selection Rule | Macro F0.5 | Macro Prec (%) | Macro Rec (%) | Pairwise Prec (%) | Pairwise Rec (%) | TP | FP | FN | Pred Pairs | Mean Pred/S1 | Med Pred/S1 | Zero-Match with Preds | Zero-Match Empty |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline 0.99** | 0.125500 | 11.125% | **83.142%** | 11.281% | 52.374% | **2,195** | 17,263 | 1,996 | 19,458 | 38.92 | 23.00 | 94 / 94 | 0 / 94 |
| **Top-K 1** | 0.308465 | **46.600%** | 39.611% | **46.600%** | 5.560% | 233 | **267** | 3,958 | 500 | 1.00 | 1.00 | 94 / 94 | 0 / 94 |
| **Top-K 2 (Best F0.5)** | **0.341333** | 42.900% | 55.329% | 42.900% | 10.236% | 429 | 571 | 3,762 | 1,000 | 2.00 | 2.00 | 94 / 94 | 0 / 94 |
| **Top-K 3** | 0.312130 | 37.000% | 61.887% | 37.000% | 13.243% | 555 | 945 | 3,636 | 1,500 | 3.00 | 3.00 | 94 / 94 | 0 / 94 |
| **Top-K 5** | 0.259144 | 28.953% | 69.010% | 29.014% | 17.204% | 721 | 1,764 | 3,470 | 2,485 | 4.97 | 5.00 | 94 / 94 | 0 / 94 |
| **Top-K 10** | 0.187991 | 19.893% | 73.964% | 19.923% | 22.310% | 935 | 3,758 | 3,256 | 4,693 | 9.39 | 10.00 | 94 / 94 | 0 / 94 |
| **Top-K 20** | 0.151843 | 15.042% | 76.678% | 14.683% | 27.869% | 1,168 | 6,787 | 3,023 | 7,955 | 15.91 | 20.00 | 94 / 94 | 0 / 94 |
| **Top-K 50** | 0.134509 | 12.415% | 80.579% | 11.684% | 37.724% | 1,581 | 11,950 | 2,610 | 13,531 | 27.06 | 23.00 | 94 / 94 | 0 / 94 |
| **Top-K 100** | 0.127839 | 11.442% | 82.139% | 11.149% | 46.123% | 1,933 | 15,405 | 2,258 | 17,338 | 34.68 | 23.00 | 94 / 94 | 0 / 94 |
| **Gap $\le$ 0.0001 (Best Balanced)** | **0.317727** | 32.602% | 70.270% | 27.258% | 35.719% | 1,497 | 3,995 | 2,694 | 5,492 | 10.98 | 4.00 | 94 / 94 | 0 / 94 |
| **Gap $\le$ 0.0005** | 0.209166 | 19.808% | 78.289% | 14.851% | 47.650% | 1,997 | 11,450 | 2,194 | 13,447 | 26.89 | 10.00 | 94 / 94 | 0 / 94 |
| **Gap $\le$ 0.001** | 0.178108 | 16.354% | 81.740% | 12.620% | 51.133% | 2,143 | 14,838 | 2,048 | 16,981 | 33.96 | 16.00 | 94 / 94 | 0 / 94 |
| **Gap $\le$ 0.002** | 0.148021 | 13.403% | 82.817% | 11.606% | 52.088% | 2,183 | 16,627 | 2,008 | 18,810 | 37.62 | 22.00 | 94 / 94 | 0 / 94 |
| **Gap $\le$ 0.005** | 0.127834 | 11.345% | **83.142%** | 11.304% | 52.374% | **2,195** | 17,223 | 1,996 | 19,418 | 38.84 | 23.00 | 94 / 94 | 0 / 94 |
| **Gap $\le$ 0.01** | 0.125500 | 11.125% | **83.142%** | 11.281% | 52.374% | **2,195** | 17,263 | 1,996 | 19,458 | 38.92 | 23.00 | 94 / 94 | 0 / 94 |
| **C1: Keys $\ge$ 2** | 0.125807 | 11.154% | **83.142%** | 11.287% | 52.374% | **2,195** | 17,253 | 1,996 | 19,448 | 38.90 | 23.00 | 94 / 94 | 0 / 94 |
| **C2: Strong Name (Recall-Safe)** | **0.159197** | 14.318% | **83.142%** | 12.995% | 52.374% | **2,195** | 14,696 | 1,996 | 16,891 | 33.78 | 18.00 | 94 / 94 | 0 / 94 |
| **C3: Strong Addr** | 0.161596 | 17.125% | 57.470% | 11.907% | 13.601% | 570 | 4,217 | 3,621 | 4,787 | 9.57 | 6.00 | 92 / 94 | 2 / 94 |
| **C4: Strong Name or Addr** | 0.142371 | 12.714% | **83.142%** | 12.271% | 52.374% | **2,195** | 15,693 | 1,996 | 17,888 | 35.78 | 20.00 | 94 / 94 | 0 / 94 |

---

## 5. Experiment A: Top-K Diagnostics
Under pure Top-K selection, predictions are sorted by `(-probability, target_lex_rank)` per S1 entity and truncated at $K$:

### Group Breakdown for Top-K 2 (Best Macro F0.5):
- **Ground-Truth Zero-Match (94 S1s)**:
  - Predictions: 188 (2 per S1).
  - FP: 188, TP: 0.
  - Macro F0.5: `0.000000` (all 94 entities receive false matches).
- **Ground-Truth Singleton (134 S1s)**:
  - Predictions: 268 (2 per S1).
  - TP: 99, FP: 169.
  - Macro Precision: `36.940%`, Macro Recall: `73.881%`, Macro F0.5: `0.410448` (vs 0.091284 baseline).
- **Ground-Truth Multi-Match (272 S1s)**:
  - Predictions: 544 (2 per S1).
  - TP: 330, FP: 214.
  - Macro Precision: `60.662%`, Macro Recall: `30.753%`, Macro F0.5: `0.425245` (vs 0.185728 baseline).

**Insight**: Truncating at $K=2$ dramatically boosts precision on matched entities (from 11% to 43%), driving Macro F0.5 to **0.341333**. However, it caps recall at 55.33% because multi-match entities cannot capture their 3rd, 4th, or subsequent true matches.

---

## 6. Experiment B: Score-Gap Stopping Diagnostics
Experiment B evaluates dynamic truncation based on confidence drops:
$$\text{Keep } p_0; \quad \text{continue while } (p_i - p_{i+1}) \le \text{gap\_threshold}$$

### Performance of Gap $\le$ 0.0001:
- **Macro F0.5**: **0.317727** (+153.2% vs baseline).
- **Macro Recall**: **70.270%** (vs 55.33% for Top-K 2).
- **True Positives**: **1,497** (vs 429 for Top-K 2 — **3.5x more true pairs recovered!**).
- **False Positives**: **3,995** (vs 17,263 baseline — **76.9% reduction in false alarms**).
- **Average Predictions / S1**: **10.98** (median 4.00, down from 38.92).

**Insight**: Gap stopping dynamically adapts: for entities with tight probability clusters (multi-match clusters), it retains 5 to 20 candidates, capturing 1,497 true positives. When a steep drop occurs ($> 0.0001$), it stops, pruning 13,268 false positives.

---

## 7. Experiment C: Evidence-Aware Filtering Diagnostics
Experiment C tests filtering predictions $\ge 0.99$ by requiring corroborating evidence:

- **C2 (Strong Name Evidence: Exact Match OR Token Jaccard $\ge 0.5$ OR Name JW $\ge 0.90$)**:
  - **Macro F0.5**: **0.159197** (+26.9% improvement).
  - **Macro Recall**: **83.142%** (**100% Recall Retention — Zero True Positives Lost!**).
  - **TP**: **2,195 / 2,195**.
  - **FP Reduction**: Eliminates **2,567 false positives** (17,263 $\to$ 14,696).
- **C1 (Shared Keys $\ge 2$)**: Too weak (only eliminates 10 FPs).
- **C3 (Strong Address Evidence)**: Too restrictive because many targets omit addresses (destroys recall: 83.14% $\to$ 57.47%).

---

## 8. Experiment D: Zero-Match Safety Diagnostic
Across the 500 S1 queries, we evaluated whether existing features can distinguish zero-match entities without ground truth:

| Group | Max Probability (Median) | Preds $\ge 0.99$ (Median) | Max Name JW (Median) | Max Address Jaccard (Median) | Max Shared Keys (Median) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **GT Zero-Match (94 S1s)** | 1.0000 (Mean: 0.9999) | 19.50 (Mean: 25.88) | 0.9650 (Mean: 0.9568) | 0.7778 (Mean: 0.7773) | 4.00 (Mean: 3.96) |
| **GT Singleton (134 S1s)** | 1.0000 (Mean: 1.0000) | 15.00 (Mean: 25.51) | **1.0000** (Mean: 0.9980) | 0.7778 (Mean: 0.7883) | 4.00 (Mean: 3.99) |
| **GT Multi-Match (272 S1s)**| 1.0000 (Mean: 1.0000) | 30.00 (Mean: 50.03) | **1.0000** (Mean: 0.9991) | 0.7778 (Mean: 0.7787) | 4.00 (Mean: 4.00) |

### Technical Feasibility of a Zero-Match Gate:
1. **Model Probability Cannot Distinguish Zero-Match**: In all three groups, maximum probability saturates at 1.0000 (minimum max-prob for zero-match is 0.9995).
2. **Exact Name Match Asymmetry**:
   - In Singleton and Multi-match groups, **median Max Name JW is 1.0000** (over 92% of matched entities have at least one exact string match in the candidate pool).
   - In the Zero-Match group, **median Max Name JW is 0.9650** (fuzzy variation).
3. **Conclusion**: A zero-match gate cannot be based on probability alone, but an exact-match corroboration rule (`has_exact_name_candidate`) is technically viable.

---

## 9. Final Synthesis: Answering Key Questions

1. **Which rule achieves the highest Macro F0.5?**
   - **Top-K 2** achieves the highest scalar Macro F0.5 of **0.341333** (+172.0% over baseline 0.125500).
2. **Which rule provides the best precision/recall balance?**
   - **Score Gap $\le$ 0.0001** achieves **0.317727 Macro F0.5** (+153.2%) while maintaining **70.27% Macro Recall** and recovering **1,497 True Positives** (vs only 429 for Top-K 2).
3. **Are zero-match entities still systematically over-predicted?**
   - **Yes.** Under pure Top-K and gap rules, 94/94 zero-match entities still receive predictions because every S1 entity has candidates with $p \ge 0.99$.
4. **Are multi-match entities harmed by Top-K?**
   - **Yes.** Top-K 2 truncates multi-match entities from 2,067 TPs down to 330 TPs. In contrast, **Gap $\le$ 0.0001** retains 1,386 TPs in multi-match entities (67% of all multi-match true pairs).

---

## 10. Production Firewall Verification
- All evaluations performed strictly offline on the validation set.
- Zero test data, test labels, or submission files generated.
- Production fingerprint confirmed as `87f20ceeb84ccc6ea2d48678c7810ac5`.
- Production codebase completely untouched.
