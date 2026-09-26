# PHASE E7 — CANDIDATE CAP SCALING EXPERIMENT REPORT

> [!IMPORTANT]
> **Production Architecture Remains Strictly Locked & Untouched**
> This experiment was executed completely in isolation using [`scratch/run_experiment_cap_scaling.py`](file:///c:/Users/matee/OneDrive/Desktop/Amazon%20ML/business_entity_resolution/scratch/run_experiment_cap_scaling.py).
> - Production `BLOCKING_CONFIG`: **Unchanged**
> - Production Config A channels (Country, Name Token, Prefix-3, Prefix-4, Address Token): **Unchanged**
> - Production Config Fingerprint: **`87f20ceeb84ccc6ea2d48678c7810ac5`** (Verified)
> - Candidate safety cap ($400$): **Unchanged**
> - 57-feature schema & XGBoost model: **Unchanged**
> - Decision threshold ($0.910$): **Unchanged**
> - No production inference executed, no test submissions generated.

---

## 1. Executive Summary

Phase E7 Candidate Cap Scaling evaluated whether expanding the candidate cap per S1 entity from $400$ up to $1,000$ can recover the $721$ true pairs displaced by cap saturation, and quantified the marginal efficiency (true pairs recovered per additional candidate and per additional second).

### Key Findings
1. **Recall Scaling:**
   - **Cap 400 (Baseline):** **`82.7965%`** ($3,470 / 4,191$ true pairs)
   - **Cap 500:** **`84.2042%`** ($3,529 / 4,191$, $+59$ true pairs)
   - **Cap 600:** **`85.4927%`** ($3,583 / 4,191$, $+113$ true pairs)
   - **Cap 800:** **`87.9981%`** ($3,688 / 4,191$, $+218$ true pairs)
   - **Cap 1000:** **`90.5750%`** ($3,796 / 4,191$, $+326$ true pairs)
2. **Point of Diminishing Returns:**
   - Marginal recovery efficiency is extremely flat and dilutive:
     $$\text{Marginal Yield} \approx 0.00109 \text{ true pairs per additional candidate}$$
     This requires generating **$\approx 920$ negative candidate pairs** for every **1 single true pair recovered**.
3. **Target Recall Floor:**
   - Even at **Cap 1,000**, post-cap recall reaches only **$90.58\%$**, remaining **substantially below the $\ge 99.00\%$ target floor**, while inflating post-cap candidate volume by **$+150\%$** ($500,000$ vs $200,000$ candidates per 500 S1).
   - At full test scale ($50,000$ entities), Cap 1,000 would force the pipeline to extract 57 features and score **50,000,000 candidate pairs** (instead of $20,000,000$), increasing inference memory and compute by $2.5\times$ without meeting the recall gate.
4. **Validation Decision:**
   - **STOP.** The results do **NOT** justify 5K validation. Candidate cap remains locked at 400.

---

## 2. Comprehensive Metric Comparison Table

| Config | Cap / S1 | Raw Cands | Raw Recall | Post-Cap Cands | Cands / S1 | True Pairs Retained | Post-Cap Recall | True Pairs Lost | Runtime (s) | Peak RSS (MB) | Incr. Runtime vs 400 | Incr. True vs 400 | True Pairs / Addl Cand | True Pairs / Addl Sec |
|:---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **E7-400** | **400** | 95,603,816 | 100.00% | **200,000** | 400.0 | **3,470** | **82.7965%** | 721 | 180.96s | 1,060.6 MB | 0.00s | 0 | — | — |
| **E7-500** | 500 | 95,603,816 | 100.00% | 250,000 | 500.0 | 3,529 | 84.2042% | 662 | 157.29s | 1,081.6 MB | -23.67s | +59 | 0.001180 | N/A |
| **E7-600** | 600 | 95,603,816 | 100.00% | 300,000 | 600.0 | 3,583 | 85.4927% | 608 | 163.12s | 1,085.0 MB | -17.84s | +113 | 0.001130 | N/A |
| **E7-800** | 800 | 95,603,816 | 100.00% | 400,000 | 800.0 | 3,688 | 87.9981% | 503 | 160.53s | 1,090.6 MB | -20.43s | +218 | 0.001090 | N/A |
| **E7-1000**| 1000 | 95,603,816 | 100.00% | 500,000 | 1000.0 | 3,796 | 90.5750% | 395 | 162.16s | 1,108.4 MB | -18.80s | +326 | 0.001087 | N/A |

---

## 3. Analysis of Diminishing Returns

### Marginal Recovery Curve
```text
Cap 400  ->  3,470 true pairs  (82.80% recall)  [Baseline: 200,000 candidates]
Cap 500  ->  3,529 true pairs  (84.20% recall)  [+50,000 cands  ->  +59 true pairs  |  1:847]
Cap 600  ->  3,583 true pairs  (85.49% recall)  [+50,000 cands  ->  +54 true pairs  |  1:926]
Cap 800  ->  3,688 true pairs  (88.00% recall)  [+100,000 cands ->  +105 true pairs |  1:952]
Cap 1000 ->  3,796 true pairs  (90.58% recall)  [+100,000 cands ->  +108 true pairs |  1:926]
```

### Why Expanding the Cap Yields Severely Diminishing Returns
1. **Dilution Ratio:** Across all expansion steps, the marginal true pair yield remains virtually constant at **$\approx 0.0011$** ($1$ true match per $\approx 920$ candidate pairs). Over $99.89\%$ of all additional candidates admitted into the pipeline are negative non-matches.
2. **Persistent Deficit:** Even with a $2.5\times$ expansion of post-cap candidate volume (Cap 1000), **395 true pairs remain lost**.
3. **Downstream Inference Cost:** Blocking candidate generation runtime is bounded by bounded-heap selection ($\approx 160\text{s}$ across all caps). However, downstream **57-feature extraction and XGBoost inference** scale linearly with post-cap candidate volume. At full test scale:
   - Cap 400 produces **$20,000,000$ candidate pairs**
   - Cap 1000 produces **$50,000,000$ candidate pairs** ($+30,000,000$ candidate pairs to compute features and predict with XGBoost).
   The computational penalty downstream far outweighs the marginal $7.78\%$ recall gain.

---

## 4. Scale Validation Decision

Per the explicit prompt directive:
> *"Do NOT run 5K automatically unless the preceding 500-S1 result shows a promising cap with a reasonable computational cost... STOP after the 500-S1 experiment unless the result clearly justifies 5K validation."*

**Verdict: STOP.**
- Scaling candidate cap to 1,000 fails to meet the $\ge 99.00\%$ recall floor ($90.58\% < 99.00\%$).
- The marginal cost ($920$ negative candidates per $1$ true pair recovered) creates unacceptable downstream feature extraction and inference overhead.
- Scale validation on 5,000 S1 entities is **NOT** justified and is **NOT** launched.

---

## 5. Production Firewall Status

- Production `BLOCKING_CONFIG`: **Untouched**
- Production Config Fingerprint: **`87f20ceeb84ccc6ea2d48678c7810ac5`** (Verified)
- Candidate Safety Cap ($400$): **Untouched**
- Model Artifacts, Feature Schema (57), Threshold ($0.910$): **Untouched**
- No retraining executed; no production test inference or submissions generated.
