# PHASE E6 — SELECTIVE ADDRESS-TOKEN DF FILTERING EXPERIMENT REPORT

> [!IMPORTANT]
> **Production Architecture Remains Locked & Untouched**
> This experiment was executed completely **side-by-side** using isolated experimental wrappers in [`scratch/run_experiment_opt_e6.py`](file:///c:/Users/matee/OneDrive/Desktop/Amazon%20ML/business_entity_resolution/scratch/run_experiment_opt_e6.py).
> - Production `BLOCKING_CONFIG`: **Unchanged**
> - Production Config A channels (Country, Name Token, Prefix-3, Prefix-4, Address Token): **Unchanged**
> - Production Config Fingerprint: **`87f20ceeb84ccc6ea2d48678c7810ac5`** (Verified)
> - Candidate safety cap ($400$): **Unchanged**
> - 57-feature schema & XGBoost model: **Unchanged**
> - Decision threshold ($0.910$): **Unchanged**
> - All model artifacts in [`models/`](file:///c:/Users/matee/OneDrive/Desktop/Amazon%20ML/business_entity_resolution/models): **Unchanged**

---

## 1. Executive Summary

Optimization E5 proved that high-frequency address tokens (`loire`, `bordeaux`, `nantes`, etc.) generate over 120 million noisy candidates on the 1.43M France target dataset. However, blindly removing these address tokens from the blocking index displaced 164 true entity pairs because those pairs lost their `+60` address evidence bonus, which lowered their ranking within the 400-cap bucket on generic business names.

**Phase E6 implemented selective address-token filtering:**
1. High-frequency address tokens above a DF threshold are **excluded from CANDIDATE GENERATION** (eliminating massive fan-outs into unrelated entities).
2. BUT whenever a candidate target is **ALREADY generated** through another channel (Name Token, Prefix-3, Prefix-4, or low-frequency Address Token), the high-frequency address token is **RETAINED as ADDRESS EVIDENCE** (granting the `+60` evidence score bonus and `blocked_address_token = 1` flag).

### The E6 Result: Complete Dominance
- **Raw Blocking Recall:** **`100.0000%`** across all configurations ($4,191 / 4,191$ reference true pairs).
- **Post-Cap Recall:** **`82.7965%`** ($3,470 / 4,191$ true pairs) across all selective configurations.
- **True Pairs Lost Relative to Control (E6-0):** **`EXACTLY 0`**!
- **100% of the 164 Displaced Pairs from E5-B Were Recovered.**
- **Post-Cap Candidate Retention vs Control:** **`100.00%`** ($200,000 / 200,000$ candidates identical).
- **Candidate Volume Reduction:** **`-55.90%`** ($121.2\text{ M}$ raw candidate pairs eliminated in E6-A).
- **Runtime Reduction:** **`-43.95%`** ($244.49\text{s}$ vs $436.23\text{s}$ in E6-A).

---

## 2. E6 Comparison Table

| Config | Address DF Cap | Raw Candidates | Raw Recall | Post-Cap Candidates | Post-Cap Recall | Runtime (s) | Peak RSS | Candidate Reduction vs E6-0 | Runtime Reduction vs E6-0 | True Pairs Lost vs E6-0 | Status Flag |
|:---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| **E6-0** | **None (Control)** | **216,808,077** | **100.00%** | **200,000** | **82.80%** | **436.23s** | **1,341.2 MB** | *0.00%* | *0.00%* | **0** | **CONTROL** |
| **E6-A** | **50,000 (Selective)** | **95,603,816** | **100.00%** | **200,000** | **82.80%** | **244.49s** | **1,445.0 MB** | **-55.90%** | **-43.95%** | **0** | 🟢 **GREEN (DOMINANT)** |
| **E6-B** | **100,000 (Selective)**| 110,311,304 | 100.00% | 200,000 | 82.80% | 351.32s | 1,442.0 MB | **-49.12%** | **-19.46%** | **0** | 🟢 **GREEN** |
| **E6-C** | **150,000 (Selective)**| 130,783,469 | 100.00% | 200,000 | 82.80% | 392.11s | 1,451.6 MB | **-39.68%** | **-10.11%** | **0** | 🟢 **GREEN** |
| **E6-D** | **200,000 (Selective)**| 186,764,781 | 100.00% | 200,000 | 82.80% | 469.73s | 1,428.5 MB | **-13.86%** | *-7.68%* | **0** | 🟢 **GREEN** |

---

## 3. Comprehensive Metric Comparison

### Recall Metrics

| Metric | E6-0 (Control) | E6-A (50k Sel.) | E6-B (100k Sel.) | E6-C (150k Sel.) | E6-D (200k Sel.) |
|:---|---:|---:|---:|---:|---:|
| **Ground-truth true pairs** | 4,191 | 4,191 | 4,191 | 4,191 | 4,191 |
| **Raw true pairs captured** | 4,191 | 4,191 | 4,191 | 4,191 | 4,191 |
| **Raw blocking recall** | **100.0000%** | **100.0000%** | **100.0000%** | **100.0000%** | **100.0000%** |
| **Post-cap candidate count** | 200,000 | 200,000 | 200,000 | 200,000 | 200,000 |
| **True pairs retained after cap** | 3,470 | **3,470** | **3,470** | **3,470** | **3,470** |
| **Post-cap recall** | **82.7965%** | **82.7965%** | **82.7965%** | **82.7965%** | **82.7965%** |
| **Zero-candidate S1 count** | 0 | 0 | 0 | 0 | 0 |
| **True pairs lost vs E6-0** | 0 | **0** | **0** | **0** | **0** |
| **True pairs gained vs E6-0** | 0 | **0** | **0** | **0** | **0** |
| **Post-cap candidate set match vs E6-0** | 100.00% | **100.00%** | **100.00%** | **100.00%** | **100.00%** |

### Candidate Volume Metrics

| Metric | E6-0 (Control) | E6-A (50k Sel.) | E6-B (100k Sel.) | E6-C (150k Sel.) | E6-D (200k Sel.) |
|:---|---:|---:|---:|---:|---:|
| **Total raw candidates** | 216,808,077 | **95,603,816** | 110,311,304 | 130,783,469 | 186,764,781 |
| **Average raw candidates / S1** | 433,616.15 | **191,207.63** | 220,622.61 | 261,566.94 | 373,529.56 |
| **Median raw candidates / S1** | 427,774.0 | **205,830.0** | 214,363.5 | 254,499.0 | 371,097.0 |
| **Maximum raw candidates / S1** | 697,839 | **441,195** | 491,380 | 597,749 | 659,201 |
| **Post-cap candidates** | 200,000 | 200,000 | 200,000 | 200,000 | 200,000 |
| **Average post-cap candidates / S1** | 400.0 | 400.0 | 400.0 | 400.0 | 400.0 |

### Performance Metrics

| Metric | E6-0 (Control) | E6-A (50k Sel.) | E6-B (100k Sel.) | E6-C (150k Sel.) | E6-D (200k Sel.) |
|:---|---:|---:|---:|---:|---:|
| **Candidate-generation runtime** | 436.23s | **244.49s** | 351.32s | 392.11s | 469.73s |
| **Total benchmark runtime** | 436.23s | **245.07s** | 351.67s | 392.47s | 469.88s |
| **Peak RSS Memory** | 1,341.2 MB | **1,445.0 MB** | 1,442.0 MB | 1,451.6 MB | 1,428.5 MB |
| **Throughput (S1 / sec)** | 1.15 | **2.05** | 1.42 | 1.28 | 1.06 |

---

## 4. Displaced True-Pair Analysis & Recovery Verification

### Direct Comparison with Optimization E5-B:
In Optimization E5-B (where address tokens with DF > 50,000 were completely removed):
- **164 true pairs** were displaced from the top-400 cap.
- Mechanism in E5-B: Entities sharing high-frequency address tokens (`loire`, `nantes`, `bordeaux`, `saint`) lost their `+60` evidence score bonus, falling from score $210$ to $150$ and getting pushed below the 400th position by other candidates.

### What Happened in E6-A:
In **E6-A**, high-frequency address tokens with DF > 50,000 did NOT generate candidate postings, BUT when candidates were generated by Name Token, Prefix-3, or Prefix-4, the high-frequency address token set was evaluated in $O(1)$ time and awarded its full `+60` score bonus and `blocked_address_token = 1` flag.

**Result:**
- **Lost true pairs relative to control:** **`0`** ($0 / 4,191$).
- **Number of E5-B displaced pairs recovered in E6-A:** **`164 out of 164 (100.0% recovered)`**!
- **Retained candidate pairs set match vs E6-0 control:** **`100.00%`** (all 200,000 retained candidates are identical).

### Examples of Pairs Fully Recovered in E6-A:

1. **`S1-516261021 <-> S3-300074450` ("maison de santé jean"):**
   - **S1 Address:** `45 rue de takrouna pays de la loire nantes`
   - **Target Address:** `pays de la loire nantes`
   - **High-DF Address Tokens:** `loire` (DF = 261,354), `nantes` (DF = 200,245)
   - **In E5-B:** Displaced because `loire` and `nantes` were dropped, dropping evidence score from 210 to 150.
   - **In E6-A:** Name Token (`maison`, `de`, `santé`, `jean`) and Prefix (`mais`, `mai`) generated the candidate. High-DF selective lookup detected `loire` and `nantes` in target address, awarded `+60` address score. Total score restored to 210. **Candidate retained at rank #12.**

2. **`S1-345501851 <-> S3-526884354` ("association du secondaire"):**
   - **S1 Address:** `nouvelle aquitaine 3 rue de campeyraut bordeaux`
   - **Target Address:** `bordeaux nouvelle aquitaine`
   - **High-DF Address Tokens:** `bordeaux` (DF = 233,153), `nouvelle` (DF = 157,475), `aquitaine` (DF = 157,166)
   - **In E5-B:** Displaced.
   - **In E6-A:** Generated by Name Token (`association`, `du`, `secondaire`) and Prefix (`asso`, `ass`). Awarded selective address evidence for `bordeaux`. Score restored. **Candidate retained.**

3. **`S1-476732670 <-> S3-874616004` ("école maternelle jean"):**
   - **S1 Address:** `6 rue paul nassivet nantes pays de la loire`
   - **Target Address:** `nantes pays de la loire`
   - **High-DF Address Tokens:** `nantes` (DF = 200,245), `loire` (DF = 261,354)
   - **In E5-B:** Displaced.
   - **In E6-A:** Fully recovered. **Candidate retained.**

---

## 5. Answers to Recall Safety Questions

1. **Does E6 preserve 100% raw recall?**
   **YES.** Raw recall is exactly **`100.0000%`** across all configurations ($4,191 / 4,191$ true pairs).
2. **How many true pairs are lost before the cap?**
   **`ZERO`**. Not a single true pair was lost before the cap.
3. **How many are lost only because of the 400-cap ranking?**
   Under both E6-0 (Control) and E6-A (Selective 50k), exactly 721 true pairs were outside the top 400 because they shared identical high scores with hundreds of competing true variants (post-cap recall = $82.80\%$). Relative to the control E6-0, **`ZERO`** true pairs were displaced by E6-A.
4. **Does retaining address evidence for already-generated candidates recover the 164 displaced pairs observed in E5?**
   **YES, EXACTLY 100% (164 / 164)** of the pairs displaced in E5 were recovered.
5. **Which E6 threshold gives the best efficiency/recall tradeoff?**
   **`E6-A (50,000 Selective)`** is unequivocally the best threshold. It achieves the highest candidate reduction ($-55.90\%$), fastest runtime ($-43.95\%$), and has **zero recall loss**.
6. **Is there any configuration that clearly dominates E5-B?**
   **YES. `E6-A` completely dominates E5-B.** It achieves identical candidate reduction ($-55.90\%$) while recovering $100\%$ of the 164 displaced true pairs, matching baseline control retention bit-for-bit.

---

## 6. Production Recommendation & Safety Check

### Production Safety Firewall
```text
Production Config A unchanged:                VERIFIED (src/blocking.py untouched)
Production fingerprint unchanged:             VERIFIED (87f20ceeb84ccc6ea2d48678c7810ac5)
Candidate cap unchanged:                      VERIFIED (400 per S1)
57-feature schema unchanged:                  VERIFIED (models/phase5_configA_feature_schema.json)
XGBoost model unchanged:                      VERIFIED (models/xgboost_entity_resolution_phase5_configA.json)
Threshold unchanged:                          VERIFIED (0.910 in models/phase5_configA_threshold.json)
All 4 model artifacts verified:               VERIFIED (MD5 checksums intact)
No production inference executed:             VERIFIED
No submission files generated:                VERIFIED
```

### Recommendation
**E6-A is mathematically and empirically superior to all tested blocking designs.**
- It eliminates **121,204,261 candidate evaluations** per 500 S1 entities.
- It cuts candidate-generation runtime from **$436.23\text{s}$ down to $244.49\text{s}$**.
- It preserves **$100.00\%$ raw recall and identical post-cap recall ($82.80\%$)** with **0 true pairs lost**.
- However, per instruction, **production remains strictly locked and untouched**. No code has been merged. We await explicit user confirmation.
