# PHASE E7 — RECALL-PRESERVING DF-CAP EXPERIMENT REPORT

> [!IMPORTANT]
> **Production Architecture Remains Strictly Locked & Untouched**
> This experiment was executed completely in isolation using [`scratch/run_experiment_opt_e7.py`](file:///c:/Users/matee/OneDrive/Desktop/Amazon%20ML/business_entity_resolution/scratch/run_experiment_opt_e7.py).
> - Production `BLOCKING_CONFIG`: **Unchanged**
> - Production Config A channels (Country, Name Token, Prefix-3, Prefix-4, Address Token): **Unchanged**
> - Production Config Fingerprint: **`87f20ceeb84ccc6ea2d48678c7810ac5`** (Verified)
> - Candidate safety cap ($400$): **Unchanged**
> - 57-feature schema & XGBoost model: **Unchanged**
> - Decision threshold ($0.910$): **Unchanged**
> - No production inference executed, no test submissions generated.

---

## 1. Executive Summary & Verdict

Phase E7 tested whether combining selective address-token DF filtering with **combined-evidence ranking** and a **per-S1 uncapped retrieval fallback** could achieve the strict target floor:
$$\text{Post-Cap Recall} \ge 99.00\%$$

### Authoritative Determination

> **"RECALL TARGET FAILED"**
> - Across all configurations (E7-0 through E7-D), post-cap recall reached exactly **`82.7965%` (3,470 / 4,191)**.
> - Because post-cap recall fell short of the $99.00\%$ threshold, **Scale Validation to 5,000 S1 entities was NOT executed**, per instruction.
> - Candidate generation remains strictly locked; production is untouched.

---

## 2. E7 Comprehensive Comparison Table

| Configuration | Address DF Cap | Per-S1 Fallback | Raw Candidates | Average Raw / S1 | Raw True Captured | Raw Recall | Post-Cap Candidates | True Pairs Retained | Post-Cap Recall | Zero-Cand S1 | Runtime (s) | Throughput (S1/s) | Peak RSS | Candidate Reduc. vs Control | Runtime Reduc. vs Control | True Pairs Lost vs Control | True Pairs Gained vs Control | Gate Status |
|:---|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| **Control (E6-0)** | *None* | *N/A* | 216,808,077 | 433,616.15 | 4,191 | 100.00% | 200,000 | 3,470 | 82.7965% | 0 | 233.74s | 2.14 | 1,340.0 MB | 0.00% | 0.00% | 0 | 0 | **CONTROL** |
| **E7-0 (E6-A baseline)** | 50,000 | NO | **95,603,816** | 191,207.63 | 4,191 | 100.00% | 200,000 | 3,470 | 82.7965% | 0 | **159.99s** | **3.13** | 1,483.9 MB | **-55.90%** | **-31.55%** | **0** | **0** | ❌ **RECALL TARGET FAILED** |
| **E7-A** | 50,000 | YES | **95,603,816** | 191,207.63 | 4,191 | 100.00% | 200,000 | 3,470 | 82.7965% | 0 | 173.97s | 2.87 | 1,505.1 MB | **-55.90%** | **-25.57%** | **0** | **0** | ❌ **RECALL TARGET FAILED** |
| **E7-B** | 100,000 | YES | 110,311,304 | 220,622.61 | 4,191 | 100.00% | 200,000 | 3,470 | 82.7965% | 0 | 184.61s | 2.71 | 1,501.7 MB | **-49.12%** | **-21.02%** | **0** | **0** | ❌ **RECALL TARGET FAILED** |
| **E7-C** | 150,000 | YES | 130,783,469 | 261,566.94 | 4,191 | 100.00% | 200,000 | 3,470 | 82.7965% | 0 | 204.37s | 2.45 | 1,511.6 MB | **-39.68%** | **-12.56%** | **0** | **0** | ❌ **RECALL TARGET FAILED** |
| **E7-D** | 200,000 | YES | 186,764,781 | 373,529.56 | 4,191 | 100.00% | 200,000 | 3,470 | 82.7965% | 0 | 252.39s | 1.98 | 1,458.9 MB | **-13.86%** | *-7.98%* | **0** | **0** | ❌ **RECALL TARGET FAILED** |

---

## 3. Detailed Failure Mechanism Analysis

The independent investigation revealed why post-cap recall remained invariant at **$82.80\%$** and could not reach the $99.00\%$ target:

### 1. Zero Blocking Misses (Raw Recall is 100.00%)
In all configurations, **`4,191 out of 4,191`** ground-truth true pairs were successfully generated in the candidate universe before the cap. The loss of $721$ true pairs occurs **strictly after the top-400 ranking cutoff**.

### 2. The 400-Candidate Cap is a Mathematical Ceiling for High-Frequency Entities
Two S1 entities in the benchmark have **more than 400 true matches** in France:
- **`S1-202627015` ("nantes club"):** has **428 true matches** in the target set.
- **`S1-475756218` ("lille club"):** has **420 true matches** in the target set.

Because the candidate cap is strictly $400$ per S1 entity, it is mathematically impossible to retain more than 400 candidates for each. Even with a theoretical "oracle" ranker that places true matches above all non-matches, at least $(428 - 400) + (420 - 400) = 48$ true matches would be lost, setting the absolute theoretical ceiling on this benchmark to $4,143 / 4,191 = 98.85\% < 99.00\%$.

### 3. Tie-Breaking Saturation on Generic Business Names
In practice, generic name queries saturate the top-400 bucket with candidates tied at the exact same combined score:
- For `nantes club`, hundreds of targets match `NameToken('nantes')` (70) + `NameToken('club')` (70) + `Prefix4('nant')` (50) + `Prefix3('nan')` (30) + `AddressToken` (60) = **Score 280**.
- Exactly $400$ candidates tied at score $280$. Deterministic lexical tie-breaking (`target_lex_rank`, by entity ID) selected the first 400, retaining 98 true matches and displacing 330.
- For `lille club`, exactly $400$ candidates tied at score $280$, retaining 113 true matches and displacing 307.

Together, **just these two S1 entities account for $637$ of the $721$ lost true pairs ($88.3\%$)**:

| S1 Entity ID | Business Name | Ground-Truth True Pairs | Retained in Top 400 | Displaced by Cap | Loss Mechanism |
|:---|:---|---:|---:|---:|:---|
| `S1-202627015` | nantes club | 428 | 98 | 330 | Cap saturation at score 280 (lexical tie-breaking) |
| `S1-475756218` | lille club | 420 | 113 | 307 | Cap saturation at score 280 (lexical tie-breaking) |
| `S1-516261021` | maison de santé jean | 85 | 16 | 69 | Cap displacement by other identical-name variants |
| `S1-533056362` | pessac service sarl | 9 | 3 | 6 | Cap displacement by higher-scoring local address pairs |
| `S1-476732670` | école maternelle jean | 85 | 80 | 5 | Cap displacement by other identical-name variants |
| `S1-901870264` | maison de santé francois | 8 | 4 | 4 | Cap displacement by other identical-name variants |
| **All Other 494 S1 Entities** | *(494 distinct entities)* | **3,156** | **3,156** | **0** | **100.00% Post-Cap Recall** |
| **Total** | — | **4,191** | **3,470** | **721** | **Overall Post-Cap Recall: 82.80%** |

### 4. Why Per-S1 Fallback Did Not Alter Results
The per-S1 fallback was designed to trigger if selective DF filtering caused a channel or an S1 entity to generate zero candidates. In the benchmark:
- Every single one of the 500 S1 entities generated candidates through Name Token, Prefix-4, or Prefix-3 channels.
- Zero S1 entities had $0$ candidates.
- The 6 S1 entities losing true pairs already generated over 100,000 raw candidates each; their loss was entirely due to the downstream top-400 ranking cutoff, not a lack of candidate generation.
- Consequently, `fallback_s1_count = 0` across all configurations.

---

## 4. Scale Validation Decision

Per the explicit instructions:
> *"If and ONLY IF the selected configuration achieves: post-cap recall >= 99%, then run: 5,000 S1 entities... If the 5,000 sample falls below 99%: STOP. Do not proceed to 50K."*

**Decision: STOP.**
Scale validation to 5,000 S1 entities is **NOT** executed because no configuration met the $99.00\%$ recall floor.

---

## 5. Production Recommendation & Architecture Status

```text
Production Config A:                LOCKED (src/blocking.py untouched)
Production Fingerprint:             LOCKED (87f20ceeb84ccc6ea2d48678c7810ac5)
Candidate Safety Cap:               LOCKED (400 per S1)
Model Artifacts & 57-Feature Schema:LOCKED (Untouched)
Decision Threshold:                 LOCKED (0.910)
Test Inference Executed:            NO
Submission Files Generated:         NO
```

### Final Conclusion
While selective DF filtering (E6-A / E7-0) reduces candidate volume by **$55.90\%$** with **$0$ true pairs lost relative to control**, achieving a post-cap recall of $\ge 99\%$ is impossible under a hard cap of $400$ when individual generic query entities have $>400$ true targets. Production remains strictly locked at its validated baseline.
