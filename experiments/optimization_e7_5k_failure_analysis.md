# PHASE E7 — 5K VALIDATION GATE STOP & FAILURE ANALYSIS

> [!CAUTION]
> **GATE TRIGGER: EXECUTION HALTED PER PROTOCOL**
> The preceding Phase E7 experiment explicitly recorded a post-cap recall of **`82.7965%`**, failing the required $\ge 99.00\%$ gate.
> Under the protocol rule:
> *"Run this ONLY if the preceding E6 reconciliation and recall-preserving DF-cap experiment explicitly pass their required gates. DO NOT run this if post-cap recall is below the defined target."*
> **The 5,000 S1 benchmark was NOT launched.** Production remains locked.

---

## 1. Gate Status Summary

| Gate Condition | Requirement | Phase E7 Result | Gate Outcome |
|:---|:---|:---:|:---:|
| **E6 Reconciliation** | Single authoritative post-cap recall | 82.7965% verified bit-for-bit (XOR = 0) | ✅ **PASSED** |
| **Production Firewall** | Fingerprint `87f20ceeb84ccc6ea2d48678c7810ac5` | Verified intact | ✅ **PASSED** |
| **Post-Cap Recall Gate** | Post-cap recall $\ge 99.00\%$ | **`82.7965%`** ($3,470 / 4,191$) | ❌ **FAILED** |
| **5K Validation Prerequisite** | All preceding gates pass | Recall gate failed | ⛔ **EXECUTION HALTED** |

---

## 2. Mathematical Failure Analysis: Why $\ge 99\%$ is Impossible Under Cap 400

The investigation proved that the sub-99% recall is **not** caused by candidate-generation failure, nor by DF filtering:

### 1. Raw Blocking Recall is 100.00%
Across all tested configurations (E6-0, E6-A, E7-0 through E7-D), **`4,191 / 4,191`** ground-truth true pairs were successfully captured in the pre-cap candidate universe. No true pairs were missed by the blocking channels.

### 2. High-Frequency S1 Entity Saturation
In the 500 France S1 evaluation sample, two entities have more than 400 true matches in France:
- **`S1-202627015` ("nantes club"):** has **428 true matches** in the target set.
- **`S1-475756218` ("lille club"):** has **420 true matches** in the target set.

Because the candidate cap is fixed at $400$ per S1 entity:
$$\text{Max Possible Retention for } S1\text{-}202627015 = 400 / 428 \implies \ge 28 \text{ pairs lost}$$
$$\text{Max Possible Retention for } S1\text{-}475756218 = 400 / 420 \implies \ge 20 \text{ pairs lost}$$
$$\text{Theoretical Benchmark Maximum Recall} = \frac{4,191 - 48}{4,191} = \mathbf{98.85\%} < 99.00\%$$

Even with an omniscient oracle ranker that places true matches above all non-matches, **a recall of $\ge 99.00\%$ is mathematically impossible** on this benchmark under candidate cap = 400.

### 3. Score Saturation and Lexical Tie-Breaking
In practice, generic business names generate hundreds of candidates that tie at the maximum possible evidence score ($280$ points: Name Token $70 + 70$ + Prefix-4 $50$ + Prefix-3 $30$ + Address $60$):
- For `nantes club`, all 400 top slots tied at score $280$. The deterministic lexical tie-breaking (`target_lex_rank`, by candidate entity ID) selected the first 400, retaining 98 true matches and displacing 330.
- For `lille club`, all 400 top slots tied at score $280$, retaining 113 true matches and displacing 307.
- Just these two entities account for **$637$ of the $721$ lost true pairs ($88.3\%$)**.

### 4. Per-S1 Fallback Ineffectiveness
The per-S1 fallback triggers when selective filtering leaves an entity with zero candidates or zero address candidates. Because every S1 entity in the benchmark already generated hundreds of candidates through Name Token and Prefix channels, the fallback never activated (`fallback_s1_count = 0`), and could not prevent downstream cap displacement.

---

## 3. Protocol Decision

Per protocol instructions:
> *"If post-cap recall < 99%: STOP. Do not run any further optimization. Produce a failure analysis."*

1. **Execution is STOPPED.**
2. **5,000 S1 validation is NOT launched.**
3. **No production changes are made.** Production remains locked at `87f20ceeb84ccc6ea2d48678c7810ac5`.
