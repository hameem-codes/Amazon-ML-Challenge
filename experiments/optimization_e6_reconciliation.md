# PHASE E6-A AUDIT & RECONCILIATION REPORT

> [!IMPORTANT]
> **Production Firewall Status: STRICTLY LOCKED & VERIFIED**
> - Production `BLOCKING_CONFIG`: **Untouched**
> - Production Blocking Channels: **Untouched**
> - Production Config Fingerprint: **`87f20ceeb84ccc6ea2d48678c7810ac5`** (Verified)
> - Candidate Safety Cap ($400$): **Untouched**
> - 57-Feature Schema & XGBoost Model: **Untouched**
> - Decision Threshold ($0.910$): **Untouched**
> - No production inference executed, no submissions generated.

---

## 1. Authoritative Final Result for E6-A

An independent, ground-up recalculation was executed on the full 1,434,993 France target dataset and 500 S1 records by [`scratch/reconcile_e6.py`](file:///c:/Users/matee/OneDrive/Desktop/Amazon%20ML/business_entity_resolution/scratch/reconcile_e6.py). The resulting candidate pairs were persisted directly to [`scratch/candidates_e6_0.tsv.gz`](file:///c:/Users/matee/OneDrive/Desktop/Amazon%20ML/business_entity_resolution/scratch/candidates_e6_0.tsv.gz) and [`scratch/candidates_e6_a.tsv.gz`](file:///c:/Users/matee/OneDrive/Desktop/Amazon%20ML/business_entity_resolution/scratch/candidates_e6_a.tsv.gz).

```yaml
E6-A final raw recall: 100.0000% (4,191 / 4,191 reference true pairs)
E6-A final post-cap recall: 82.7965% (3,470 / 4,191 reference true pairs)
E6-A final raw candidate count: 95,603,816
E6-A final post-cap candidate count: 200,000
E6-A final candidates lost vs control: 0
E6-A final true pairs lost vs control: 0
E6-A final true pairs gained vs control: 0
E6-A final runtime: 153.54s (generation only) / 264.77s (initial full benchmark)
E6-A final peak RSS: 1,444.5 MB
```

### Formal Determination

**A) "82.80% is the authoritative final post-cap recall."**

---

## 2. Independent Set Comparison & Mathematical Reconciliation

The perceived contradiction arose because the control baseline in E5 (`E5-0`) had a post-cap recall of **$47.34\%$** ($1,984 / 4,191$), whereas the control baseline in E6 (`E6-0`) had a post-cap recall of **$82.80\%$** ($3,470 / 4,191$).

When comparing E6-A against E6-0, the candidate sets are **100% bit-for-bit identical**, and both achieve exactly **$82.80\%$ post-cap recall**.

### Independent Bit-for-Bit Set Operations Table

| Metric | Production Baseline (E0 / E5-0) | E6-0 (Control) | E6-A (Selective 50k) | E6-0 vs E6-A Comparison |
|:---|---:|---:|---:|:---:|
| **Raw Candidate Count** | 216,808,077 | 216,808,077 | **95,603,816** | **-55.90% (-121.2M raw pairs)** |
| **Post-Cap Candidate Count** | 200,000 | 200,000 | **200,000** | Identical ($400 / S1$) |
| **Ground-Truth True Pairs** | 4,191 | 4,191 | **4,191** | Exact name matches |
| **Raw True Pairs Captured** | 4,191 | 4,191 | **4,191** | $100.00\%$ raw recall |
| **Post-Cap True Pairs Captured**| 1,984 | 3,470 | **3,470** | **Identical ($3,470$ pairs)** |
| **Post-Cap Recall** | **47.3395%** | **82.7965%** | **82.7965%** | **Identical (82.80%)** |
| **Candidate Overlap ($E6\_0 \cap E6\_A$)** | — | — | — | **200,000 (100.00%)** |
| **Candidate Symmetric Diff ($E6\_0 \oplus E6\_A$)**| — | — | — | **0** |
| **Candidates Unique to E6-0** | — | — | — | **0** |
| **Candidates Unique to E6-A** | — | — | — | **0** |
| **True Pairs Lost vs E6-0 Control**| — | — | — | **0** |
| **True Pairs Gained vs E6-0 Control**| — | — | — | **0** |

---

## 3. Explicit Answers to Critical Questions

### 1. What EXACT candidate set is used to calculate E6-0 post-cap recall?
The set of 200,000 $(entity\_id\_s1, entity\_id\_cand)$ pairs retained after ranking and applying the top-400 cap per S1 entity in [`scratch/run_experiment_opt_e6.py`](file:///c:/Users/matee/OneDrive/Desktop/Amazon%20ML/business_entity_resolution/scratch/run_experiment_opt_e6.py#L465-L488) under `addr_df_cap = None`.

### 2. What EXACT candidate set is used to calculate E6-A post-cap recall?
The set of 200,000 $(entity\_id\_s1, entity\_id\_cand)$ pairs retained after ranking and applying the top-400 cap per S1 entity in [`scratch/run_experiment_opt_e6.py`](file:///c:/Users/matee/OneDrive/Desktop/Amazon%20ML/business_entity_resolution/scratch/run_experiment_opt_e6.py#L465-L488) under `addr_df_cap = 50,000` with selective address evidence evaluation.

### 3. What EXACT set is used for the "true pairs lost vs E6-0" calculation?
The set difference of true pairs:
$$\text{Lost} = (PostCap_{E6-0} \cap TruePairs) \setminus (PostCap_{E6-A} \cap TruePairs)$$
Both $PostCap_{E6-0} \cap TruePairs$ and $PostCap_{E6-A} \cap TruePairs$ contain the exact same 3,470 true pairs. The set difference is $\emptyset$ (count = $0$).

### 4. What EXACT set is used for the "100% candidate set equality" calculation?
The candidate pair intersection and symmetric difference between the full post-cap candidate sets:
$$PostCap_{E6-0} \cap PostCap_{E6-A} = 200,000$$
$$PostCap_{E6-0} \oplus PostCap_{E6-A} = 0$$
Every single $(entity\_id\_s1, entity\_id\_cand)$ pair in E6-A is identical to E6-0.

### 5. Are these calculations comparing the same candidate-pair universe?
**Yes.** All calculations operate over the exact same universe of 500 France S1 entities and 1,434,993 France targets (Source 2 + Source 3), with candidate pairs represented as unique `(entity_id_s1, entity_id_cand)` tuples.

### 6. Is 82.80% actually the final E6-A post-cap recall?
**Yes.** Exactly $3,470 / 4,191 = 82.7965\%$ ($82.80\%$).

### 7. Is 82.80% actually inherited/copied from E5-B or another intermediate experiment?
**No.** In E5-B, post-cap recall was **$82.51\%$** ($3,458 / 4,191$). The 82.80% in E6-A is independently computed from the candidate outputs of E6-A.

### 8. Was there a later safety-net/fallback/recovery step that changed the candidate set after the 82.80% calculation?
**No.** The candidate set is generated, ranked, capped, and evaluated in a single forward pass. There are no post-processing fallbacks, union merges, or safety-net passes.

### 9. If yes, identify that step precisely and show where it occurs in code.
Not applicable (no post-hoc recovery step exists). The selective address lookup occurs *in-situ* during candidate generation in [`scratch/run_experiment_opt_e6.py:L156-166`](file:///c:/Users/matee/OneDrive/Desktop/Amazon%20ML/business_entity_resolution/scratch/run_experiment_opt_e6.py#L156-L166).

### 10. Does the final E6-A candidate set actually equal E6-0 bit-for-bit?
**YES.**
$$\text{Overlap} = 200,000 / 200,000 \quad (100.00\%)$$
$$\text{Symmetric Difference (XOR)} = 0$$

### 11. Does the final E6-A candidate set contain exactly the same true pairs as E6-0?
**YES.** Both contain exactly the same 3,470 true pairs ($0$ lost, $0$ gained).

### 12. Recalculate post-cap recall independently from the FINAL SAVED candidate sets.
Direct recalculation using [`scratch/reconcile_e6.py`](file:///c:/Users/matee/OneDrive/Desktop/Amazon%20ML/business_entity_resolution/scratch/reconcile_e6.py) from the saved candidate outputs [`scratch/candidates_e6_0.tsv.gz`](file:///c:/Users/matee/OneDrive/Desktop/Amazon%20ML/business_entity_resolution/scratch/candidates_e6_0.tsv.gz) and [`scratch/candidates_e6_a.tsv.gz`](file:///c:/Users/matee/OneDrive/Desktop/Amazon%20ML/business_entity_resolution/scratch/candidates_e6_a.tsv.gz):
- Total ground truth true pairs: **4,191**
- True pairs in `candidates_e6_0.tsv.gz`: **3,470** $\implies \mathbf{82.7965\%}$
- True pairs in `candidates_e6_a.tsv.gz`: **3,470** $\implies \mathbf{82.7965\%}$

---

## 4. Root Cause of the Apparent Contradiction

The source of confusion was the difference in address evidence scoring between the production blocking implementation and the experimental E6 script:

1. **In production [`src/blocking.py`](file:///c:/Users/matee/OneDrive/Desktop/Amazon%20ML/business_entity_resolution/src/blocking.py#L305):**
   Address evidence bonus was awarded cumulatively for *every* matching address token:
   $$\text{cand\_scores}[idx] += 60 \quad (\forall \text{ matching address tokens})$$
   A candidate sharing 5 generic address tokens (`rue`, `saint`, `france`, `paris`, `cedex`) received $+300$ points, displacing exact name matches (which scored $70 + 50 + 30 = 150$) and causing production baseline post-cap recall to drop to **$47.34\%$** ($1,984 / 4,191$).

2. **In experimental [`scratch/run_experiment_opt_e6.py:L140`](file:///c:/Users/matee/OneDrive/Desktop/Amazon%20ML/business_entity_resolution/scratch/run_experiment_opt_e6.py#L140):**
   Address evidence bonus was clamped to $+60$ maximum per candidate:
   $$\text{if not } (f \ \& \ \text{BIT\_ADDR}): \text{cand\_scores}[idx] += 60$$
   This single change prevented address noise from ever scoring above $60$. As a result, in both E6-0 and E6-A, exact name matches (scoring $\ge 150$) cleanly outranked all pure address noise, lifting the post-cap recall to **$82.80\%$** ($3,470 / 4,191$).

3. **Why E6-A is 100% Identical to E6-0:**
   Because all candidates with scores $\ge 70$ were generated by Name Token, Prefix-4, or Prefix-3 channels, eliminating high-DF address tokens from candidate generation ($-55.9\%$ raw candidates) removed only candidates whose score would have been at most $60$.
   For all surviving candidates (score $\ge 70$), the selective address evaluation in E6-A awarded the identical $+60$ bonus that E6-0 awarded. Consequently, the top-400 ranking produced **bit-for-bit identical candidate sets** ($XOR = 0$).
