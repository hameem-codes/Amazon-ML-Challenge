# Phase 4B: XGBoost Pair Classifier & Macro F0.5 Optimization Report

- **Pipeline Version:** 4.0.0
- **Config Fingerprint:** `af143f61867ee2fcd7238f76ea4eb09f`
- **Total Runtime:** 511.9s | **Peak RAM:** 988.8 MB
- **Status:** MODEL & THRESHOLD LOCKED (NO TEST INFERENCE RUN)

## 1. Training & Candidate Data Summary

- **Source 1 entities evaluated:** 2,500
- **Target records pool (S2+S3):** 38,652 (19,190 S2, 19,462 S3)
- **Ground-Truth True Pairs:** 8,674
- **Candidate Pairs Before Cap:** 1,568,447
- **Candidate Pairs After Cap (max=350/S1):** 670,783
- **Overflowing S1 Entities:** 1435
- **Candidate Recall (Blocking):** 90.97%
- **Candidate Recall (Post-Cap):** 90.12% (Cap retention: 99.06%)
- **Training Pairs Constructed:** 125,072 (Pos: 7,817, Neg: 117,255)
- **Hard Negatives Selected:** 82,078 (Top 70% by blocking evidence score)
- **Random Negatives Selected:** 35,177

## 2. Strict Entity-Level Train / Validation Split

- **Train S1 Entities:** 1,996 (101,185 pairs; Pos: 6,191, Neg: 94,994)
- **Validation S1 Entities:** 499 (23,887 pairs; Pos: 1,626, Neg: 22,261)
- **Leakage Verification:** `Train S1 ∩ Val S1 = ∅` (Strictly verified)
- **Dynamic `scale_pos_weight`:** `15.3439`

## 3. Feature Matrix & Numerical Safety

- **Feature Count:** 57
- **Feature Groups:** Name (14), Address (10), Country (5), Directional Missingness (8), Structural (10), Blocking Evidence (6), Source (2), TF-IDF Cosine (2)
- **Total NaNs:** 0 | **Total Infs:** 0
- **Transductive TF-IDF:** Fitted on unlabeled text union across train and test without ground-truth labels.

## 4. XGBoost Model Configuration

```json
{
  "objective": "binary:logistic",
  "eval_metric": "logloss",
  "tree_method": "hist",
  "learning_rate": 0.05,
  "max_depth": 6,
  "n_estimators": 300,
  "random_state": 42,
  "scale_pos_weight": 15.343886286544985,
  "subsample": 0.85,
  "colsample_bytree": 0.85,
  "min_child_weight": 3,
  "reg_alpha": 0.1,
  "reg_lambda": 1.0,
  "early_stopping_rounds": 30
}
```

## 5. Probability Calibration Diagnostics

- **Brier Score:** `0.00105`
- **Expected Calibration Error (ECE):** `0.00081`

```text
bin_range  count  mean_predicted_prob  empirical_match_rate  absolute_gap
  0.0-0.1  22229             0.000169              0.000270      0.000101
  0.1-0.2     10             0.136642              0.100000      0.036642
  0.2-0.3      3             0.244227              0.333333      0.089107
  0.3-0.4      2             0.339244              0.000000      0.339244
  0.4-0.5      5             0.431053              0.200000      0.231053
  0.5-0.6      6             0.546984              0.500000      0.046984
  0.6-0.7      4             0.652462              0.500000      0.152462
  0.7-0.8      3             0.772630              0.333333      0.439296
  0.8-0.9      3             0.834901              0.000000      0.834901
  0.9-1.0   1622             0.999349              0.993218      0.006131
```

## 6. Threshold Sweep & Macro F0.5 Optimization

```text
 threshold  macro_f0_5  macro_precision  macro_recall  pairwise_f0_5  pairwise_precision  pairwise_recall   tp  fp  fn  predicted_matches  zero_match_s1_count
      0.10    0.942861         0.976119      0.915025       0.980867            0.977081         0.996310 1620  38   6               1658                   34
      0.20    0.946078         0.982465      0.914357       0.985033            0.982403         0.995695 1619  29   7               1648                   35
      0.30    0.946301         0.982799      0.913956       0.985864            0.983587         0.995080 1618  27   8               1645                   35
      0.40    0.946910         0.983500      0.913956       0.986826            0.984784         0.995080 1618  25   8               1643                   35
      0.50    0.947991         0.985170      0.913455       0.988628            0.987179         0.994465 1617  21   9               1638                   35
      0.60    0.948722         0.986406      0.912220       0.989698            0.988971         0.992620 1614  18  12               1632                   35
      0.70    0.948860         0.987208      0.911151       0.990415            0.990172         0.991390 1612  16  14               1628                   35
      0.75    0.950864         0.989212      0.911151       0.990902            0.990781         0.991390 1612  15  14               1627                   36
      0.80    0.952659         0.991216      0.910650       0.991263            0.991385         0.990775 1611  14  15               1625                   37
      0.85    0.953566         0.992285      0.910650       0.992239            0.992606         0.990775 1611  12  15               1623                   37
      0.88    0.954138         0.992953      0.910650       0.992729            0.993218         0.990775 1611  11  15               1622                   37
      0.90    0.954138         0.992953      0.910650       0.992729            0.993218         0.990775 1611  11  15               1622                   37
      0.92    0.954054         0.992953      0.910364       0.992602            0.993214         0.990160 1610  11  16               1621                   37
      0.94    0.954190         0.993287      0.909696       0.992965            0.993823         0.989545 1609  10  17               1619                   37
      0.96    0.953943         0.993955      0.907625       0.992947            0.994424         0.987085 1605   9  21               1614                   37
      0.98    0.953972         0.994623      0.905654       0.992930            0.995028         0.984625 1601   8  25               1609                   37
```

## 7. Locked Model & Threshold Selection

- **Selected Classification Threshold:** `0.970`
- **Validation Macro F0.5:** `0.9544`
- **Validation Macro Precision:** `99.46%`
- **Validation Macro Recall:** `90.72%`
- **Pairwise F0.5:** `0.9933` (Precision: `99.50%`, Recall: `98.65%`)
- **Confusion Matrix:** TP = 1604.0, FP = 8.0, FN = 22.0
- **Predicted Matches:** 1,612.0 across validation S1 entities
- **Zero-Match S1 Count:** 37.0 / 499 (preserves correct singletons/empty predictions)

## 8. Many-to-Many Collision Diagnostics

- **Total Predicted Matches:** 1,612
- **Unique S1 Entities with Matches:** 462
- **Unique Candidates Predicted:** 1611
- **Colliding Candidates (matched to >1 S1):** 1 (0.06%)
  * S2 Collisions: 1
  * S3 Collisions: 0

## 9. Representative Error Analysis

### False Positive Examples (Predicted Match, Ground Truth = 0):
- **Pair:** `S1-654697159 -> S2-690526024 (S2)` | **p:** `0.9984`
  - S1: `sector co of noida` | `d 72 sector 55 noida gautam budh nagar noida gautam buddha nagar uttar pradesh`
  - Cand: `noida breeding limited services` | `unit no 1 noida gautam buddha nagar uttar pradesh`
- **Pair:** `S1-630839433 -> S2-849565658 (S2)` | **p:** `0.9879`
  - S1: `chiropractic associates inc` | `511 55th street renton wa`
  - Cand: `chiropractic inc` | ``
- **Pair:** `S1-238331147 -> S3-874715964 (S3)` | **p:** `0.9958`
  - S1: `sree it limited` | `plot no 21 office no 101 106 first floor block g vardhman chamber ii community centre vikaspuri new delhi west delhi delhi`
  - Cand: `artemis it ltd` | `plot no 45 new delhi south west delhi dl`
- **Pair:** `S1-846870843 -> S2-531536762 (S2)` | **p:** `0.9993`
  - S1: `surgical physicians llc` | `75 cottonwood road wake forest nc`
  - Cand: `surgical physicians inc` | ``
- **Pair:** `S1-817683544 -> S2-118275942 (S2)` | **p:** `0.9984`
  - S1: `mhs sangh` | `5sl 07vill barolanoidag b nagar noida gautam buddha nagar uttar pradesh`
  - Cand: `mhs sangh ventures` | `5sl 18vill barolanoidag b naga noida उत्तर प्रदेश`

### False Negative Examples (Predicted Non-Match, Ground Truth = 1):
- **Pair:** `S1-252242682 -> S2-84842746 (S2)` | **p:** `0.0491`
  - S1: `aggie e nagle l c s w` | `11900 54th avenue plymouth mn`
  - Cand: `aggieenagle com` | `54nd avenue minneaplis mn`
- **Pair:** `S1-429691516 -> S3-948989150 (S3)` | **p:** `0.0746`
  - S1: `led mail pvt ltd` | `tamil nadu kk ngr 187 chennai sv lingam stkk ngr chennai kk ngr chennai`
  - Cand: `led pvt ltd center` | `plot 225 187 chennai tn`
- **Pair:** `S1-352234501 -> S2-379384733 (S2)` | **p:** `0.9477`
  - S1: `clean investment services` | `34 abbott square birmingham al`
  - Cand: `clean investment center` | ``
- **Pair:** `S1-467198990 -> S3-367123111 (S3)` | **p:** `0.2203`
  - S1: `national kensington` | `114 3rd street west terre haute in`
  - Cand: `national service` | ``
- **Pair:** `S1-410040347 -> S3-186997268 (S3)` | **p:** `0.1892`
  - S1: `temple synagogue` | `10202 w maple road omaha ne`
  - Cand: `temple partners` | ``

## 10. Locked Artifacts & Verification

- Model: `models/xgboost_entity_resolution_phase4b.json`
- Feature Schema: `models/phase4b_feature_schema.json`
- Threshold: `models/phase4b_threshold.json`
- Metadata: `models/phase4b_metadata.json`

### Absolute Test Inference Firewall Confirmation:
- Test inference was NOT executed.
- No test records were evaluated by the classifier.
- No submission files were produced.
