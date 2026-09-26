# Phase 5: Final Config A Training, XGBoost Retraining & Threshold Optimization Report

## Executive Summary
Following the empirical results of the Final Blocking Tournament, the **Config A — Team Baseline** blocking architecture was officially locked as the production blocker.

The complete training and validation pipeline was rebuilt using **only** Config A channels:
1. **Country**
2. **Name Token**
3. **Name Prefix-3**
4. **Name Prefix-4**
5. **Address Token**

An XGBoost pair classifier was retrained from scratch on the Config A candidate pairs, followed by independent decision threshold re-optimization for validation Macro F0.5.

---

## 1. Pipeline & Architecture Configuration
- **Locked Blocker**: `CONFIG_A_TEAM_BASELINE`
- **Active Channels**: `country`, `name_token`, `name_prefix3`, `name_prefix4`, `address_token`
- **Config Fingerprint**: `87f20ceeb84ccc6ea2d48678c7810ac5`
- **Source 1 Entities**: 2,500
- **Target Data Pool**: 38,652 (19,190 S2, 19,462 S3)
- **Candidate Safety Cap**: 400 candidates / S1
- **Feature Count**: 57 (Authoritative schema preserved with 0 NaNs / 0 Infs)
- **XGBoost Objective**: `binary:logistic` (`tree_method='hist'`)
- **Scale Pos Weight**: 15.0688 (dynamic ratio: negatives / positives)
- **Best Iteration**: 299

---

## 2. Candidate Generation & Recall Metrics
- **Raw Candidate Pairs**: 2,604,211
- **Average Candidates / S1**: 1041.68
- **Median Candidates / S1**: 756.0
- **Max Candidates / S1**: 4,912
- **Zero-Candidate S1 Count**: 0
- **Raw Candidate Recall**: **99.95%** (8,670 / 8,674 true links, 4 missed)

### Post-Cap Safety Results (Cap = 400):
- **Post-Cap Candidates**: 823,860
- **Overflowing S1 Entities**: 1,672
- **True Pairs Retained After Cap**: 8,605 (69 missed)
- **Post-Cap Candidate Recall**: **99.20%**
- **Cap Recall Retention**: **99.25%**

---

## 3. Training Pairs & Split Isolation
- **Training Pair Construction**:
  - Positives: 8,605
  - Hard Negatives (70% top evidence): 90,352
  - Random Negatives (30% diversity): 38,723
  - Total Negative Pool: 129,075
- **Entity-Level 80/20 Split**:
  - Train S1 Entities: 2,000 (Pairs: 110,698, Positives: 6,889, Negatives: 103,809)
  - Validation S1 Entities: 500 (Pairs: 26,982, Positives: 1,716, Negatives: 25,266)
  - S1 Entity Leakage: **0 (Strict Isolation Verified)**

---

## 4. Threshold Optimization on Validation Set
Swept validation thresholds to maximize **Macro F0.5**:

| Threshold | Macro F0.5 | Macro Precision | Macro Recall | Pairwise Precision | Pairwise Recall | TP | FP | FN | Matches |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.80 | 0.9851 | 98.84% | 98.39% | 98.77% | 98.43% | 1689 | 21 | 27 | 1710 |
| 0.85 | 0.9855 | 98.93% | 98.29% | 98.94% | 98.37% | 1688 | 18 | 28 | 1706 |
| 0.88 | 0.9858 | 98.97% | 98.29% | 99.00% | 98.37% | 1688 | 17 | 28 | 1705 |
| 0.90 | 0.9859 | 99.02% | 98.16% | 99.12% | 98.19% | 1685 | 15 | 31 | 1700 |
| 0.91 | 0.9863 | 99.10% | 98.07% | 99.23% | 98.08% | 1683 | 13 | 33 | 1696 |
| 0.92 | 0.9860 | 99.10% | 97.99% | 99.23% | 97.96% | 1681 | 13 | 35 | 1694 |
| 0.93 | 0.9860 | 99.13% | 97.89% | 99.35% | 97.84% | 1679 | 11 | 37 | 1690 |
| 0.94 | 0.9860 | 99.13% | 97.89% | 99.35% | 97.84% | 1679 | 11 | 37 | 1690 |
| 0.95 | 0.9859 | 99.13% | 97.84% | 99.35% | 97.79% | 1678 | 11 | 38 | 1689 |
| 0.96 | 0.9858 | 99.17% | 97.70% | 99.41% | 97.61% | 1675 | 10 | 41 | 1685 |
| 0.97 | 0.9849 | 99.17% | 97.48% | 99.40% | 97.32% | 1670 | 10 | 46 | 1680 |
| 0.98 | 0.9861 | 99.42% | 97.18% | 99.52% | 97.03% | 1665 | 8 | 51 | 1673 |
| 0.99 | 0.9850 | 99.53% | 96.54% | 99.70% | 96.27% | 1652 | 5 | 64 | 1657 |

### Optimal Threshold Selection:
- **Locked Optimal Threshold**: **0.910**
- **Validation Macro F0.5**: **0.9863**
- **Validation Macro Precision**: **99.10%**
- **Validation Macro Recall**: **98.07%**
- **Pairwise Precision**: **99.23%**
- **Pairwise Recall**: **98.08%**
- **Confusion Matrix**: TP = 1683.0, FP = 13.0, FN = 33.0

---

## 5. Validation Diagnostics

### A. Singleton / Zero-Match Diagnostics
- True zero-match S1 entities: 33
- Correctly predicted zero-match: 31
- False positive predictions on zero-match entities: 2

### B. Collision Diagnostics
- Total Predictions: 1696
- Colliding Candidates: 1 (0.06%)
- S2 Collisions: 1, S3 Collisions: 0

### C. Probability Calibration
- **Brier Score**: 0.00169
- **Expected Calibration Error (ECE)**: 0.00131

### D. Representative Error Examples

#### False Positives (Predicted Match, Ground Truth = 0):
- **Pair**: `S1-881848076` -> `S2-820558687` (S2)
  - S1: `ozl india` | `plot no 155 sahid nagar bhubaneswar khordha orissa` | `india`
  - Target: `tavozeph` | `plot no 6 143 bhubaneswar khordha orissa` | `india`
  - Score: 0.9218 | Blocking Keys: 1 | Evidence Score: 240.0
- **Pair**: `S1-333207746` -> `S3-692514298` (S3)
  - S1: `townsend associates` | `9113 sycamore leaf drive fort worth tx` | `us`
  - Target: `townsend iron works` | `009114 sycamore leaf dr fort worth texas` | `us`
  - Score: 0.9992 | Blocking Keys: 4 | Evidence Score: 390.0
- **Pair**: `S1-881015038` -> `S3-870808310` (S3)
  - S1: `bharat industries private limited` | `b 3 276 chitrakoot vaishali nagar jaipur rajasthan` | `india`
  - Target: `bharat industries` | `` | `india`
  - Score: 0.9938 | Blocking Keys: 3 | Evidence Score: 220.0
- **Pair**: `S1-702336597` -> `S2-439359291` (S2)
  - S1: `physical therapy coastal physicians llc` | `4219 110th street village of pleasant prairie wi` | `us`
  - Target: `physical therapy coastal` | `` | `us`
  - Score: 0.9813 | Blocking Keys: 3 | Evidence Score: 290.0
- **Pair**: `S1-631152952` -> `S2-672734557` (S2)
  - S1: `first lutheran church` | `595 hicks road unit e nashville tn` | `us`
  - Target: `first lutheran church trading` | `` | `us`
  - Score: 0.9893 | Blocking Keys: 3 | Evidence Score: 290.0

#### False Negatives (Missed True Link, Ground Truth = 1):
- **Pair**: `S1-795631320` -> `S2-526762534` (S2)
  - S1: `smart trading private limited` | `flat no 1 22 vijaya enclave nb bangalore south bangalore karnataka` | `india`
  - Target: `ಸ್ಮಾರ್ಟ್ ಟ್ರೇಡಿಂಗ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್` | `fdat no 1 22 bangalore south bangalore ಕರ್ನಾಟಕ` | `india`
  - Score: 0.0132 | Blocking Keys: 1 | Evidence Score: 120.0
- **Pair**: `S1-900870097` -> `S3-962132902` (S3)
  - S1: `gujarat technologies private limited` | `cabin no 108 first floor tower 1 steller crest business center it park c 25 sector 6 2 noida gautam buddha nagar uttar pradesh` | `india`
  - Target: `गुजरात टेक्नोलॉजीज प्राइवेट लिमिटेड` | `cabin no 108 noida gautam buddha nagar up` | `india`
  - Score: 0.8968 | Blocking Keys: 1 | Evidence Score: 240.0
- **Pair**: `S1-91085118` -> `S2-581064933` (S2)
  - S1: `eastern trading private limited` | `flat no 204 building 47 sector 54 56 58 seawood thane maharashtra` | `india`
  - Target: `ईस्टर्न ट्रेडिंग प्राइवेट लिमिटेड` | `thane maharashtra no 04 thane` | `india`
  - Score: 0.0082 | Blocking Keys: 1 | Evidence Score: 120.0
- **Pair**: `S1-483281264` -> `S3-246582348` (S3)
  - S1: `louisa s group` | `599 mountain shadow lane maryville tn` | `us`
  - Target: `lrusir s group` | `599d mountain shadow lane po box 2895 maryille tennessee` | `us`
  - Score: 0.9052 | Blocking Keys: 1 | Evidence Score: 120.0
- **Pair**: `S1-360188878` -> `S3-875542917` (S3)
  - S1: `good constructions` | `c o jaharoom bee qureshi near jobat tent house rajiv colony deodara mandla madhya pradesh` | `india`
  - Target: `गुड कंस्ट्रक्शंस` | `h no 78 c o jaharoom bee qureshi near jobat tent house mandla मध्य प्रदेश` | `india`
  - Score: 0.6636 | Blocking Keys: 1 | Evidence Score: 360.0

---

## 6. Artifact Inventory
- **Model**: [`models/xgboost_entity_resolution_phase5_configA.json`](models/xgboost_entity_resolution_phase5_configA.json)
- **Feature Schema**: [`models/phase5_configA_feature_schema.json`](models/phase5_configA_feature_schema.json)
- **Threshold**: [`models/phase5_configA_threshold.json`](models/phase5_configA_threshold.json)
- **Metadata**: [`models/phase5_configA_metadata.json`](models/phase5_configA_metadata.json)
- **Metrics JSON**: [`experiments/phase5_configA_validation_metrics.json`](experiments/phase5_configA_validation_metrics.json)

---

## 7. Test Firewall Status
- **Test Candidate Generation**: NOT RUN
- **Test Feature Generation**: NOT RUN
- **Test Model Scoring**: NOT RUN
- **Test Threshold Optimization**: NOT RUN
- **Test Matching Results (matching_results.tsv)**: NOT GENERATED
- **Submission**: NOT GENERATED
- **Test Ground Truth Usage**: NONE (No test labels exist or were used)
