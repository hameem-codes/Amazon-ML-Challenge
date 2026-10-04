# Business Entity Resolution

## Amazon ML Challenge

An end-to-end machine learning system for resolving business entities across multiple independent and noisy data sources.

The system identifies records that represent the same real-world business despite variations in names, addresses, countries, formatting, spelling, missing values, and other inconsistencies.

---

## Table of Contents

* [1. Project Overview](#1-project-overview)
* [2. Problem Definition](#2-problem-definition)
* [3. System Architecture](#3-system-architecture)
* [4. End-to-End Pipeline](#4-end-to-end-pipeline)
* [5. Data Processing Pipeline](#5-data-processing-pipeline)
* [6. Candidate Generation and Blocking](#6-candidate-generation-and-blocking)
* [7. Candidate Safety Layer](#7-candidate-safety-layer)
* [8. Training Data Construction](#8-training-data-construction)
* [9. Feature Engineering](#9-feature-engineering)
* [10. Machine Learning Model](#10-machine-learning-model)
* [11. Threshold Optimization](#11-threshold-optimization)
* [12. Inference Pipeline](#12-inference-pipeline)
* [13. Blocking Experiments](#13-blocking-experiments)
* [14. Model Evaluation](#14-model-evaluation)
* [15. Scalability](#15-scalability)
* [16. Reproducibility](#16-reproducibility)
* [17. Project Structure](#17-project-structure)
* [18. Technology Stack](#18-technology-stack)
* [19. Running the Project](#19-running-the-project)
* [20. Experiments and Documentation](#20-experiments-and-documentation)
* [21. Limitations](#21-limitations)
* [22. Key Engineering Decisions](#22-key-engineering-decisions)

---

# 1. Project Overview

Business Entity Resolution is a record-linkage problem.

The objective is to determine whether two records from different data sources represent the same real-world business.

The project contains three independent sources:

```text
Source 1
Reference / master entities

Source 2
Independent business records

Source 3
Independent business records
```

For every entity in Source 1, the system searches Source 2 and Source 3 for corresponding records.

The difficulty is that the same business may be represented differently.

For example:

```text
Source 1
ACME Technologies Private Limited

Source 2
ACME TECHNOLOGIES PVT LTD

Source 3
Acme Technologies Pvt. Ltd.
```

These records are potentially the same entity even though their raw representations are different.

The system therefore combines deterministic preprocessing, scalable candidate generation, similarity-based feature engineering, and supervised machine learning.

---

# 2. Problem Definition

## 2.1 Input

The system receives multiple tab-separated datasets containing business information such as:

* Business name
* Address
* Country
* Entity identifiers
* Other available attributes

The original input data is preserved and never modified.

## 2.2 Output

For each Source 1 entity, the system produces candidate and predicted matches from Source 2 and Source 3.

Conceptually:

```text
Source 1 Entity
       |
       v
Candidate Search
       |
       +---------- Source 2 candidates
       |
       +---------- Source 3 candidates
       |
       v
ML Matching
       |
       v
Final Entity Matches
```

## 2.3 Core Constraints

The system must simultaneously address:

| Requirement        | Challenge                                                       |
| ------------------ | --------------------------------------------------------------- |
| High recall        | True matches must not be eliminated during candidate generation |
| High precision     | Similar but unrelated businesses must not be incorrectly linked |
| Scalability        | Millions of records cannot be compared pairwise                 |
| Determinism        | Repeated runs should produce reproducible results               |
| Leakage prevention | Validation must not leak entity information                     |
| No external lookup | Matching must rely on the provided datasets                     |

---

# 3. System Architecture

The overall architecture consists of six major layers.

```text
                         BUSINESS ENTITY RESOLUTION
                                   |
                                   v
                  +-------------------------------+
                  |        INPUT DATASETS         |
                  |    Source 1 / Source 2 / 3    |
                  +---------------+---------------+
                                  |
                                  v
                  +-------------------------------+
                  |      DATA NORMALIZATION       |
                  | Unicode / Case / Punctuation  |
                  | Whitespace / Missing Values   |
                  +---------------+---------------+
                                  |
                                  v
                  +-------------------------------+
                  |       BLOCKING LAYER          |
                  | Country / Name / Prefix /      |
                  | Address-based candidate keys   |
                  +---------------+---------------+
                                  |
                                  v
                  +-------------------------------+
                  |    CANDIDATE SAFETY LAYER     |
                  | Evidence ranking / 400 cap    |
                  +---------------+---------------+
                                  |
                                  v
                  +-------------------------------+
                  |     FEATURE ENGINEERING       |
                  | Name / Address / Country /    |
                  | Structure / Blocking / TF-IDF |
                  +---------------+---------------+
                                  |
                                  v
                  +-------------------------------+
                  |       XGBOOST MODEL           |
                  |      Pair Classification      |
                  +---------------+---------------+
                                  |
                                  v
                  +-------------------------------+
                  |     DECISION LAYER             |
                  | Probability / Threshold /     |
                  | Final Match Selection         |
                  +---------------+---------------+
                                  |
                                  v
                  +-------------------------------+
                  |       FINAL OUTPUT            |
                  |     Entity Relationships      |
                  +-------------------------------+
```

---

# 4. End-to-End Pipeline

<details>
<summary><strong>Pipeline Overview</strong></summary>

The complete system follows this sequence:

```text
Raw Data
   |
   v
Data Loading
   |
   v
Deterministic Normalization
   |
   v
Blocking
   |
   v
Candidate Generation
   |
   v
Candidate Deduplication
   |
   v
Candidate Safety / Ranking
   |
   v
Training Pair Construction
   |
   v
57-Dimensional Feature Engineering
   |
   +----------------------+
   |                      |
   v                      v
String Similarity       TF-IDF Similarity
   |                      |
   +----------+-----------+
              |
              v
         XGBoost Model
              |
              v
       Match Probability
              |
              v
      Threshold Decision
              |
              v
       Final Predictions
```

Each stage exists for a specific reason.

The system is intentionally designed as a pipeline rather than a single machine-learning model.

</details>

---

# 5. Data Processing Pipeline

<details>
<summary><strong>5.1 Data Loading</strong></summary>

The datasets are stored as TSV files.

Tabs are used as the delimiter because several fields can contain commas, particularly addresses and identifier lists.

The data loading layer is responsible for:

* Reading the datasets
* Preserving original columns
* Handling missing values
* Maintaining entity identifiers
* Preparing data for preprocessing

</details>

<details>
<summary><strong>5.2 Deterministic Normalization</strong></summary>

Raw text is normalized before it is used for matching.

The normalization pipeline includes:

```text
Raw String
    |
    v
Unicode Normalization
    |
    v
Case Normalization
    |
    v
Punctuation / Symbol Handling
    |
    v
Whitespace Normalization
    |
    v
Missing Value Handling
    |
    v
Normalized String
```

For example:

```text
"ACME, Ltd."
        |
        v
"acme ltd"
```

The purpose is not to determine whether two businesses are equal.

Instead, normalization creates a consistent representation that allows later stages to compare records more effectively.

Implementation:

`src/normalize.py`

</details>

---

# 6. Candidate Generation and Blocking

Blocking is the main scalability mechanism of the system.

## 6.1 Why Blocking Is Necessary

Assume:

```text
Source 1 = 100,000 entities
Source 2 = 5,000,000 entities
Source 3 = 5,000,000 entities
```

A naive approach would require approximately:

```text
100,000 × (5,000,000 + 5,000,000)
```

potential comparisons.

This is not practical.

Instead, the system first identifies a smaller set of plausible candidates.

```text
All Possible Pairs
        |
        v
     Blocking
        |
        v
Plausible Candidate Pairs
        |
        v
 Feature Engineering
        |
        v
      XGBoost
```

---

## 6.2 Blocking Architecture

The production blocking architecture combines multiple deterministic channels.

```text
                    Source 1 Entity
                          |
          +---------------+---------------+
          |               |               |
          v               v               v
       Country       Name Tokens       Prefixes
          |               |               |
          |          +----+----+      +---+---+
          |          |         |      |       |
          |       Prefix 3   Prefix 4 |       |
          |                             
          +---------------+---------------+
                          |
                          v
                   Address Tokens
                          |
                          v
                 Candidate Union
                          |
                          v
                 Candidate Pairs
```

The major blocking signals include:

| Blocking Channel | Purpose                                            |
| ---------------- | -------------------------------------------------- |
| Country          | Restrict candidates using geographic information   |
| Name Token       | Identify records sharing informative name tokens   |
| Name Prefix-3    | Capture short prefix-level similarities            |
| Name Prefix-4    | Provide a stronger prefix signal                   |
| Address Token    | Recover candidates using location/address evidence |

Candidates produced by multiple channels are merged and their blocking evidence is retained.

Implementation:

`src/blocking.py`

---

# 7. Candidate Safety Layer

Blocking can still produce large candidate sets for common businesses, generic names, or highly frequent tokens.

A candidate safety layer therefore controls the number of pairs that reach the expensive ML stage.

```text
Blocking
   |
   v
Potential Candidates
   |
   v
Evidence Ranking
   |
   v
Top 400 Candidates / Source 1 Entity
   |
   v
Feature Engineering
```

Candidates are ranked using evidence including:

* Exact name evidence
* Number of shared blocking keys
* Name-token evidence
* Address-token evidence
* Prefix evidence

The ordering is deterministic.

This provides two benefits:

1. Computational control
2. Reproducible inference

Implementation:

`src/candidate_safety.py`

---

# 8. Training Data Construction

The model is trained on entity pairs rather than individual entities.

Each training example represents:

```text
Entity A
    +
Entity B
    |
    v
Similarity Features
    |
    v
Match / Non-Match Label
```

## 8.1 Positive Pairs

Positive pairs represent known corresponding entities.

```text
Entity A
    +
Entity A's True Match
    |
    v
Label = 1
```

## 8.2 Hard Negatives

Hard negatives are particularly important.

Instead of only generating obviously unrelated examples, the pipeline focuses on candidates that appear plausible but are actually different.

```text
Very Similar Name
       +
Similar Address
       +
Wrong Entity
       |
       v
Hard Negative
```

This forces the model to learn subtle differences.

## 8.3 Leakage Prevention

Train/validation splitting is performed at the Source 1 entity level.

This prevents different candidate pairs belonging to the same reference entity from being arbitrarily distributed across training and validation.

Implementation:

`src/training_pairs.py`

`src/split.py`

---

# 9. Feature Engineering

Each candidate pair is converted into a numerical feature vector.

The final feature representation contains **57 features**.

<details>
<summary><strong>Name Similarity Features</strong></summary>

Name comparison includes multiple complementary measurements:

* Exact equality
* Levenshtein similarity
* Jaro-Winkler similarity
* Character-set similarity
* Token overlap
* Token Jaccard similarity
* Containment
* Length ratios
* Length differences
* Token statistics

The purpose is to capture both character-level and token-level similarity.

</details>

<details>
<summary><strong>Address Similarity Features</strong></summary>

Address comparison includes:

* Exact equality
* Levenshtein similarity
* Jaro-Winkler similarity
* Character-set similarity
* Token overlap
* Token Jaccard similarity
* Length statistics

Addresses are particularly useful because two businesses with similar names may still be distinguishable through their locations.

</details>

<details>
<summary><strong>Country Features</strong></summary>

Country-level features provide geographic evidence:

* Country agreement
* Country mismatch
* Missing country indicators

</details>

<details>
<summary><strong>Structural Features</strong></summary>

Structural features describe the shape of the data:

* Number of tokens
* Number of digits
* Presence of digits
* String length
* Length differences
* Missingness indicators

</details>

<details>
<summary><strong>Blocking Features</strong></summary>

The model also knows how a candidate was discovered.

Features include:

* Blocking-channel indicators
* Shared blocking-key counts

This allows the classifier to distinguish between candidates discovered through strong and weak evidence.

</details>

<details>
<summary><strong>TF-IDF Features</strong></summary>

TF-IDF representations provide additional semantic-style lexical evidence.

The system generates:

* Name cosine similarity
* Address cosine similarity

These complement edit-distance and token-based metrics.

Implementation:

`src/features.py`

`src/tfidf_model.py`

---

# 10. Machine Learning Model

## XGBoost Pair Classifier

After feature engineering, each candidate pair is represented as:

```text
Candidate Pair
      |
      v
57 Numerical Features
      |
      v
XGBoost
      |
      v
P(match)
```

The model is a binary classifier.

The target is:

```text
1 = Same Entity

0 = Different Entity
```

The model uses the relationships between multiple features rather than relying on a single similarity metric.

For example:

```text
High Name Similarity
        +
High Address Similarity
        +
Same Country
        +
Multiple Blocking Signals
        |
        v
Higher Match Probability
```

### Model Configuration

The implemented configuration includes:

| Parameter          | Value              |
| ------------------ | ------------------ |
| Objective          | `binary:logistic`  |
| Tree Method        | Histogram-based    |
| Learning Rate      | `0.05`             |
| Maximum Depth      | `6`                |
| Estimators         | Up to `300`        |
| Class Weighting    | `scale_pos_weight` |
| Subsampling        | Enabled            |
| Column Subsampling | Enabled            |
| Regularization     | Enabled            |
| Early Stopping     | Configured         |

Implementation:

`src/train_phase5_configA.py`

---

# 11. Threshold Optimization

The XGBoost model outputs a probability.

For example:

```text
Candidate A → 0.982
Candidate B → 0.731
Candidate C → 0.043
```

A decision threshold converts these probabilities into match decisions.

The final locked threshold is:

```text
0.910
```

The threshold is not arbitrarily set to `0.50`.

The project uses **F0.5**, which places greater emphasis on precision.

This is important because a false positive entity match can be more damaging than leaving an uncertain candidate unmatched.

---

# 12. Inference Pipeline

The final inference pipeline follows the same architecture as training.

```text
                 TEST DATA
                     |
                     v
              Normalization
                     |
                     v
                 Blocking
                     |
                     v
             Candidate Union
                     |
                     v
            Candidate Safety
                     |
                     v
             Feature Creation
                     |
                     v
              XGBoost Model
                     |
                     v
            Match Probability
                     |
                     v
             Threshold = 0.910
                     |
                     v
             Final Predictions
```

The inference layer also validates the locked model configuration and feature schema before performing predictions.

Implementation:

`src/run_test_inference.py`

---

# 13. Blocking Experiments

The blocking architecture was selected through controlled experiments rather than by assumption.

A 10K Source 1 benchmark was used to compare multiple architectures.

| Configuration | Approach                          | Raw Recall | Candidates | Candidates / True Pair |
| ------------- | --------------------------------- | ---------: | ---------: | ---------------------: |
| A             | Team Baseline                     |     99.96% |     24.68M |                 710.38 |
| B             | Filtered Character/Token Channels |     90.96% |     15.21M |                 481.22 |
| C             | Config B + Address Token          |     99.96% |     35.87M |                1032.63 |
| D             | Full Union                        |     99.96% |     36.36M |                1046.74 |

## Selected Configuration

**Configuration A** was selected as the locked architecture.

The decision considered:

* Candidate recall
* Candidate volume
* Computational cost
* Downstream scalability
* Behavior under the candidate safety layer

At the tested 400-candidate cap:

```text
Post-cap candidate recall = 98.66%
```

Detailed experiment:

`experiments/final_blocking_tournament_report.md`

---

# 14. Model Evaluation

The locked validation benchmark produced the following results:

| Metric             |     Result |
| ------------------ | ---------: |
| Macro F0.5         | **98.63%** |
| Macro Precision    | **99.10%** |
| Macro Recall       | **98.07%** |
| Pairwise Precision | **99.23%** |
| Pairwise Recall    | **98.08%** |
| Pairwise F0.5      | **99.00%** |
| True Positives     |  **1,683** |
| False Positives    |     **13** |
| False Negatives    |     **33** |

The relatively high precision reflects the project's precision-oriented decision threshold.

Validation results:

`experiments/phase5_configA_validation_metrics.json`

---

# 15. Scalability

A recorded inference experiment evaluated the pipeline against millions of target records.

| Measurement                      |     Result |
| -------------------------------- | ---------: |
| Source 1 entities                |        100 |
| Source 2 entities                |  4,887,273 |
| Source 3 entities                |  5,082,316 |
| Raw candidates                   | 41,212,184 |
| Candidates after safety cap      |     40,000 |
| Zero-candidate Source 1 entities |          0 |
| Predicted matches                |      2,377 |

This experiment demonstrates the purpose of the architecture.

Without candidate generation and safety controls, all downstream processing would have to operate over tens of millions of candidate pairs.

The blocking and candidate-safety stages reduce this to a bounded number of pairs before expensive feature computation and ML scoring.

Benchmark metadata:

`experiments/phase6_test_inference_metadata.json`

---

# 16. Reproducibility

The project was designed with deterministic execution in mind.

<details>
<summary><strong>Deterministic preprocessing</strong></summary>

Normalization operations are deterministic.

The same input produces the same normalized representation.

</details>

<details>
<summary><strong>Deterministic candidate ordering</strong></summary>

Candidate ranking uses deterministic evidence ordering and stable lexical tie-breaking.

</details>

<details>
<summary><strong>Configuration fingerprint</strong></summary>

The final pipeline uses a configuration fingerprint to identify the locked configuration.

```text
87f20ceeb84ccc6ea2d48678c7810ac5
```

</details>

<details>
<summary><strong>Fixed feature schema</strong></summary>

The model expects a fixed 57-feature representation during inference.

</details>

<details>
<summary><strong>Leakage prevention</strong></summary>

Train/validation splitting is performed at the entity level rather than by randomly splitting candidate rows.

</details>

<details>
<summary><strong>Input integrity</strong></summary>

Original input datasets are not modified by the pipeline.

No external lookup dataset is required for matching.

</details>

---

# 17. Project Structure

```text
Amazon-ML-Challenge/
│
├── data/
│   ├── train/
│   └── test/
│
├── src/
│   ├── normalize.py
│   ├── blocking.py
│   ├── candidate_safety.py
│   ├── training_pairs.py
│   ├── split.py
│   ├── features.py
│   ├── tfidf_model.py
│   ├── train_phase5_configA.py
│   ├── model_artifacts.py
│   ├── run_test_inference.py
│   ├── metrics.py
│   └── collision.py
│
├── models/
│
├── experiments/
│   ├── final_architecture.md
│   ├── final_blocking_tournament_report.md
│   ├── final_blocking_readiness_audit.md
│   ├── phase5_configA_validation_metrics.json
│   └── phase6_test_inference_metadata.json
│
├── output/
│
├── requirements.txt
└── README.md
```

---

# 18. Technology Stack

| Technology   | Role in the System                    |
| ------------ | ------------------------------------- |
| Python       | Core pipeline implementation          |
| Pandas       | Data loading and tabular processing   |
| NumPy        | Numerical operations                  |
| Scikit-learn | TF-IDF and machine-learning utilities |
| XGBoost      | Supervised entity-pair classification |
| RapidFuzz    | String similarity calculations        |
| SciPy        | Numerical/scientific utilities        |
| Jupyter      | Exploration and experimentation       |

---

# 19. Running the Project

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Dataset Structure

Place the required competition files under:

```text
data/
├── train/
└── test/
```

## Run Inference

```bash
python src/run_test_inference.py
```

The inference pipeline validates the required model artifacts, configuration and feature schema before scoring candidates.

---

# 20. Experiments and Documentation

The repository contains the experiment trail used to arrive at the final architecture.

<details>
<summary><strong>Final Architecture</strong></summary>

`experiments/final_architecture.md`

Documents the selected end-to-end architecture and major engineering decisions.

</details>

<details>
<summary><strong>Blocking Tournament</strong></summary>

`experiments/final_blocking_tournament_report.md`

Contains the controlled comparison of blocking architectures and candidate-generation performance.

</details>

<details>
<summary><strong>Validation Metrics</strong></summary>

`experiments/phase5_configA_validation_metrics.json`

Contains the validation results for the locked XGBoost configuration.

</details>

<details>
<summary><strong>Inference Benchmark</strong></summary>

`experiments/phase6_test_inference_metadata.json`

Contains the recorded large-scale inference benchmark.

</details>

<details>
<summary><strong>Blocking Readiness Audit</strong></summary>

`experiments/final_blocking_readiness_audit.md`

Documents stress testing, candidate-cap behavior and identified limitations.

</details>

---

# 21. Limitations

A major engineering limitation identified during testing is the fixed candidate safety cap.

The 400-candidate limit provides an important computational guarantee, but extremely high-frequency entities can generate more plausible candidates than the cap allows.

This creates a theoretical recall ceiling for those entities.

Therefore, the system should not be interpreted as having perfect recall for every possible entity distribution.

The limitation is explicitly documented because a robust ML system should report both its strengths and its failure modes.

---

# 22. Key Engineering Decisions

<details>
<summary><strong>Why not use exact matching?</strong></summary>

Exact matching fails when names or addresses contain spelling, formatting, punctuation, abbreviation or Unicode variations.

The system therefore uses multiple similarity signals.

</details>

<details>
<summary><strong>Why not compare every record?</strong></summary>

The number of possible pairs becomes computationally infeasible at millions of records.

Blocking reduces the search space before expensive ML operations.

</details>

<details>
<summary><strong>Why multiple blocking strategies?</strong></summary>

No single field is reliable enough.

Name-based signals can fail when names change, while address-based signals can fail when addresses are incomplete.

Combining independent channels improves candidate recall.

</details>

<details>
<summary><strong>Why hard negatives?</strong></summary>

Easy negatives do not teach the model enough about difficult entity pairs.

Hard negatives force the model to distinguish between highly similar but unrelated businesses.

</details>

<details>
<summary><strong>Why XGBoost?</strong></summary>

The problem is naturally represented as structured/tabular pairwise features.

XGBoost can learn nonlinear interactions between:

* Name similarity
* Address similarity
* Country agreement
* Blocking evidence
* Structural features
* TF-IDF similarity

</details>

<details>
<summary><strong>Why optimize the threshold?</strong></summary>

Entity resolution is precision-sensitive.

A false positive can incorrectly merge two different real-world businesses.

The decision threshold is therefore optimized instead of blindly using 0.50.

</details>

---

# Final System Summary

The completed system can be summarized as:

```text
                  INPUT DATA
                      |
                      v
              DATA NORMALIZATION
                      |
                      v
                  BLOCKING
                      |
                      v
             CANDIDATE GENERATION
                      |
                      v
              CANDIDATE SAFETY
                      |
                      v
             FEATURE ENGINEERING
                      |
             +--------+--------+
             |                 |
             v                 v
       STRING FEATURES      TF-IDF
             |                 |
             +--------+--------+
                      |
                      v
                  XGBOOST
                      |
                      v
             MATCH PROBABILITY
                      |
                      v
            THRESHOLD DECISION
                      |
                      v
              FINAL MATCHES
```

The central engineering principle behind the project is:

> **Do not ask a machine-learning model to compare millions of impossible pairs. First reduce the search space intelligently, then use machine learning to make the difficult decisions.**

This architecture combines scalable candidate generation with detailed pairwise modeling to create a practical entity-resolution pipeline for large, noisy business datasets.


