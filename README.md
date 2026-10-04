# Business Entity Resolution — Amazon ML Challenge

> A scalable entity-resolution pipeline for identifying matching business records across independent and noisy data sources.

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python\&logoColor=white)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-150458?logo=pandas\&logoColor=white)](https://pandas.pydata.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-orange?logo=scikit-learn\&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Classifier-red)](https://xgboost.readthedocs.io/)
[![RapidFuzz](https://img.shields.io/badge/RapidFuzz-String%20Similarity-purple)](https://github.com/rapidfuzz/RapidFuzz)

---

## 📌 Project Overview

Business Entity Resolution is the task of determining whether records from different datasets represent the same real-world business.

This project processes three independent sources:

```text
Source 1
Reference / master entities
        │
        ├───────────────┐
        │               │
        ▼               ▼
    Source 2         Source 3
        │               │
        └───────┬───────┘
                ▼
        Entity Resolution
                │
                ▼
        Final Entity Matches
```

The same business can appear with:

```text
Different spellings
Different punctuation
Different abbreviations
Different token ordering
Different addresses
Missing values
Unicode variations
```

For example:

```text
ACME Technologies Private Limited
ACME TECHNOLOGIES PVT LTD
Acme Technologies Pvt. Ltd.
```

The system must determine whether these records represent the same entity.

---

# 🎯 1. Problem Definition

A naive solution would compare every Source 1 entity against every Source 2 and Source 3 entity.

For millions of records, this creates an enormous Cartesian product.

```text
                    ALL POSSIBLE PAIRS
                           │
                           ▼
                  ┌─────────────────┐
                  │     BLOCKING     │
                  └────────┬────────┘
                           ▼
                 PLAUSIBLE CANDIDATES
                           │
                           ▼
                  FEATURE ENGINEERING
                           │
                           ▼
                       XGBOOST
                           │
                           ▼
                    MATCH PROBABILITY
                           │
                           ▼
                    FINAL DECISION
```

The central engineering problem is therefore:

> Reduce the search space aggressively without eliminating true matches.

---

# 🏗️ 2. System Architecture

```text
┌───────────────────────────────────────────────┐
│                  INPUT DATA                   │
│           Source 1 / Source 2 / Source 3     │
└───────────────────────┬───────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────┐
│             DATA NORMALIZATION                │
│ Unicode • Case • Punctuation • Whitespace    │
│ Missing Values • Deterministic Cleaning      │
└───────────────────────┬───────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────┐
│                 BLOCKING                      │
│ Country • Name Tokens • Prefixes • Address    │
└───────────────────────┬───────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────┐
│             CANDIDATE SAFETY                  │
│       Evidence Ranking • Candidate Cap        │
└───────────────────────┬───────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────┐
│            FEATURE ENGINEERING                │
│ Name • Address • Country • Structure • TF-IDF│
└───────────────────────┬───────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────┐
│                 XGBOOST                      │
│             Pair Classification              │
└───────────────────────┬───────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────┐
│              DECISION LAYER                  │
│ Probability • Threshold • Final Matching     │
└───────────────────────┬───────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────┐
│                 OUTPUT                       │
│             Final Entity Links               │
└───────────────────────────────────────────────┘
```

---

# 🔄 3. End-to-End Pipeline

<details>
<summary><strong>View complete pipeline</strong></summary>

```text
Raw TSV Data
     │
     ▼
Data Loading
     │
     ▼
Deterministic Normalization
     │
     ▼
Blocking Key Generation
     │
     ▼
Multi-Channel Candidate Generation
     │
     ▼
Candidate Union + Deduplication
     │
     ▼
Candidate Safety / Ranking
     │
     ▼
Training Pair Construction
     │
     ▼
Pairwise Feature Engineering
     │
     ├───────────────┐
     │               │
     ▼               ▼
String Similarity   TF-IDF
     │               │
     └───────┬───────┘
             ▼
        XGBoost Model
             │
             ▼
      Match Probability
             │
             ▼
      Threshold Decision
             │
             ▼
       Final Matches
```

Each stage solves a different problem:

| Stage               | Purpose                                      |
| ------------------- | -------------------------------------------- |
| Data Loading        | Read and validate source datasets            |
| Normalization       | Make noisy values comparable                 |
| Blocking            | Reduce the candidate search space            |
| Candidate Safety    | Control worst-case candidate volume          |
| Feature Engineering | Convert entity pairs into numerical evidence |
| XGBoost             | Learn complex matching relationships         |
| Thresholding        | Convert probabilities into final decisions   |
| Output              | Produce entity-level matches                 |

</details>

---

# 🧹 4. Data Processing

<details>
<summary><strong>Normalization pipeline</strong></summary>

The system performs deterministic normalization before matching.

```text
Raw Value
   │
   ▼
Unicode Normalization
   │
   ▼
Case Normalization
   │
   ▼
Punctuation / Symbol Handling
   │
   ▼
Whitespace Normalization
   │
   ▼
Missing Value Handling
   │
   ▼
Normalized Value
```

Examples:

```text
"ACME, Ltd."
"Acme Ltd."
"ACME LTD"
```

are transformed into a consistent representation.

The purpose of normalization is not to declare two businesses equal.

It simply makes later comparison more reliable.

Implementation:

`src/normalize.py`

</details>

---

# 🔎 5. Blocking & Candidate Generation

Blocking is the main scalability mechanism.

Without blocking:

```text
Source 1 × (Source 2 + Source 3)
```

creates an impractical number of comparisons.

The implemented blocking system uses multiple signals.

| Blocking Strategy | Purpose                       |
| ----------------- | ----------------------------- |
| Country           | Geographic restriction        |
| Name Token        | Shared business-name evidence |
| Name Prefix-3     | Short prefix similarity       |
| Name Prefix-4     | Stronger prefix similarity    |
| Address Token     | Address-based recovery        |

The candidate sets from different channels are combined.

```text
             Source 1 Entity
                    │
       ┌────────────┼────────────┐
       ▼            ▼            ▼
    Country      Name         Address
       │         Tokens          │
       │           │             │
       │       Prefix 3/4        │
       └───────────┼─────────────┘
                   ▼
            Candidate Union
                   │
                   ▼
          Candidate Pairs
```

Implementation:

`src/blocking.py`

---

# 📊 6. The 10K Blocking Experiment

The blocking architecture was tested using **10,000 Source 1 records**.

The experiment measured both candidate volume and candidate recall against the available ground truth.

## Candidate Generation Results

| Strategy       | Candidate Pairs |
| -------------- | --------------: |
| Country        |               0 |
| Name Token     |       5,383,099 |
| Name Prefix-3  |       2,762,145 |
| Name Prefix-4  |       2,232,229 |
| Address Token  |       7,537,139 |
| Combined Union |  **15,192,981** |

The country strategy produced zero candidates because the frequency filter excluded overly common countries under the configured limit.

## Candidate Recall

| Strategy       |     Recall |
| -------------- | ---------: |
| Country        |      0.00% |
| Name Token     |     80.84% |
| Name Prefix-3  |     64.58% |
| Name Prefix-4  |     51.13% |
| Address Token  |     87.78% |
| Combined Union | **99.48%** |

### Key observation

No single blocking strategy was sufficient.

```text
Name Token       → 80.84%
Address Token    → 87.78%
Combined Union   → 99.48%
```

This demonstrated why multi-channel blocking is necessary.

The experiment also exposed an important trade-off:

```text
Higher Recall
      ↕
Larger Candidate Set
      ↕
Higher Computation
```

---

# 🧪 7. Blocking Architecture Experiments

<details>
<summary><strong>Why multiple blocking architectures were tested</strong></summary>

Blocking is a recall-sensitive stage.

Once a true pair is removed during blocking, the downstream model cannot recover it.

Therefore the project evaluated alternative blocking architectures rather than selecting one purely based on candidate volume.

</details>

<details>
<summary><strong>Experiment design</strong></summary>

The blocking experiments evaluated:

* Candidate recall
* Candidate volume
* Candidate efficiency
* Frequency filtering
* Prefix strategies
* Token strategies
* Address evidence
* Combined blocking channels
* Behavior under candidate caps

</details>

<details>
<summary><strong>Engineering lesson</strong></summary>

A blocking strategy should not simply minimize the number of candidates.

A very aggressive blocker may appear efficient while silently removing true matches.

The objective is:

```text
High Candidate Recall
          +
Manageable Candidate Volume
          +
Stable Runtime
```

</details>

---

# 🛡️ 8. Candidate Safety Layer

Some blocking keys can generate thousands of candidates for a single entity.

The system therefore introduces a candidate safety layer.

```text
Blocking
   │
   ▼
Large Candidate Set
   │
   ▼
Evidence Ranking
   │
   ▼
Candidate Safety Cap
   │
   ▼
Bounded Candidate Set
```

The configured safety cap is:

```text
400 candidates per Source 1 entity
```

Candidates are ranked using available matching evidence such as:

* Name similarity
* Shared blocking keys
* Address evidence
* Prefix evidence
* Blocking-channel evidence

The ordering is deterministic.

Implementation:

`src/candidate_safety.py`

---

# 🧩 9. Training Pair Construction

The ML model operates on candidate pairs.

```text
Entity A + Entity B
        │
        ▼
Feature Generation
        │
        ▼
Match / Non-Match
```

## Positive Pairs

Known matching entities are labelled:

```text
1 = Match
```

## Negative Pairs

Non-matching candidates are labelled:

```text
0 = Non-Match
```

## Hard Negatives

Hard negatives are particularly important.

Example:

```text
Apple Technologies Pvt Ltd
Apple Technology Private Limited
```

may look extremely similar while representing different entities.

These difficult examples teach the model to distinguish between:

```text
Similar
```

and:

```text
Actually the same
```

Training/validation splitting is performed at the entity level to reduce leakage.

Implementation:

`src/training_pairs.py`

`src/split.py`

---

# 🧮 10. Feature Engineering

Candidate pairs are transformed into numerical features.

The feature system combines several independent sources of evidence.

<details>
<summary><strong>View feature categories</strong></summary>

### Name Features

* Exact similarity
* Levenshtein similarity
* Jaro-Winkler similarity
* Token similarity
* Token overlap
* Character-set similarity
* Length statistics
* Containment signals

### Address Features

* Exact similarity
* Levenshtein similarity
* Jaro-Winkler similarity
* Token similarity
* Character-set similarity
* Length statistics

### Country Features

* Country agreement
* Country mismatch
* Missingness indicators

### Structural Features

* String lengths
* Token counts
* Digit counts
* Digit presence
* Missing values

### Blocking Features

* Blocking channel indicators
* Shared blocking-key evidence
* Candidate-generation evidence

### TF-IDF Features

* Name cosine similarity
* Address cosine similarity

</details>

Implementation:

`src/features.py`

---

# 🔤 11. String Similarity

The project uses multiple string-comparison techniques because no single metric works reliably for all business-name variations.

<details>
<summary><strong>Why multiple similarity metrics?</strong></summary>

### Levenshtein

Useful for character-level edits:

```text
TECHNOLOGIES
TECHNOLOGY
```

### Jaro-Winkler

Useful for short strings and prefix similarities.

### Token Similarity

Useful when word order changes:

```text
ABC Technology India
India ABC Technology
```

### Character-Level Similarity

Useful when punctuation and formatting vary.

Together, these features provide complementary evidence to the classifier.

</details>

---

# 📐 12. TF-IDF Similarity

TF-IDF provides another representation of textual similarity.

```text
Business Name
      │
      ▼
TF-IDF Vector
      │
      ▼
Cosine Similarity
      │
      ▼
Numerical Feature
```

The system uses TF-IDF similarity for:

* Business names
* Addresses

TF-IDF complements edit-distance methods because it focuses on the importance of tokens rather than only character-level changes.

Implementation:

`src/tfidf_model.py`

---

# 🤖 13. XGBoost Matching Model

The final classifier operates on engineered pairwise features.

```text
Entity A
   +
Entity B
   │
   ▼
Similarity Features
   │
   ▼
XGBoost
   │
   ▼
P(match)
```

The model learns nonlinear interactions between signals.

For example:

```text
High Name Similarity
        +
High Address Similarity
        +
Same Country
        +
Multiple Blocking Signals
        │
        ▼
Higher Match Probability
```

The implemented model uses XGBoost for binary classification.

Key configuration includes:

| Parameter          | Configuration   |
| ------------------ | --------------- |
| Objective          | Binary Logistic |
| Learning Rate      | 0.05            |
| Maximum Depth      | 6               |
| Estimators         | Up to 300       |
| Tree Method        | Histogram       |
| Class Weighting    | Enabled         |
| Subsampling        | Enabled         |
| Column Subsampling | Enabled         |
| Regularization     | Enabled         |
| Early Stopping     | Configured      |

Implementation:

`src/train_phase5_configA.py`

---

# 🎚️ 14. Threshold Optimization

The model outputs a probability rather than a direct match.

Example:

```text
Candidate A → 0.982
Candidate B → 0.731
Candidate C → 0.043
```

The probability must be converted into a final decision.

The locked decision threshold is:

```text
0.910
```

The project uses **F0.5** as an important optimization metric because it places greater emphasis on precision.

This is appropriate for entity resolution where an incorrect match can be more harmful than leaving an uncertain candidate unmatched.

---

# 📈 15. Validation Results

The locked validation experiment produced:

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

These results demonstrate strong precision and recall on the locked validation setup.

Validation artifact:

`experiments/phase5_configA_validation_metrics.json`

---

# ⚡ 16. Large-Scale Inference

The inference benchmark tested the pipeline against millions of target records.

| Metric                           |         Result |
| -------------------------------- | -------------: |
| Source 1 entities tested         |            100 |
| Source 2 entities                |      4,887,273 |
| Source 3 entities                |      5,082,316 |
| Raw candidates                   | **41,212,184** |
| Candidates after cap             |     **40,000** |
| Zero-candidate Source 1 entities |          **0** |
| Predicted matches                |      **2,377** |

The important result is the reduction:

```text
41,212,184 raw candidates
          │
          ▼
      Candidate Cap
          │
          ▼
40,000 downstream candidates
```

This demonstrates why candidate generation and candidate safety are essential for large-scale entity resolution.

Benchmark:

`experiments/phase6_test_inference_metadata.json`

---

# 🔬 17. Experiment Journey

The project was developed through iterative experiments rather than a single model-training step.

```text
Initial Data Understanding
          │
          ▼
Normalization
          │
          ▼
Blocking Experiments
          │
          ▼
Candidate Recall Analysis
          │
          ▼
Candidate Safety
          │
          ▼
Training Pair Construction
          │
          ▼
Feature Engineering
          │
          ▼
Model Experiments
          │
          ▼
Threshold Optimization
          │
          ▼
Large-Scale Inference
          │
          ▼
Readiness Audit
```

The experiment trail is retained to make the engineering decisions reproducible.

---

# ⚠️ 18. Limitations & Challenges

<details>
<summary><strong>Candidate-cap limitation</strong></summary>

The fixed 400-candidate limit controls computation but introduces a theoretical recall ceiling.

If an entity generates more plausible candidates than the cap allows, a true match may be removed before the ML stage.

</details>

<details>
<summary><strong>High-frequency blocking keys</strong></summary>

Very common tokens can generate extremely large candidate sets.

This requires frequency filtering and candidate ranking.

The 10K experiment showed why naive blocking keys cannot simply be used without controlling their frequency.

</details>

<details>
<summary><strong>Blocking recall vs efficiency</strong></summary>

A blocker that produces very few candidates may look efficient but can lose true matches.

Conversely, a blocker with extremely high recall may produce too many candidates.

The project therefore treats blocking as an optimization problem:

```text
Candidate Recall
       vs
Candidate Volume
       vs
Computational Cost
```

</details>

<details>
<summary><strong>Ambiguous business names</strong></summary>

Common names can correspond to many different entities.

Name similarity alone is therefore insufficient.

Address, country and structural evidence are required to separate similar entities.

</details>

<details>
<summary><strong>No external lookup data</strong></summary>

The system intentionally operates using the provided data rather than relying on external business databases or manually created mappings.

This makes the pipeline deterministic and reproducible within the challenge constraints.

</details>

---

# 🔐 19. Reproducibility

The pipeline emphasizes deterministic execution.

| Mechanism                   | Purpose                          |
| --------------------------- | -------------------------------- |
| Deterministic normalization | Stable text representation       |
| Stable candidate ordering   | Reproducible candidate selection |
| Entity-level splitting      | Reduced validation leakage       |
| Fixed feature schema        | Consistent model inputs          |
| Locked threshold            | Reproducible decisions           |
| Configuration fingerprint   | Detect configuration drift       |
| Preserved source data       | Input integrity                  |

Configuration fingerprint:

```text
87f20ceeb84ccc6ea2d48678c7810ac5
```

---

# 📁 20. Project Structure

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
├── requirements.txt
└── README.md
```

---

# 🧰 21. Technology Stack

| Technology   | Role                        |
| ------------ | --------------------------- |
| Python       | Core implementation         |
| Pandas       | Data processing             |
| NumPy        | Numerical operations        |
| Scikit-learn | TF-IDF and ML utilities     |
| XGBoost      | Pairwise classification     |
| RapidFuzz    | String similarity           |
| SciPy        | Numerical utilities         |
| Jupyter      | Exploration and experiments |

---

# 🚀 22. Running the Project

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Dataset Layout

```text
data/
├── train/
└── test/
```

## Run Inference

```bash
python src/run_test_inference.py
```

The inference pipeline validates the model artifacts, configuration and feature schema before producing predictions.

---

# 📚 23. Experiment Documentation

<details>
<summary><strong>Final Architecture</strong></summary>

`experiments/final_architecture.md`

Documents the selected end-to-end architecture.

</details>

<details>
<summary><strong>Blocking Tournament</strong></summary>

`experiments/final_blocking_tournament_report.md`

Documents the blocking experiments, candidate recall and candidate-volume trade-offs.

</details>

<details>
<summary><strong>Validation Metrics</strong></summary>

`experiments/phase5_configA_validation_metrics.json`

Contains the locked model validation results.

</details>

<details>
<summary><strong>Inference Benchmark</strong></summary>

`experiments/phase6_test_inference_metadata.json`

Contains large-scale inference statistics.

</details>

<details>
<summary><strong>Readiness Audit</strong></summary>

`experiments/final_blocking_readiness_audit.md`

Documents final blocking validation, limitations and production-readiness considerations.

</details>

---

# 🧭 Final Architecture at a Glance

```text
                     BUSINESS RECORDS
                            │
                            ▼
                  ┌──────────────────┐
                  │   NORMALIZATION  │
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │     BLOCKING     │
                  │ Multi-channel    │
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │ CANDIDATE SAFETY │
                  │    400 / S1      │
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │    FEATURES      │
                  │ Name / Address   │
                  │ Country / TF-IDF │
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │     XGBOOST      │
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │    THRESHOLD     │
                  │      0.910       │
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │ FINAL MATCHES    │
                  └──────────────────┘
```

---

# Key Takeaway

The project is not simply a classification model.

It is a complete entity-resolution system designed around the following principle:

```text
Do not compare everything.

First:
    Reduce the search space intelligently.

Then:
    Generate rich evidence for each plausible pair.

Finally:
    Use machine learning to make the difficult matching decision.
```

The combination of **multi-channel blocking, candidate safety, pairwise feature engineering, XGBoost classification, threshold optimization and large-scale validation** forms the complete solution.

---
