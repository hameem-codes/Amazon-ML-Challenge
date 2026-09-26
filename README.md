# Business Entity Resolution - Amazon ML Challenge

An end-to-end scalable pipeline for high-precision Business Entity Resolution across disparate structured data sources.

## Overview
The goal is to link corresponding business entities across multiple noisy sources (Source 1, Source 2, Source 3) under strict computational and memory constraints.

## Pipeline Architecture
1. **Data Loading & Preprocessing**: Efficient streaming and parsing of multi-source tabular data.
2. **Text Normalization**: Multi-lingual text cleaning, company suffix normalization, punctuation stripping, and tokenization.
3. **Blocking / Candidate Generation**:
   - Multi-channel candidate generation (exact match, character n-gram hashing, TF-IDF / inverted index blocking).
   - Inverted indexing with adaptive candidate caps to scale across millions of entities.
4. **Feature Engineering**:
   - String similarity metrics (Levenshtein, Jaro-Winkler, Token Sort, Token Set via RapidFuzz).
   - Geographic & address matching (country, postal code, city, coordinates).
   - Entity type, phone, and metadata overlap features.
5. **Matching Model**:
   - High-throughput XGBoost classification model trained on calibrated negative/positive candidate distributions.
   - Robust probability estimation and precision-recall tuning.
6. **Threshold Optimization & Clustering**:
   - Optimal F-score thresholding with precision safety guards.
   - Candidate grouping and final submission generation.

## Directory Structure
```
├── data/                  # Raw dataset directory (.gitkeep; data excluded from git)
│   ├── train/             # Training data files (train_source1/2/3, ground truth)
│   └── test/              # Test data files (test_source1/2/3)
├── src/                   # Core pipeline source code
│   ├── blocking.py        # Candidate blocking & indexing algorithms
│   ├── features.py        # Feature extraction & similarity computations
│   ├── normalize.py       # Entity text & field normalization routines
│   ├── train_phase5_configA.py # XGBoost model training and evaluation
│   └── run_test_inference.py   # Test inference & candidate pair generator
├── models/                # Trained model artifacts & schemas
├── experiments/           # Benchmark reports, audits, and validation metrics
├── output/                # Generated candidate pairs & matching results
└── requirements.txt       # Project dependencies
```

## Setup & Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Dataset Preparation
Place the competition datasets in `data/train/` and `data/test/`.

### 3. Run Inference
```bash
python src/run_test_inference.py
```

