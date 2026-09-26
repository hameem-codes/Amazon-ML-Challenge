# Final End-to-End System Architecture

This document describes the complete, aligned production architecture for the Amazon ML Challenge 2026 Business Entity Resolution project.

```mermaid
flowchart TD
    subgraph Data Ingestion & Normalization
        A[Raw Train & Test TSV Files] --> B[Conservative Normalizer\nsrc/normalize.py]
        B --> C1[Source 1: Deduplicated Reference\nraw + normalized columns]
        B --> C2[Target Sources 2 & 3\nraw + normalized columns]
    end

    subgraph Multi-Key Inverted-Index Blocking
        C1 & C2 --> D[Blocking Engine\nsrc/blocking.py]
        D --> D1[Filtered Char 3-Gram\nPrimary Channel]
        D --> D2[Selective Token\nSecondary Channel]
        D --> D3[Rare Token <=1%\nComplementary]
        D --> D4[Token Pair Overlap\nPrecision Channel]
        D --> D5[Exact Normalized Name\nHigh Precision]
        D1 & D2 & D3 & D4 & D5 --> E[Multi-Key Union & Channel Evidence Retention]
    end

    subgraph Candidate Safety & Materialization
        E --> F[Candidate Safety Cap\nsrc/candidate_safety.py]
        F --> F1[Evidence-Ranked Overflow Filtering\nmax_candidates_per_s1]
        F1 --> G[Materialized Candidate Pairs\ncandidate_pairs.tsv]
        F1 --> G1[Blocking & Cap Recall Metrics Ceiling]
    end

    subgraph Training Data Construction & Splitting
        G --> H[Training Pair Construction\nsrc/training_pairs.py]
        H --> H1[Surviving True Positives]
        H --> H2[Hard Negatives Selection\nhigh similarity, shared keys, same country]
        H1 & H2 --> I[Entity-Level Train/Validation Split\nsrc/split.py - Split by S1 ID]
    end

    subgraph Feature Engineering & Diagnostic Pipeline
        I --> J[Pairwise Feature Engineering\nsrc/features.py & src/build_features.py\n57 numerical features]
        K[Unlabeled Transductive TF-IDF\nTrain + Test Union Text] --> J
        J --> L[Feature Validation\nNo NaNs, No Infs, No Leakage, Strict Determinism]
    end

    subgraph Modeling & Prediction Assembly
        L --> M[XGBoost Pair Classifier\nDynamic scale_pos_weight]
        M --> N[Probability Calibration Check & Diagnostics]
        N --> O[F0.5 Threshold Optimization\nMacro F0.5 Metric]
        O --> P[Match Set Assembly\nMany-to-Many Allowed, Zero/One/Many Matches]
        P --> Q[Collision Analysis\nsrc/collision.py - Diagnostic]
        P --> R[Final Output: matching_results.tsv]
    end
```

## Architectural Guarantees
1. **Recall-First Blocking with Bound Candidate Budget:** Filtering high-frequency tokens and n-grams avoids Cartesian explosion, while the safety cap deterministically ranks candidates by evidence.
2. **Strict Entity-Level Isolation:** Train and validation sets are split strictly by Source 1 entity IDs to eliminate reference entity leakage.
3. **Evidence Preservation:** Every candidate pair carries bitflags for the channels through which it was generated (`num_blocking_keys`, `blocked_exact_name`, etc.), enriching the downstream classifier.
4. **Hard-Negative Training:** The model trains on challenging negative candidates that survived blocking, preventing trivial negatives from degrading discriminative precision.
5. **Transductive Unlabeled TF-IDF:** Fits vectorizers on the union of all raw names and addresses across train and test without labels, mitigating out-of-vocabulary drift across country sets.
6. **Macro F0.5 Optimization:** Tailored specifically to penalize precision errors heavily, honoring the competition evaluation.
