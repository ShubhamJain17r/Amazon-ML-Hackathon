# Amazon ML Challenge 2026 — Multilingual Business Entity Resolution
### Scalable S3-Centric Pipeline for Large-Scale Entity Matching

---

## 1. Leaderboard Score Progression

| Run / Iteration | Key Architecture & Innovations | Validation $F_{0.5}$ | Unstop Leaderboard Score | Status |
| :--- | :--- | :---: | :---: | :---: |
| **Run 1** | Broken string slicing, 35 positive pairs | ~0.08 | `0.060` | Superceded |
| **Run 2** | Ground-truth anchored sampling baseline | ~0.20 | `0.144` | Superceded |
| **Run 3** | Multi-token intersection (Source 3 starved due to global posting cap) | ~0.35 | `0.253` | Superceded |
| **Run 4** | Dual Inverted Index (fair S2/S3 quota), 9 features, unconstrained matches | ~0.42 | `0.409` | Superceded |
| **Ablation Run** | Hard-truncated to 1 ID per entity (destroyed dual-catalog recall) | ~0.39 | `0.380` | Proved dual-source structure |
| **Phase 2 (Latest)** | **14 Features + 799 Trees + Dual-Source Top-1 Policy (threshold = 0.60)** | **`0.8447`** | **`0.459`** | **Current Benchmark** 🚀 |

---

## 2. System Architecture

```text
  ┌────────────────────────────────────────────────────────┐
  │                 AWS S3 Cloud Lake / DB                 │
  │               (Single Source of Truth)                 │
  │  s3://<bucket>/student_resource/ ──> features/         │
  │                                  ──> models/           │
  │                                  ──> submissions/      │
  └───────────────▲────────────────────────▲───────────────┘
                  │ Read / Write           │ Read / Write
                  │ Parquet & TSV          │ Models & ZIPs
  ┌───────────────┴───────────────┐ ┌──────┴───────────────┐
  │   Data Pipeline & Blocking    │ │  Modeling & Submission│
  │   - Text Normalization        │ │  - LightGBM (14 Feat) │
  │   - Dual Inverted Indexes     │ │  - Macro F_0.5 Sweep  │
  │   - Fair S2/S3 Quotas         │ │  - Dual-Source Top-1  │
  │   - RapidFuzz Feature Store   │ │  - Output Validation │
  │   (Google Colab / Local)      │ │  (Google Colab / Local)
  └───────────────────────────────┘ └───────────────────────┘
                  ▲                        ▲
                  │                        │
                  └─────────┬──────────────┘
                            │ Git push / pull
            ┌───────────────┴───────────────┐
            │       GitHub Repository       │
            │   Code & Version Control      │
            └───────────────────────────────┘
```

> **Cloud Infrastructure Decision:**  
> Our pipeline uses **strictly Amazon S3** as a high-throughput, centralized cloud storage backend. Compute is executed on **Google Colab or Local Machines**, completely bypassing AWS SageMaker quota restrictions and instance costs.

---

## 3. Central S3 Storage Hierarchy

All raw data, pre-engineered feature stores, trained model weights, and generated submissions are centrally organized on S3:

```text
s3://<your-bucket-name>/
├── student_resource/                      # Official competition datasets
│   └── dataset/
│       ├── train/
│       │   ├── train_source1.tsv          # ~2.2M source entities
│       │   ├── train_source2.tsv          # ~5.0M catalog records
│       │   ├── train_source3.tsv          # ~5.2M catalog records
│       │   └── train_ground_truth.tsv     # Official multi-catalog ground truth
│       └── test/
│           ├── test_source1.tsv           # 1,732,544 test query entities
│           ├── test_source2.tsv           # 4,887,273 reference catalog records
│           └── test_source3.tsv           # 5,082,316 reference catalog records
│
├── features/                              # Precomputed C++ RapidFuzz feature tables
│   ├── phase2_14feat/                     # Production 14-feature training store
│   │   ├── train_candidates.parquet       # 542,226 rows (80% split)
│   │   └── val_candidates.parquet         # 135,800 rows (20% split, GroupShuffleSplit)
│   └── anchored_100k/                     # Previous baseline feature partitions
│       ├── train_candidates.parquet
│       └── val_candidates.parquet
│
├── models/                                # Serialized production LightGBM models
│   └── lgbm_model_latest.pkl              # 14-feature model (Best Iteration: 799, 5.32 MB)
│
└── submissions/                           # Generated predictions and archives
    ├── matching_results.tsv               # Final submission file (1,732,544 rows)
    ├── candidate_pairs.tsv                # Candidate blocking pairs (1,732,544 rows)
    └── final_submission.zip               # Complete competition archive
```

---

## 4. Pipeline Stages & Notebooks

The pipeline is organized into sequential notebooks in `notebooks/`:

| Notebook | Purpose | Input | Output |
| :--- | :--- | :--- | :--- |
| [`00_s3_setup_test.ipynb`](notebooks/00_s3_setup_test.ipynb) | AWS S3 connectivity & IAM credentials check | S3 bucket credentials | Verified cloud access |
| [`01_eda_multilingual.ipynb`](notebooks/01_eda_multilingual.ipynb) | Multilingual EDA, text distributions, & country breakdown | Raw TSVs | Token frequency profiles |
| [`02_train_lightgbm.ipynb`](notebooks/02_train_lightgbm.ipynb) | 14-Feature store generation & LightGBM training (799 trees) | `student_resource/dataset/train` | `features/phase2_14feat/`, `lgbm_model_latest.pkl` |
| [`03_threshold_tuning.ipynb`](notebooks/03_threshold_tuning.ipynb) | Macro $F_{0.5}$ threshold sweep under Dual-Source Top-1 Policy | `val_candidates.parquet` | Optimal threshold ($\	au = 0.60$, $F_{0.5} = 0.8447$) |
| [`04_submission.ipynb`](notebooks/04_submission.ipynb) | High-speed streaming test inference & rule validation | `test_source*.tsv` | `matching_results.tsv`, `candidate_pairs.tsv` |

---

## 5. Core Algorithmic Breakthroughs

### A. Dual-Source Top-1 Selection Policy
In the Amazon ML Challenge, ground-truth analysis revealed that **~43% of entities match BOTH Source 2 AND Source 3**, while duplicate matches from the *same* source almost never exist.
* **Old Failure Mode**: Emitting multiple IDs from the same source severely degraded precision under Macro $F_{0.5}$. Hard-truncating each line to 1 ID slashed dual-catalog recall.
* **The Solution**: For each `source1_entity_id`, select the highest-scoring candidate from Source 2 (if $\\ge \	au$) and the highest-scoring candidate from Source 3 (if $\\ge \	au$). This emits `max 1x S2 + max 1x S3`, eliminating same-source false positives while capturing full dual-catalog recall.

### B. 14 Discriminative Features
1. `name_ratio`: Levenshtein ratio between business names.
2. `name_partial`: Best substring similarity.
3. `name_token_sort`: Token-sorted ratio (handles word reordering).
4. `name_token_set`: Token set ratio (handles extraneous noise tokens).
5. `addr_ratio`: Full address string ratio.
6. `addr_partial`: Address substring ratio.
7. `addr_token_set`: Address set intersection ratio.
8. `name_len_diff`: Absolute character length discrepancy of names.
9. `addr_len_diff`: Absolute character length discrepancy of addresses.
10. `pin_exact_match`: Binary indicator (1.0/0.0) of exact 5–6 digit postal code identity.
11. `name_jaro_winkler`: Character transposition distance weighted on prefix.
12. `name_token_jaccard`: Word token set intersection over union.
13. `first_word_ratio`: Exact similarity of the critical brand anchor word.
14. `is_source2`: Source origin prior indicator.

---

## 6. Guide: How to Submit on Unstop

### Daily Leaderboard Submission (`matching_results.tsv`)
During the iterative phase, Unstop accepts daily submissions to evaluate the model on the public leaderboard.

1. **Verify Formatting Locally**:
   Before uploading, always run the official submission validator:
   ```bash
   python3 utils/validate_submission.py \
       --matching matching_results.tsv \
       --candidate output/candidate_pairs.tsv \
       --test-dir dataset/test
   ```
   Ensure it outputs `PASS — no blocking issues found. Safe to submit.` (Exit Code: 0).

2. **Upload to Unstop Portal**:
   - Go to the **Amazon ML Challenge 2026** competition page on Unstop.
   - Click **Submit** on the leaderboard track.
   - Select and upload `matching_results.tsv`.
   - Wait for the automated scoring engine to evaluate your file and update your rank.

---

### Final Round Submission (`final_submission.zip`)
For the final project submission, Unstop requires a complete zip archive containing code, predictions, and documentation:

1. **Package the Archive**:
   Run the packaging utility:
   ```bash
   python3 utils/package_submission.py \
       --team-name <your_team_name> \
       --matching matching_results.tsv \
       --candidate output/candidate_pairs.tsv \
       --doc docs/Documentation_template.md
   ```

2. **Archive Structure Verification**:
   Ensure `<your_team_name>_submission.zip` contains:
   ```text
   <your_team_name>_submission.zip
   ├── output/
   │   ├── matching_results.tsv       # Exact 1,732,544 rows
   │   └── candidate_pairs.tsv        # Exact 1,732,544 rows
   ├── code/
   │   └── business_entity_resolution/
   │       ├── src/                   # normalization.py, blocking.py, features.py
   │       ├── notebooks/             # 00 to 04 Jupyter notebooks
   │       ├── requirements.txt       # Frozen dependencies
   │       └── README.md              # Reproduction instructions
   └── Documentation_template.md      # Completed technical report
   ```
3. Upload the resulting `.zip` to the Unstop Final Round portal before the submission deadline.
