# Amazon ML Challenge 2026 — Multilingual Business Entity Resolution
### Scalable S3-Centric Pipeline for Large-Scale Entity Matching

---

## 1. System Architecture

```text
  ┌────────────────────────────────────────────────────────┐
  │                 AWS S3 Cloud Lake / DB                 │
  │               (Single Source of Truth)                 │
  │  s3://<bucket>/raw/ ──> filtered/ ──> candidates/      │
  │                      ──> features/   ──> models/       │
  └───────────────▲────────────────────────▲───────────────┘
                  │ Read / Write           │ Read / Write
                  │ Parquet & TSV          │ Parquet & Models
  ┌───────────────┴───────────────┐ ┌──────┴───────────────┐
  │            Team A             │ │            Team B    │
  │   Data Pipeline & Features    │ │  Modeling & Submission│
  │   - Text Normalization        │ │  - LightGBM Model     │
  │   - Multilingual Filtering    │ │  - Macro F_0.5 Sweep  │
  │   - Inverted Index Blocking   │ │  - Singleton Logic    │
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
> In AWS, our pipeline uses **strictly Amazon S3** as a central, high-throughput cloud database. Heavy compute runs on **Google Colab or Local Machines**, bypassing SageMaker instance quota limits and minimizing cloud costs.

---

## 2. Central S3 Storage Hierarchy

All raw datasets, filtered tables, candidate pairs, features, and model weights are centrally coordinated through S3:

```text
s3://<your-bucket-name>/
├── raw/                               # Unprocessed source TSV files
│   ├── train_source1.tsv              # ~2.2M entities
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   ├── train_ground_truth.tsv
│   ├── test_source1.tsv               # 1,732,544 test entities
│   ├── test_source2.tsv
│   └── test_source3.tsv
│
├── filtered/                          # Cleaned, normalized, & filtered text data
│   ├── clean_train_s1.parquet
│   ├── clean_train_s23.parquet
│   ├── clean_test_s1.parquet
│   └── clean_test_s23.parquet
│
├── candidates/                        # Country-partitioned blocked pairs (<= 5 per S1)
│   ├── train_candidates_sample50k.parquet
│   ├── train_candidates_full.parquet
│   └── test_candidates_full.parquet
│
├── features/                          # RapidFuzz similarity feature tables
│   ├── sample_50k/                    # Early baseline for Team B (By 1:00 PM Sept 26)
│   │   ├── train_candidates.parquet
│   │   └── val_candidates.parquet
│   └── full/                          # Full competition training set (By 10:00 PM Sept 26)
│       ├── train_candidates.parquet
│       ├── val_candidates.parquet
│       └── test_candidates.parquet
│
├── models/                            # Trained LightGBM weights & hyperparams
│   └── lgbm_model_latest.pkl
│
└── submissions/                       # Generated TSVs and final validated ZIPs
    ├── matching_results.tsv
    ├── candidate_pairs.tsv
    └── final_submission.zip
```

---

## 3. Pipeline Stages & Notebooks

The pipeline is organized in sequential stages within the `notebooks/` directory:

| Notebook | Purpose | Input | Output |
| :--- | :--- | :--- | :--- |
| [`00_s3_setup_test.ipynb`](file:///home/shubham/Projects/Amazon%20ML%20Hackathon/notebooks/00_s3_setup_test.ipynb) | AWS S3 connectivity & integrity checks | S3 bucket credentials | Verified S3 access |
| [`01_eda_multilingual.ipynb`](file:///home/shubham/Projects/Amazon%20ML%20Hackathon/notebooks/01_eda_multilingual.ipynb) | Multilingual exploratory data analysis | Raw TSVs | EDA insights & distributions |
| [`02_train_lightgbm.ipynb`](file:///home/shubham/Projects/Amazon%20ML%20Hackathon/notebooks/02_train_lightgbm.ipynb) | Ground-truth anchored feature store & LightGBM training | Cleaned pairs | `lgbm_model_latest.pkl` |
| [`03_threshold_tuning.ipynb`](file:///home/shubham/Projects/Amazon%20ML%20Hackathon/notebooks/03_threshold_tuning.ipynb) | Macro $F_{0.5}$ decision threshold calibration sweep | Val predictions | Optimal threshold (0.74) |
| [`04_submission.ipynb`](file:///home/shubham/Projects/Amazon%20ML%20Hackathon/notebooks/04_submission.ipynb) | Test inference & submission generation | `test_source*.tsv` | `matching_results.tsv`, `candidate_pairs.tsv` |

---

## 4. Source Code Modules (`src/`)

All reusable production modules are located in `src/`:
- [`src/normalization.py`](file:///home/shubham/Projects/Amazon%20ML%20Hackathon/src/normalization.py): High-performance multilingual text cleaner, legal suffix canonicalization (`pvt ltd`, `gmbh`, `sarl`, etc.), and Unicode normalizer (0.0065 ms/record).
- [`src/blocking.py`](file:///home/shubham/Projects/Amazon%20ML%20Hackathon/src/blocking.py): Country-partitioned inverted index candidate generator with token-frequency ranking and dynamic posting caps.
- [`src/features.py`](file:///home/shubham/Projects/Amazon%20ML%20Hackathon/src/features.py): RapidFuzz C-accelerated similarity feature extractor (Token Sort Ratio, Partial Ratio, Levenshtein, Jaro-Winkler, Prefix Overlap, Address Jaccard).

---

## 5. End-to-End Reproduction Instructions

### Step 1: Environment Setup
```bash
pip install -r requirements.txt
```

### Step 2: Training & Threshold Tuning
1. Run `notebooks/02_train_lightgbm.ipynb` to construct the anchored feature dataset and train the LightGBM classifier.
2. Run `notebooks/03_threshold_tuning.ipynb` to evaluate precision vs. recall across thresholds $[0.50, 0.95]$ and determine the optimal $F_{0.5}$ decision threshold.

### Step 3: Inference & Submission Generation
1. Run `notebooks/04_submission.ipynb` to stream `test_source1.tsv` against the pre-built Source 2/3 inverted index, compute similarity features, apply the calibrated threshold, and emit:
   - `output/matching_results.tsv`
   - `output/candidate_pairs.tsv`

### Step 4: Validate and Package Submission
```bash
# 1. Validate format against official competition rules
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test

# 2. Package into official submission ZIP
python3 utils/package_submission.py \
    --team-name <your_team_name> \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --doc docs/Documentation_template.md
```

---

## 6. Official Submission Archive Structure
```text
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv       # Scored matches
│   └── candidate_pairs.tsv        # Candidate blocking pairs
├── code/
│   └── business_entity_resolution/
│       ├── src/                   # Source code
│       ├── README.md              # Reproduction instructions
│       └── requirements.txt       # Dependencies
└── Documentation_template.md      # Methodology report
```
