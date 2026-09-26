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

## 3. Team Division & Responsibilities

| Sub-Team | Members | Primary Responsibilities | Milestone Deadlines |
|---|---|---|---|
| **Team A** (Data & Features) | **Shubham (Lead)**<br>Data Partner | • Multilingual cleaning (Devanagari, French, legal suffixes)<br>• Country-partitioned blocking (inverted index)<br>• RapidFuzz similarity feature extraction<br>• Central S3 export in Parquet format | **Sept 26, 01:00 PM:** 50k Sample to S3<br>**Sept 26, 10:00 PM:** Full Parquets to S3 |
| **Team B** (Modeling & Eval) | Model Specialist<br>MLOps Engineer | • LightGBM training with `scale_pos_weight`<br>• Precision-heavy Macro $F_{0.5}$ decision threshold sweep<br>• Singleton handling (empty string = 1.0 score)<br>• Format checking via `validate_submission.py` | **Sept 26, 03:00 PM:** Day 1 Baseline TSV<br>**Sept 27, 10:00 PM:** Final Verified ZIP |

---

## 4. Quickstart Guide (Google Colab & S3)

### 1. Open Notebooks via Colab GitHub Integration
Our team uses Colab's native GitHub UI (**Zero terminal commands required**):
1. In Colab: **File $\rightarrow$ Open notebook $\rightarrow$ GitHub tab**.
2. Select repository `ShubhamJain17r/Amazon-ML-Hackathon`.
3. Open your notebook in `notebooks/<YourName>/`.
4. When finished, click **File $\rightarrow$ Save a copy in GitHub** to commit directly to `main`!

### 2. AWS S3 Credentials Configuration
Store your AWS credentials in Colab's left sidebar (**Secrets / Key icon**):
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_DEFAULT_REGION` (`us-east-1`)
- `S3_BUCKET_NAME`

### 3. Verify S3 Connection in Colab
```python
import s3fs
fs = s3fs.S3FileSystem()
print("S3 connection established successfully!")
```

---

## 5. Antigravity AI Pair Programming

We use **Google Antigravity** as our core AI coding assistant.
To keep full context across conversations, start new Antigravity chats with our Master Context reference:
```text
Project: Amazon ML Challenge 2026
Master Context: conversation://afe57c16-937c-4ea0-9b50-31b154b41a0a
Storage: AWS S3 central cloud lake at s3://<BUCKET>/
Compute: Google Colab
```
Refer to [`docs/ai_prompt_playbook.md`](file:///home/shubham/Projects/Amazon%20ML%20Hackathon/docs/ai_prompt_playbook.md) for detailed prompts tailored for every stage of the pipeline.

---

## 6. Documentation & Roadmaps

- 📘 [**Team Setup & Colab-S3 Guide**](file:///home/shubham/Projects/Amazon%20ML%20Hackathon/docs/Team_Setup_Colab_Git_S3_Guide.md): 3-minute setup instructions for Google Colab, GitHub UI, and S3 credentials.
- 📙 [**Team A Data Pipeline Roadmap**](file:///home/shubham/Projects/Amazon%20ML%20Hackathon/docs/Team_A_Data_Pipeline_Roadmap.md): In-depth guide for filtering, text normalization, blocking, and feature extraction.
- 📗 [**Team B Modeling & Submission Roadmap**](file:///home/shubham/Projects/Amazon%20ML%20Hackathon/docs/Team_B_Modeling_Submission_Roadmap.md): In-depth guide for LightGBM training, threshold tuning, and submission formatting.
- 📕 [**AI Prompt Playbook**](file:///home/shubham/Projects/Amazon%20ML%20Hackathon/docs/ai_prompt_playbook.md): Battle-tested Antigravity prompts across every hackathon stage.
- 📋 [**Documentation Template**](file:///home/shubham/Projects/Amazon%20ML%20Hackathon/docs/Documentation_template.md): Official methodology document to include in the final submission zip.

---

## 7. Git Workflow Rules

1. **Use Colab UI:** Save your notebooks via **File $\rightarrow$ Save a copy in GitHub**.
2. **Never commit raw or processed datasets, Parquet files, or model binaries to Git.** (Enforced via `.gitignore` and central S3 storage).
3. Work strictly inside your designated personal directory: `notebooks/<YourName>/`.
