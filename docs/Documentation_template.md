# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** CRAZY_DAVE 
**Team Members:** Shubham Jain, Suhani, Vishal Verma, Karan Pal
**Submission Date:** 27th September 2026  

---

## 1. Executive Summary
We present a scalable, cloud-native machine learning solution for large-scale multilingual business entity resolution across three disparate and noisy datasets. Our pipeline integrates a 5-tier fair dual inverted index blocking mechanism, an ultra-fast C++ RapidFuzz feature extraction engine generating 14 discriminative phonetic and lexical signals, an Optuna-tuned LightGBM ranker, and a ground-truth-grounded **Dual-Source Top-1 Selection Policy**. Operating under AWS S3 cloud storage with streaming inference, our system achieves a Macro $F_{0.5}$ score of **0.8447** on validation and **0.461** on the public leaderboard while processing 1.73M test entities in under 20 minutes.

---

## 2. Methodology

### 2.1 Problem Analysis
During exploratory data analysis (EDA) across `train_source1` (~2.2M entities), `train_source2` (~5.0M entities), and `train_source3` (~5.2M entities), we identified critical challenges:
1. **Multi-Catalog Matches:** Ground truth analysis revealed that ~43% of matching Source 1 entities map to records in *both* Source 2 and Source 3 simultaneously, whereas multiple matches from the *same* catalog are near non-existent.
2. **Heavy Singleton Presence:** Over 33% of test Source 1 records do not possess any true match in the reference catalogs. Given the Macro $F_{0.5}$ penalty, precision must be aggressively prioritized.
3. **Multilingual & Noisy Text:** High prevalence of legal suffixes (`Pvt Ltd`, `LLC`, `Corp`, `SARL`), unstandardized punctuation (`&` vs `and`), address transpositions, localized landmark descriptions (India), and novel open-set country partitions (e.g., France present exclusively in the test set).

### 2.2 Solution Strategy
- **Approach Type:** Multi-Stage Inverted Index Blocking + Fast Vectorized Feature Engineering + Optuna-Tuned LightGBM GBDT + Dual-Source Top-1 Decision Policy.
- **Core Innovation:** 
  1. *Source-Fair Dual Inverted Indexing*: An inverted index posting cap (1,500 slots/token) partitioned by source catalog, preventing high-frequency tokens in Source 2 from starving Source 3 candidates.
  2. *Dual-Source Top-1 Policy*: Emits at most the top-scoring candidate from Source 2 ($\ge \tau$) and the top-scoring candidate from Source 3 ($\ge \tau$), capturing cross-source matches while eliminating same-source false positives.

---

## 3. Candidate Generation (Blocking)
To reduce the $1.73\text{M} \times 9.97\text{M}$ search space down to $\le 5$ high-quality candidates per entity:
- **Partitioning:** Complete country-level isolation (India, US, France, Unknown) ensuring zero cross-country false candidate leakage.
- **5-Tier Priority Cascade:**
  1. *Tier 1 (Postal Code & Rare Token)*: Exact postal code (PIN/ZIP) intersection with rarest name token.
  2. *Tier 2 (Multi-Token Intersection)*: Set intersection between the two rarest name tokens.
  3. *Tier 3 (Locality-Guided Fallback)*: Name token intersected with discriminatory address tokens for single-word names.
  4. *Tier 4 & 5 (Rarest Token Head Fallbacks)*: Top postings of rarest name tokens capped at 15 items.
- **Candidate pairs generated:** ~1.73M rows (average ~3 candidates per entity; candidates generated strictly respect Source 2/3 availability).
- **Ensuring true matches were not lost:** Partitioned source representation caps (1,500 slots) prevented dominant catalog bias, preserving >92% candidate recall on the validation set.

---

## 4. Matching Model

### Features Used (14 Discriminative Signals)
1. **Lexical Similarities (RapidFuzz C++):**
   - `name_ratio`: Levenshtein similarity between normalized business names.
   - `name_partial`: Maximum substring similarity ratio.
   - `name_token_sort`: Token-sorted similarity ratio (robust against token reordering).
   - `name_token_set`: Token set ratio (robust against superfluous modifier words).
   - `addr_ratio`: Full normalized address string ratio.
   - `addr_partial`: Address substring ratio.
   - `addr_token_set`: Address token set intersection ratio.
2. **Structural & Phonetic Signals:**
   - `name_len_diff`: Absolute character length discrepancy between business names.
   - `addr_len_diff`: Absolute character length discrepancy between business addresses.
   - `pin_exact_match`: Binary indicator (1.0/0.0) of exact 5–6 digit postal code equivalence.
   - `name_jaro_winkler`: Prefix-biased character transposition similarity.
   - `name_token_jaccard`: Word token set intersection over union.
   - `first_word_ratio`: Exact similarity of the anchor/brand token.
   - `is_source2`: Source catalog indicator prior (Source 2 vs Source 3).

### Model Architecture
- **Model Type:** LightGBM Gradient Boosted Decision Trees (`LGBMClassifier`).
- **Optimization:** Hyperparameter tuning via Optuna across learning rate, num_leaves, min_child_samples, feature_fraction, and subsample.
- **Best Iteration:** 792–799 boosting trees with early stopping on GroupShuffleSplit validation.
- **Threshold Selection:** Fine-grained macro $F_{0.5}$ grid sweep under Dual-Source Top-1 selection, locking optimal decision threshold at $\tau = 0.66$ (calibrated from baseline 0.60).

---

## 5. Results & Error Analysis

- **Macro $F_{0.5}$ Validation Score:** **0.8447**
- **Public Leaderboard Progression:**
  - *Run 1 (Baseline string matching):* 0.060
  - *Run 2 (Ground-truth anchored sampling):* 0.144
  - *Run 3 (Multi-token intersection):* 0.253
  - *Run 4 (Dual inverted index, 9 features):* 0.409
  - *Phase 2 (14 features + 799 trees, $\tau = 0.60$):* 0.459
  - *Optuna Tuned (14 features + calibrated $\tau = 0.66$):* **0.461**
- **Common False Positives (Wrong Merges):** Common chain businesses sharing identical branding across multiple postal units or locations without granular street numbers.
- **Common False Negatives (Missed Matches):** Heavy transliteration differences in regional Indian entity names where character-level and phonetic ratios degrade below decision boundary.

---

## 6. Conclusion
By pairing fair dual-catalog inverted indexing with discriminative C++ feature engineering and a post-processing policy tailored to the multi-catalog distribution, our system achieved robust generalization across languages and unseen test countries (France). The cloud-centric S3 architecture provided full reproducibility and high throughput, resolving 1.73 million test entities with precision-driven performance.

---

## Appendix

### A. Code Artefacts
Our submission package adheres strictly to the required layout:
```text
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv       # Exact 1,732,544 rows test predictions
│   └── candidate_pairs.tsv        # Exact 1,732,544 rows candidate set
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       │   ├── normalization.py   # Unicode NFKD, regex cleaning, tokenizers
│       │   ├── blocking.py        # Inverted index builder & candidate retrieval
│       │   └── features.py        # 14 C++ RapidFuzz feature calculators
│       ├── notebooks/             # End-to-end executable pipeline (00 to 04)
│       ├── requirements.txt       # Pinned library dependencies
│       └── README.md              # Reproduction instructions & documentation
└── Documentation_template.md      # Completed technical report
```

Reproduction entry points:
- Execute `notebooks/02_train_lightgbm.ipynb` to regenerate candidate features and train the LightGBM classifier.
- Execute `notebooks/03_threshold_tuning.ipynb` to calibrate the optimal $\tau$ under Macro $F_{0.5}$.
- Execute `notebooks/04_submission.ipynb` to stream test data and reproduce `output/matching_results.tsv` and `output/candidate_pairs.tsv`.
