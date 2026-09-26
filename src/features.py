"""
src/features.py
High-performance feature extraction pipeline using RapidFuzz (C++ engine).
Extracts 14 string similarity, postal, prefix, and catalog origin features.
Assigns ground truth labels, performs leak-proof GroupShuffleSplit,
downcasts data types (float32/int16/int8), and exports Parquet files directly to S3 or local disk.
"""

import gc
import os
import re
import sys
import time
from typing import Any, Dict, Generator, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from sklearn.model_selection import GroupShuffleSplit

RE_POSTAL = re.compile(r"\b\d{5,6}\b")

FEATURE_COLS: List[str] = [
    "name_ratio",
    "name_partial",
    "name_token_sort",
    "name_token_set",
    "addr_ratio",
    "addr_partial",
    "addr_token_set",
    "name_len_diff",
    "addr_len_diff",
    "pin_exact_match",
    "name_jaro_winkler",
    "name_token_jaccard",
    "first_word_ratio",
    "is_source2"
]


def load_ground_truth_set(
    gt_path: str,
    storage_options: Optional[Dict[str, Any]] = None
) -> Set[Tuple[str, str]]:
    """
    Loads train_ground_truth.tsv into a Python set of (source1_id, matched_id) tuples.
    Provides O(1) lookup speed for positive labeling without memory explosion.
    """
    print(f"Loading ground truth labels from: {gt_path} ...")
    gt_df = pd.read_csv(
        gt_path,
        sep="\t",
        dtype=str,
        storage_options=storage_options
    )

    s1_col = "source1_entity_id" if "source1_entity_id" in gt_df.columns else gt_df.columns[0]
    matched_col = "matched_entity_ids" if "matched_entity_ids" in gt_df.columns else gt_df.columns[1]

    true_pairs: Set[Tuple[str, str]] = set()
    for _, row in gt_df.iterrows():
        s1_id = str(row[s1_col]).strip()
        matched_str = str(row[matched_col]).strip()
        if matched_str and matched_str.lower() != "nan":
            for mid in matched_str.split(","):
                clean_mid = mid.strip()
                if clean_mid:
                    true_pairs.add((s1_id, clean_mid))

    print(f"✅ Loaded {len(true_pairs):,} true match pairs into memory.")
    return true_pairs


def _get_column_values(df: pd.DataFrame, possible_col_names: List[str]) -> List[str]:
    """Helper to retrieve string list from DataFrame given candidate column aliases."""
    for col in possible_col_names:
        if col in df.columns:
            return df[col].fillna("").astype(str).tolist()
    return [""] * len(df)


def extract_postal_code(text: str) -> Optional[str]:
    """Extracts 5/6 digit postal codes (PIN / ZIP) from address string."""
    m = RE_POSTAL.search(text)
    return m.group(0) if m else None


def compute_chunk_features(
    chunk_df: pd.DataFrame,
    true_pairs: Optional[Set[Tuple[str, str]]] = None
) -> pd.DataFrame:
    """
    Computes 14 RapidFuzz similarity metrics, postal matching, and catalog origin features.
    Pre-allocates downcasted NumPy arrays (float32, int16, int8) to cut RAM consumption by 50%.
    """
    n_rows = len(chunk_df)
    if n_rows == 0:
        return pd.DataFrame()

    # Retrieve IDs and text with graceful alias handling
    s1_ids = _get_column_values(chunk_df, ["source1_entity_id", "s1_entity_id", "entity_id"])
    c_ids = _get_column_values(chunk_df, ["candidate_entity_id", "s23_entity_id", "matched_entity_id"])
    s1_names = _get_column_values(chunk_df, ["s1_name", "s1_business_name", "business_name"])
    s23_names = _get_column_values(chunk_df, ["s23_name", "s23_business_name", "candidate_name"])
    s1_addrs = _get_column_values(chunk_df, ["s1_addr", "s1_business_address", "business_address"])
    s23_addrs = _get_column_values(chunk_df, ["s23_addr", "s23_business_address", "candidate_address"])

    # Pre-allocate arrays with downcasted data types
    name_ratio = np.empty(n_rows, dtype=np.float32)
    name_partial = np.empty(n_rows, dtype=np.float32)
    name_token_sort = np.empty(n_rows, dtype=np.float32)
    name_token_set = np.empty(n_rows, dtype=np.float32)

    addr_ratio = np.empty(n_rows, dtype=np.float32)
    addr_partial = np.empty(n_rows, dtype=np.float32)
    addr_token_set = np.empty(n_rows, dtype=np.float32)

    name_len_diff = np.empty(n_rows, dtype=np.int16)
    addr_len_diff = np.empty(n_rows, dtype=np.int16)

    # 5 New Discriminative Features
    pin_exact_match = np.empty(n_rows, dtype=np.float32)
    name_jaro_winkler = np.empty(n_rows, dtype=np.float32)
    name_token_jaccard = np.empty(n_rows, dtype=np.float32)
    first_word_ratio = np.empty(n_rows, dtype=np.float32)
    is_source2 = np.empty(n_rows, dtype=np.int8)

    # Vectorized feature calculation
    for i in range(n_rows):
        n1, n2 = s1_names[i], s23_names[i]
        a1, a2 = s1_addrs[i], s23_addrs[i]
        cid = c_ids[i]

        # Core String Similarity
        name_ratio[i] = fuzz.ratio(n1, n2) / 100.0
        name_partial[i] = fuzz.partial_ratio(n1, n2) / 100.0
        name_token_sort[i] = fuzz.token_sort_ratio(n1, n2) / 100.0
        name_token_set[i] = fuzz.token_set_ratio(n1, n2) / 100.0

        addr_ratio[i] = fuzz.ratio(a1, a2) / 100.0
        addr_partial[i] = fuzz.partial_ratio(a1, a2) / 100.0
        addr_token_set[i] = fuzz.token_set_ratio(a1, a2) / 100.0

        name_len_diff[i] = abs(len(n1) - len(n2))
        addr_len_diff[i] = abs(len(a1) - len(a2))

        # Postal match
        p1 = extract_postal_code(a1)
        p2 = extract_postal_code(a2)
        pin_exact_match[i] = 1.0 if (p1 and p2 and p1 == p2) else 0.0

        # Jaro-Winkler (Prefix heavily weighted)
        name_jaro_winkler[i] = JaroWinkler.similarity(n1, n2)

        # Token Jaccard
        t1, t2 = set(n1.split()), set(n2.split())
        union = t1 | t2
        name_token_jaccard[i] = (len(t1 & t2) / len(union)) if union else 0.0

        # First word ratio
        w1 = n1.split()[0] if n1.split() else ""
        w2 = n2.split()[0] if n2.split() else ""
        first_word_ratio[i] = fuzz.ratio(w1, w2) / 100.0

        # Source 2 vs Source 3 indicator
        is_source2[i] = 1 if cid.startswith("S2-") else 0

    # Assemble feature DataFrame
    feat_df = pd.DataFrame({
        "source1_entity_id": s1_ids,
        "candidate_entity_id": c_ids,
        "name_ratio": name_ratio,
        "name_partial": name_partial,
        "name_token_sort": name_token_sort,
        "name_token_set": name_token_set,
        "addr_ratio": addr_ratio,
        "addr_partial": addr_partial,
        "addr_token_set": addr_token_set,
        "name_len_diff": name_len_diff,
        "addr_len_diff": addr_len_diff,
        "pin_exact_match": pin_exact_match,
        "name_jaro_winkler": name_jaro_winkler,
        "name_token_jaccard": name_token_jaccard,
        "first_word_ratio": first_word_ratio,
        "is_source2": is_source2
    })

    if "country" in chunk_df.columns:
        feat_df["country"] = chunk_df["country"].values

    if true_pairs is not None:
        labels = np.array(
            [(1 if (s1, c) in true_pairs else 0) for s1, c in zip(s1_ids, c_ids)],
            dtype=np.int8
        )
        feat_df["label"] = labels

    return feat_df


def split_and_export_to_s3(
    features_df: pd.DataFrame,
    s3_output_dir: str,
    test_size: float = 0.20,
    random_state: int = 42,
    storage_options: Optional[Dict[str, Any]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs leak-proof GroupShuffleSplit grouped strictly by source1_entity_id.
    Validates zero ID overlap and exports train_candidates.parquet and val_candidates.parquet.
    """
    print("\n" + "=" * 70)
    print("🛡️ LEAK-PROOF GROUP SHUFFLE SPLIT AUDIT")
    print("=" * 70)

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, val_idx = next(gss.split(features_df, groups=features_df["source1_entity_id"]))

    train_df = features_df.iloc[train_idx].reset_index(drop=True)
    val_df = features_df.iloc[val_idx].reset_index(drop=True)

    train_ids = set(train_df["source1_entity_id"].unique())
    val_ids = set(val_df["source1_entity_id"].unique())
    overlap = train_ids.intersection(val_ids)

    print(f"• Total Candidate Pairs Processed : {len(features_df):,}")
    print(f"• Training Set Records (80%)      : {len(train_df):,} ({len(train_ids):,} Unique S1 IDs)")
    print(f"• Validation Set Records (20%)    : {len(val_df):,} ({len(val_ids):,} Unique S1 IDs)")
    print(f"• Leakage Cross-Check (Overlap)   : {len(overlap)} S1 entities")

    if overlap:
        raise ValueError(f"❌ DATA LEAK DETECTED: {len(overlap)} S1 IDs in both splits!")
    print("✅ Zero data leakage confirmed across splits.")

    train_out = f"{s3_output_dir.rstrip('/')}/train_candidates.parquet"
    val_out = f"{s3_output_dir.rstrip('/')}/val_candidates.parquet"

    print(f"\n⏳ Exporting train split to {train_out} ...")
    train_df.to_parquet(train_out, index=False, engine="pyarrow", storage_options=storage_options)
    print("✅ Train split saved.")

    print(f"⏳ Exporting val split to {val_out} ...")
    val_df.to_parquet(val_out, index=False, engine="pyarrow", storage_options=storage_options)
    print("✅ Validation split saved.")

    return train_df, val_df
