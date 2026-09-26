"""
src/features.py
High-performance feature extraction pipeline using rapidfuzz (C++ engine).
Extracts string similarity features, assigns ground truth labels,
performs leak-proof GroupShuffleSplit, downcasts data types, and exports to Parquet.
"""

import gc
import os
import sys
from typing import Set, Tuple, Generator
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sklearn.model_selection import GroupShuffleSplit


def load_ground_truth_set(gt_path: str) -> Set[Tuple[str, str]]:
    """
    Loads train_ground_truth.tsv into a Python set of (source1_id, matched_id) tuples.
    Provides O(1) lookup speed for labeling without memory explosion.
    """
    print(f"Loading ground truth labels from: {gt_path} ...")
    gt_df = pd.read_csv(
        gt_path,
        sep="\t",
        dtype={"source1_entity_id": str, "matched_entity_ids": str}
    )
    
    true_pairs: Set[Tuple[str, str]] = set()
    for _, row in gt_df.iterrows():
        s1_id = str(row["source1_entity_id"]).strip()
        matched_str = str(row["matched_entity_ids"]).strip()
        if matched_str and matched_str != "nan":
            for mid in matched_str.split(","):
                clean_mid = mid.strip()
                if clean_mid:
                    true_pairs.add((s1_id, clean_mid))
                    
    print(f"Loaded {len(true_pairs):,} true match pairs.")
    return true_pairs


def compute_chunk_features(
    chunk_df: pd.DataFrame,
    true_pairs: Set[Tuple[str, str]] = None
) -> pd.DataFrame:
    """
    Computes rapidfuzz similarity metrics and length differentials on a chunk of candidate pairs.
    Downcasts numerical columns (float32, int16, int8) to minimize RAM usage.
    """
    s1_names = chunk_df["s1_name"].fillna("").astype(str).tolist()
    s23_names = chunk_df["s23_name"].fillna("").astype(str).tolist()
    s1_addrs = chunk_df["s1_addr"].fillna("").astype(str).tolist()
    s23_addrs = chunk_df["s23_addr"].fillna("").astype(str).tolist()

    n_rows = len(chunk_df)

    # 1. Pre-allocate arrays with downcasted data types (float32 saves 50% RAM over float64)
    name_ratio = np.empty(n_rows, dtype=np.float32)
    name_partial = np.empty(n_rows, dtype=np.float32)
    name_token_sort = np.empty(n_rows, dtype=np.float32)
    name_token_set = np.empty(n_rows, dtype=np.float32)

    addr_ratio = np.empty(n_rows, dtype=np.float32)
    addr_partial = np.empty(n_rows, dtype=np.float32)
    addr_token_set = np.empty(n_rows, dtype=np.float32)

    name_len_diff = np.empty(n_rows, dtype=np.int16)
    addr_len_diff = np.empty(n_rows, dtype=np.int16)

    # 2. Vectorized computation using RapidFuzz C++ engine
    for i in range(n_rows):
        n1, n2 = s1_names[i], s23_names[i]
        a1, a2 = s1_addrs[i], s23_addrs[i]

        name_ratio[i] = fuzz.ratio(n1, n2) / 100.0
        name_partial[i] = fuzz.partial_ratio(n1, n2) / 100.0
        name_token_sort[i] = fuzz.token_sort_ratio(n1, n2) / 100.0
        name_token_set[i] = fuzz.token_set_ratio(n1, n2) / 100.0

        addr_ratio[i] = fuzz.ratio(a1, a2) / 100.0
        addr_partial[i] = fuzz.partial_ratio(a1, a2) / 100.0
        addr_token_set[i] = fuzz.token_set_ratio(a1, a2) / 100.0

        name_len_diff[i] = abs(len(n1) - len(n2))
        addr_len_diff[i] = abs(len(a1) - len(a2))

    # 3. Assemble feature DataFrame
    feat_df = pd.DataFrame({
        "source1_entity_id": chunk_df["source1_entity_id"].values,
        "candidate_entity_id": chunk_df["candidate_entity_id"].values,
        "name_ratio": name_ratio,
        "name_partial": name_partial,
        "name_token_sort": name_token_sort,
        "name_token_set": name_token_set,
        "addr_ratio": addr_ratio,
        "addr_partial": addr_partial,
        "addr_token_set": addr_token_set,
        "name_len_diff": name_len_diff,
        "addr_len_diff": addr_len_diff,
    })

    # 4. Assign binary label if ground truth is supplied (Train mode)
    if true_pairs is not None:
        s1_ids = chunk_df["source1_entity_id"].tolist()
        c_ids = chunk_df["candidate_entity_id"].tolist()
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
    storage_options: dict = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs leak-proof GroupShuffleSplit grouped strictly by source1_entity_id.
    Validates zero ID overlap and exports train_candidates.parquet and val_candidates.parquet to S3.
    """
    print("\n" + "=" * 60)
    print("LEAK-PROOF GROUP SHUFFLE SPLIT AUDIT")
    print("=" * 60)

    # Group strictly by source1_entity_id
    groups = features_df["source1_entity_id"]
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, val_idx = next(gss.split(features_df, groups=groups))

    train_df = features_df.iloc[train_idx].reset_index(drop=True)
    val_df = features_df.iloc[val_idx].reset_index(drop=True)

    # Verification: Confirm zero overlap of Source 1 IDs
    train_s1 = set(train_df["source1_entity_id"])
    val_s1 = set(val_df["source1_entity_id"])
    overlap = train_s1.intersection(val_s1)

    print(f"  • Total Candidate Pairs : {len(features_df):,}")
    print(f"  • Train Pairs           : {len(train_df):,} ({len(train_s1):,} unique S1 IDs)")
    print(f"  • Validation Pairs      : {len(val_df):,} ({len(val_s1):,} unique S1 IDs)")
    print(f"  • S1 Overlap Count      : {len(overlap)}")

    if len(overlap) > 0:
        raise ValueError(f"CRITICAL ERROR: Data leakage detected! {len(overlap)} S1 IDs in both splits.")
    print("  ✅ [PASS] Zero S1 data leakage confirmed!")

    # Check positive label distribution
    if "label" in features_df.columns:
        train_pos = (train_df["label"] == 1).sum()
        val_pos = (val_df["label"] == 1).sum()
        print(f"  • Train Positive Matches: {train_pos:,} ({(train_pos/len(train_df))*100:.2f}%)")
        print(f"  • Val Positive Matches  : {val_pos:,} ({(val_pos/len(val_df))*100:.2f}%)")

    # Export to S3 / Target path
    s3_output_dir = s3_output_dir.rstrip("/")
    train_path = f"{s3_output_dir}/train_candidates.parquet"
    val_path = f"{s3_output_dir}/val_candidates.parquet"

    print(f"\nExporting Parquet files to {s3_output_dir} ...")
    train_df.to_parquet(train_path, index=False, engine="pyarrow", storage_options=storage_options)
    print(f"  ✅ Saved: {train_path}")
    val_df.to_parquet(val_path, index=False, engine="pyarrow", storage_options=storage_options)
    print(f"  ✅ Saved: {val_path}")

    return train_df, val_df


def process_and_export_pipeline(
    candidate_generator: Generator[pd.DataFrame, None, None],
    gt_path: str,
    s3_output_dir: str,
    storage_options: dict = None
):
    """
    Main driver: processes candidate pair chunks, extracts features,
    accumulates in memory, and performs final split and S3 export.
    """
    true_pairs = load_ground_truth_set(gt_path) if gt_path else None
    all_chunks = []

    print("Computing features across candidate chunks...")
    for i, chunk in enumerate(candidate_generator, 1):
        feat_chunk = compute_chunk_features(chunk, true_pairs)
        all_chunks.append(feat_chunk)
        if i % 5 == 0:
            print(f"  Processed {i} chunks ({sum(len(c) for c in all_chunks):,} candidate pairs)...")
        gc.collect()

    features_df = pd.concat(all_chunks, ignore_index=True)
    del all_chunks
    gc.collect()

    return split_and_export_to_s3(features_df, s3_output_dir, storage_options=storage_options)


if __name__ == "__main__":
    print("features.py module loaded successfully.")
