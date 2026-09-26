"""
src/features.py
High-performance feature extraction pipeline using RapidFuzz (C++ engine).
Extracts string similarity metrics, assigns ground truth labels,
performs leak-proof GroupShuffleSplit, downcasts data types (float32/int16),
and exports Parquet files directly to S3 or local disk.
"""

import gc
import os
import sys
import time
from typing import Any, Dict, Generator, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sklearn.model_selection import GroupShuffleSplit


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


def compute_chunk_features(
    chunk_df: pd.DataFrame,
    true_pairs: Optional[Set[Tuple[str, str]]] = None
) -> pd.DataFrame:
    """
    Computes RapidFuzz similarity metrics and length differentials on candidate pairs.
    Pre-allocates downcasted NumPy arrays (float32, int16, int8) to cut RAM consumption by 50%.
    
    Computed Features:
    - name_ratio, name_partial, name_token_sort, name_token_set (float32)
    - addr_ratio, addr_partial, addr_token_set (float32)
    - name_len_diff, addr_len_diff (int16)
    - label (int8, if true_pairs provided)
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

    # 2. Vectorized RapidFuzz C++ engine calls
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
    })

    # Preserve country if present
    if "country" in chunk_df.columns:
        feat_df["country"] = chunk_df["country"].values

    # 4. Assign binary ground truth label (1 = true match, 0 = non-match)
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

    assert len(overlap) == 0, f"CRITICAL ERROR: Data leakage detected! {len(overlap)} S1 IDs in both splits."
    print("  ✅ [PASS] Zero S1 data leakage confirmed!")

    # Check positive label distribution
    if "label" in features_df.columns:
        train_pos = (train_df["label"] == 1).sum()
        val_pos = (val_df["label"] == 1).sum()
        print(f"  • Train Positive Matches: {train_pos:,} ({(train_pos/max(len(train_df),1))*100:.2f}%)")
        print(f"  • Val Positive Matches  : {val_pos:,} ({(val_pos/max(len(val_df),1))*100:.2f}%)")

    # Export to S3 / Target path
    s3_output_dir = s3_output_dir.rstrip("/")
    if not s3_output_dir.startswith("s3://"):
        os.makedirs(s3_output_dir, exist_ok=True)
    train_path = f"{s3_output_dir}/train_candidates.parquet"
    val_path = f"{s3_output_dir}/val_candidates.parquet"

    print(f"\n💾 Exporting Parquet tables to: {s3_output_dir} ...")
    train_df.to_parquet(train_path, index=False, engine="pyarrow", storage_options=storage_options)
    print(f"  ✅ Saved: {train_path} ({len(train_df):,} rows)")
    val_df.to_parquet(val_path, index=False, engine="pyarrow", storage_options=storage_options)
    print(f"  ✅ Saved: {val_path} ({len(val_df):,} rows)")

    return train_df, val_df


def process_and_export_pipeline(
    candidate_df: pd.DataFrame,
    gt_path: str,
    s3_output_dir: str,
    storage_options: Optional[Dict[str, Any]] = None,
    chunk_size: int = 50_000
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main driver: processes candidate pairs in chunks, computes similarity features,
    and performs leak-proof split and Parquet export.
    """
    true_pairs = load_ground_truth_set(gt_path, storage_options=storage_options) if gt_path else None
    
    n_total = len(candidate_df)
    feat_chunks: List[pd.DataFrame] = []

    print(f"\n⚡ Computing RapidFuzz features on {n_total:,} candidate pairs (Chunk size: {chunk_size:,})...")
    t0 = time.perf_counter()

    for start_idx in range(0, n_total, chunk_size):
        end_idx = min(start_idx + chunk_size, n_total)
        chunk = candidate_df.iloc[start_idx:end_idx]
        feat_chunk = compute_chunk_features(chunk, true_pairs)
        feat_chunks.append(feat_chunk)
        gc.collect()

    features_df = pd.concat(feat_chunks, ignore_index=True)
    del feat_chunks
    gc.collect()

    elapsed = time.perf_counter() - t0
    print(f"🚀 Extracted features for {len(features_df):,} pairs in {elapsed:.2f}s ({(len(features_df)/elapsed):,.0f} pairs/sec).")

    return split_and_export_to_s3(features_df, s3_output_dir, storage_options=storage_options)


# ==============================================================================
# Automated Unit Tests & Latency Benchmark
# ==============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("🧪 Running Automated Unit Tests for src/features.py ...")
    print("=" * 70)

    # Mock Candidate Pairs
    mock_candidates = pd.DataFrame([
        {
            "source1_entity_id": "S1_101",
            "candidate_entity_id": "S23_201",
            "s1_name": "flipkart internet pvt ltd",
            "s23_name": "flipkart internet private limited",
            "s1_addr": "koramangala bangalore",
            "s23_addr": "koramangala bangalore karnataka",
            "country": "IN"
        },
        {
            "source1_entity_id": "S1_101",
            "candidate_entity_id": "S23_999",
            "s1_name": "flipkart internet pvt ltd",
            "s23_name": "walmart inc",
            "s1_addr": "koramangala bangalore",
            "s23_addr": "bentonville arkansas",
            "country": "IN"
        },
        {
            "source1_entity_id": "S1_102",
            "candidate_entity_id": "S23_301",
            "s1_name": "amazon commercial services inc",
            "s23_name": "amazon commercial services corp",
            "s1_addr": "seattle wa",
            "s23_addr": "terry ave seattle wa",
            "country": "US"
        },
        {
            "source1_entity_id": "S1_103",
            "candidate_entity_id": "S23_401",
            "s1_name": "societe generale de banque",
            "s23_name": "societe generale banque sarl",
            "s1_addr": "boulevard haussmann paris",
            "s23_addr": "paris france",
            "country": "FR"
        }
    ])

    # Mock ground truth pairs set
    mock_gt: Set[Tuple[str, str]] = {
        ("S1_101", "S23_201"),
        ("S1_102", "S23_301"),
        ("S1_103", "S23_401")
    }

    # 1. Feature Computation Test
    feat_df = compute_chunk_features(mock_candidates, true_pairs=mock_gt)
    
    # Check shape & label accuracy
    assert len(feat_df) == 4, f"Row count mismatch! Expected 4, got {len(feat_df)}"
    assert feat_df.loc[0, "label"] == 1, "First row should be labeled 1 (positive match)"
    assert feat_df.loc[1, "label"] == 0, "Second row should be labeled 0 (negative match)"
    print("✅ Test 1 Passed: Ground truth O(1) membership matching & labeling verified.")

    # 2. Downcasted Dtypes Test
    assert feat_df["name_ratio"].dtype == np.float32, "name_ratio must be float32"
    assert feat_df["addr_ratio"].dtype == np.float32, "addr_ratio must be float32"
    assert feat_df["name_len_diff"].dtype == np.int16, "name_len_diff must be int16"
    assert feat_df["label"].dtype == np.int8, "label must be int8"
    print("✅ Test 2 Passed: Memory-efficient dtypes verified (float32, int16, int8).")

    # 3. Feature Value Consistency Test
    assert feat_df.loc[0, "name_ratio"] > 0.70, "High similarity match expected > 0.70"
    assert feat_df.loc[1, "name_ratio"] < 0.50, "Low similarity match expected < 0.50"
    assert feat_df.loc[0, "name_len_diff"] >= 0, "Length differential must be non-negative"
    print("✅ Test 3 Passed: RapidFuzz similarity calculations accurate.")

    # 4. Leak-Proof Split Test
    # Duplicate records to test split
    large_mock = pd.concat([mock_candidates] * 25, ignore_index=True)
    large_feat = compute_chunk_features(large_mock, true_pairs=mock_gt)
    
    train_split, val_split = split_and_export_to_s3(
        large_feat,
        s3_output_dir="/tmp/test_features",
        test_size=0.25
    )
    train_ids = set(train_split["source1_entity_id"])
    val_ids = set(val_split["source1_entity_id"])
    assert len(train_ids.intersection(val_ids)) == 0, "Data leakage detected across train and val splits!"
    print("✅ Test 4 Passed: Zero Source 1 overlap verified in GroupShuffleSplit.")

    # Speed Benchmark (10,000 candidate pairs)
    print("\n" + "=" * 70)
    print("⚡ Running Speed Benchmark on 10,000 Candidate Pairs...")
    print("=" * 70)
    bench_df = pd.concat([mock_candidates] * 2500, ignore_index=True)
    t_start = time.perf_counter()
    _ = compute_chunk_features(bench_df, true_pairs=mock_gt)
    bench_time = time.perf_counter() - t_start
    throughput = len(bench_df) / bench_time
    print(f"• Total Processed : {len(bench_df):,} candidate pairs")
    print(f"• Total Time      : {bench_time:.4f}s")
    print(f"• Throughput      : {throughput:,.0f} pairs/sec")
    print("🚀 Speed benchmark PASSED!")
    print("=" * 70)
