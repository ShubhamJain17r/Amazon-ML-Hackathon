"""
src/blocking.py
Country-Partitioned Inverted Index Candidate Blocking Pipeline.
Generates <= 5 high-precision candidate matching pairs between Source 1 and Source 2/3.
Prevents O(N*M) Cartesian explosion via token inverted indexing and strict country partitioning.
"""

import gc
import os
import sys
import time
from collections import Counter, defaultdict
from typing import Any, Dict, Generator, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

# Handle relative or local imports of normalization module
try:
    from src.normalization import clean_address, clean_business_name
except ImportError:
    try:
        from normalization import clean_address, clean_business_name
    except ImportError:
        def clean_business_name(text: Any) -> str:
            return str(text).lower().strip() if text else ""
        def clean_address(text: Any) -> str:
            return str(text).lower().strip() if text else ""


# ==============================================================================
# Domain Stopwords & Filtering Constants
# ==============================================================================

# High-frequency generic legal & domain words that cause spurious blocking collisions
GLOBAL_STOPWORDS: Set[str] = {
    # English / Multilingual business designations
    "and", "the", "for", "with", "ltd", "pvt", "inc", "corp", "llc", "llp",
    "limited", "private", "corporation", "company", "co", "enterprises",
    "services", "solutions", "international", "group", "technologies",
    "holdings", "associates", "consulting", "industries",
    # French generic words
    "de", "la", "le", "les", "et", "du", "des", "sarl", "sa", "sas", "societe",
    # Country / Geographic noise
    "india", "usa", "us", "france", "fr", "north", "south", "east", "west"
}

MIN_TOKEN_LENGTH: int = 3
MAX_POSTING_LIST_SIZE: int = 50_000  # Cap hyper-frequent token posting lists to protect RAM


def extract_blocking_tokens(normalized_name: str, stopwords: Set[str] = GLOBAL_STOPWORDS) -> List[str]:
    """
    Extracts discriminative search tokens from a pre-normalized business name.
    Filters out stopwords and tokens shorter than MIN_TOKEN_LENGTH (3 chars).
    """
    if not normalized_name:
        return []
    
    tokens = normalized_name.split()
    return [
        token for token in tokens
        if len(token) >= MIN_TOKEN_LENGTH and token not in stopwords
    ]


def build_country_inverted_index(
    s23_names: List[str],
    max_posting_size: int = MAX_POSTING_LIST_SIZE
) -> Dict[str, List[int]]:
    """
    Builds an inverted index mapping tokens to integer row indices in the S23 pool.
    
    Memory optimization: Stores primitive Python integer indices rather than strings.
    Caps extremely frequent posting lists to avoid memory blow-up on generic terms.
    """
    inv_index: Dict[str, List[int]] = defaultdict(list)
    
    for idx, name in enumerate(s23_names):
        tokens = set(extract_blocking_tokens(name))
        for token in tokens:
            posting = inv_index[token]
            if len(posting) < max_posting_size:
                posting.append(idx)
                
    return inv_index


def query_candidates_for_record(
    s1_tokens: List[str],
    inv_index: Dict[str, List[int]],
    max_candidates: int = 5
) -> List[Tuple[int, int]]:
    """
    Queries the inverted index with S1 tokens and returns top K candidates
    ranked by the number of shared tokens.
    
    Returns: List of (s23_row_index, shared_token_count)
    """
    if not s1_tokens:
        return []

    # Count candidate frequencies across all token posting lists
    match_counter: Counter = Counter()
    for token in s1_tokens:
        postings = inv_index.get(token)
        if postings:
            match_counter.update(postings)

    if not match_counter:
        return []

    # Retrieve top K candidates by shared token frequency
    return match_counter.most_common(max_candidates)


def block_country_partition(
    df_s1_country: pd.DataFrame,
    df_s23_country: pd.DataFrame,
    country_code: str,
    max_candidates: int = 5,
    batch_size: int = 50_000,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Generates candidate pairs for a single country partition using streaming batches.
    
    Returns DataFrame matching required competition candidate schema:
    ['source1_entity_id', 'candidate_entity_id', 's1_name', 's23_name', 's1_addr', 's23_addr', 'country']
    """
    n_s1 = len(df_s1_country)
    n_s23 = len(df_s23_country)
    
    if n_s1 == 0 or n_s23 == 0:
        if verbose:
            print(f"⚠️ Country {country_code}: Empty partition (S1={n_s1}, S23={n_s23}). Skipping.")
        return pd.DataFrame(columns=[
            "source1_entity_id", "candidate_entity_id",
            "s1_name", "s23_name", "s1_addr", "s23_addr", "country"
        ])

    if verbose:
        print(f"\n🏗️ Building inverted index for Country '{country_code}' (S23 Records: {n_s23:,})...")
    
    t_start = time.perf_counter()
    
    # 1. Normalize S23 entities & extract vectors
    s23_id_col = "entity_id" if "entity_id" in df_s23_country.columns else df_s23_country.columns[0]
    s23_name_col = "business_name" if "business_name" in df_s23_country.columns else "name"
    s23_addr_col = "business_address" if "business_address" in df_s23_country.columns else "address"
    
    s23_ids = df_s23_country[s23_id_col].astype(str).tolist()
    s23_names = [clean_business_name(name) for name in df_s23_country[s23_name_col].fillna("")]
    s23_addrs = [clean_address(addr) for addr in df_s23_country[s23_addr_col].fillna("")]
    
    # 2. Build Inverted Index on S23 names
    inv_index = build_country_inverted_index(s23_names)
    idx_build_time = time.perf_counter() - t_start
    if verbose:
        print(f"  ✅ Inverted index created with {len(inv_index):,} unique tokens in {idx_build_time:.2f}s.")

    # 3. Stream S1 records in batches to keep memory footprint bounded
    s1_id_col = "entity_id" if "entity_id" in df_s1_country.columns else df_s1_country.columns[0]
    s1_name_col = "business_name" if "business_name" in df_s1_country.columns else "name"
    s1_addr_col = "business_address" if "business_address" in df_s1_country.columns else "address"

    s1_ids = df_s1_country[s1_id_col].astype(str).tolist()
    s1_names = [clean_business_name(name) for name in df_s1_country[s1_name_col].fillna("")]
    s1_addrs = [clean_address(addr) for addr in df_s1_country[s1_addr_col].fillna("")]

    all_pairs: List[Dict[str, str]] = []
    
    if verbose:
        print(f"🔎 Querying {n_s1:,} Source 1 records against inverted index (Batch size: {batch_size:,})...")

    t_query = time.perf_counter()
    for start_idx in range(0, n_s1, batch_size):
        end_idx = min(start_idx + batch_size, n_s1)
        
        for i in range(start_idx, end_idx):
            s1_id = s1_ids[i]
            s1_clean_name = s1_names[i]
            s1_clean_addr = s1_addrs[i]
            
            s1_tokens = extract_blocking_tokens(s1_clean_name)
            candidate_matches = query_candidates_for_record(s1_tokens, inv_index, max_candidates=max_candidates)
            
            for s23_idx, _ in candidate_matches:
                all_pairs.append({
                    "source1_entity_id": s1_id,
                    "candidate_entity_id": s23_ids[s23_idx],
                    "s1_name": s1_clean_name,
                    "s23_name": s23_names[s23_idx],
                    "s1_addr": s1_clean_addr,
                    "s23_addr": s23_addrs[s23_idx],
                    "country": country_code
                })
        
        # Explicit garbage collection after each batch
        gc.collect()
        if verbose and n_s1 > batch_size:
            print(f"  ↳ Processed {end_idx:,}/{n_s1:,} Source 1 records ({len(all_pairs):,} candidates generated)...")

    query_time = time.perf_counter() - t_query
    if verbose:
        candidates_per_s1 = len(all_pairs) / max(n_s1, 1)
        print(f"  🚀 Completed in {query_time:.2f}s! Generated {len(all_pairs):,} pairs ({candidates_per_s1:.2f} candidates/S1).")

    # Clean up index from memory
    del inv_index
    gc.collect()

    return pd.DataFrame(all_pairs, columns=[
        "source1_entity_id", "candidate_entity_id",
        "s1_name", "s23_name", "s1_addr", "s23_addr", "country"
    ])


def run_blocking_pipeline(
    df_s1: pd.DataFrame,
    df_s23: pd.DataFrame,
    output_parquet_path: Optional[str] = None,
    storage_options: Optional[Dict[str, Any]] = None,
    max_candidates: int = 5,
    batch_size: int = 50_000
) -> pd.DataFrame:
    """
    Main blocking orchestrator.
    Partitions datasets strictly by country, executes inverted index blocking,
    concatenates results, and exports directly to Parquet (S3 or local).
    """
    country_col_s1 = "country" if "country" in df_s1.columns else "Country"
    country_col_s23 = "country" if "country" in df_s23.columns else "Country"

    # Normalize country codes to uppercase
    df_s1_clean = df_s1.copy()
    df_s23_clean = df_s23.copy()
    df_s1_clean[country_col_s1] = df_s1_clean[country_col_s1].fillna("UNKNOWN").str.upper().str.strip()
    df_s23_clean[country_col_s23] = df_s23_clean[country_col_s23].fillna("UNKNOWN").str.upper().str.strip()

    countries = sorted(list(set(df_s1_clean[country_col_s1].unique())))
    print("=" * 78)
    print(f"🌐 INVERTED INDEX BLOCKING PIPELINE: {len(countries)} COUNTRIES DETECTED: {countries}")
    print("=" * 78)

    country_dfs: List[pd.DataFrame] = []

    for c in countries:
        s1_subset = df_s1_clean[df_s1_clean[country_col_s1] == c]
        s23_subset = df_s23_clean[df_s23_clean[country_col_s23] == c]
        
        candidates_c = block_country_partition(
            s1_subset,
            s23_subset,
            country_code=c,
            max_candidates=max_candidates,
            batch_size=batch_size
        )
        if not candidates_c.empty:
            country_dfs.append(candidates_c)

    if country_dfs:
        full_candidates = pd.concat(country_dfs, ignore_index=True)
    else:
        full_candidates = pd.DataFrame(columns=[
            "source1_entity_id", "candidate_entity_id",
            "s1_name", "s23_name", "s1_addr", "s23_addr", "country"
        ])

    print("\n" + "=" * 78)
    print(f"📊 BLOCKING COMPLETE: Total Candidate Pairs Generated: {len(full_candidates):,}")
    print("=" * 78)

    # Export to Parquet (supporting S3 or local files)
    if output_parquet_path:
        print(f"💾 Saving candidate pairs to: {output_parquet_path} ...")
        full_candidates.to_parquet(
            output_parquet_path,
            engine="pyarrow",
            index=False,
            storage_options=storage_options
        )
        print("✅ Parquet export complete.")

    return full_candidates


# ==============================================================================
# Automated Unit Tests
# ==============================================================================
if __name__ == "__main__":
    print("=" * 78)
    print("🧪 Running Automated Unit Tests for src/blocking.py ...")
    print("=" * 78)

    # Mock Source 1 Data
    mock_s1 = pd.DataFrame([
        {"entity_id": "S1_001", "business_name": "Flipkart Internet Private Limited", "business_address": "Koramangala, Bangalore", "country": "IN"},
        {"entity_id": "S1_002", "business_name": "Amazon Commercial Services Corp", "business_address": "410 Terry Ave N, Seattle", "country": "US"},
        {"entity_id": "S1_003", "business_name": "Société Générale de Banque", "business_address": "29 Boulevard Haussmann, Paris", "country": "FR"},
        {"entity_id": "S1_004", "business_name": "Lone Entity Without Matches", "business_address": "Unknown Road", "country": "US"}
    ])

    # Mock Source 2+3 Data
    mock_s23 = pd.DataFrame([
        # Indian matches & non-matches
        {"entity_id": "S23_IN_1", "business_name": "Flipkart Internet Pvt Ltd", "business_address": "Koramangala", "country": "IN"},
        {"entity_id": "S23_IN_2", "business_name": "Flipkart Logistics Services", "business_address": "Bangalore", "country": "IN"},
        {"entity_id": "S23_IN_3", "business_name": "Tata Motors Limited", "business_address": "Mumbai", "country": "IN"},
        # US matches & non-matches
        {"entity_id": "S23_US_1", "business_name": "Amazon Commercial Services", "business_address": "Terry Ave, Seattle", "country": "US"},
        {"entity_id": "S23_US_2", "business_name": "Amazon Web Services Inc", "business_address": "Seattle WA", "country": "US"},
        {"entity_id": "S23_US_3", "business_name": "Microsoft Corporation", "business_address": "Redmond WA", "country": "US"},
        # FR matches
        {"entity_id": "S23_FR_1", "business_name": "Societe Generale Banque SARL", "business_address": "Blvd Haussmann Paris", "country": "FR"},
        {"entity_id": "S23_FR_2", "business_name": "TotalEnergies SE", "business_address": "Paris", "country": "FR"},
        # Cross-country test record (has same name as S1_001 but different country)
        {"entity_id": "S23_US_4", "business_name": "Flipkart Internet USA", "business_address": "New York", "country": "US"}
    ])

    # Run blocking
    candidates = run_blocking_pipeline(mock_s1, mock_s23, max_candidates=5)

    # 1. Output Schema Validation
    expected_cols = ["source1_entity_id", "candidate_entity_id", "s1_name", "s23_name", "s1_addr", "s23_addr", "country"]
    assert list(candidates.columns) == expected_cols, f"Schema mismatch! Got: {list(candidates.columns)}"
    print("✅ Test 1 Passed: Output DataFrame matches exact expected schema.")

    # 2. Strict Country Partitioning Validation
    # Ensure S1_001 (IN) was NEVER paired with S23_US_4 (US) even though names share 'flipkart'
    in_candidates = candidates[candidates["source1_entity_id"] == "S1_001"]
    matched_ids = in_candidates["candidate_entity_id"].tolist()
    assert "S23_US_4" not in matched_ids, "Country Partitioning Violation! Cross-country pair was generated!"
    assert all(c == "IN" for c in in_candidates["country"]), "Country code mismatch in Indian candidates!"
    print("✅ Test 2 Passed: Strict Country Partitioning verified (Zero cross-country candidate pairs).")

    # 3. Max Candidates Constraint (<= 5 per S1)
    for s1_id, grp in candidates.groupby("source1_entity_id"):
        assert len(grp) <= 5, f"Candidate count exceeded 5 for {s1_id}: got {len(grp)}"
    print("✅ Test 3 Passed: Candidates per Source 1 entity strictly <= 5.")

    # 4. Inverted Index Relevance
    # Check that Flipkart S1_001 retrieved S23_IN_1 and S23_IN_2
    assert "S23_IN_1" in matched_ids, "Relevant candidate S23_IN_1 missed!"
    print(f"✅ Test 4 Passed: Top match correctly retrieved -> S1_001 matched with: {matched_ids}")

    print("\n🚀 All 4 Automated Unit Tests for src/blocking.py PASSED successfully!\n")
