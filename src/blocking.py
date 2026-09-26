"""
src/blocking.py
High-Recall Dual Inverted Index Candidate Blocking Pipeline.
Generates <= 5 high-precision candidate matching pairs between Source 1 and Source 2/3.
Combines Pure Name Tokens, Dedicated Postal Codes, and Address Locality Tokens.
Ensures fair representation across Source 2 and Source 3 without asymmetric starvation.
"""

import gc
import os
import re
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

RE_POSTAL = re.compile(r"\b\d{5,6}\b")
MIN_TOKEN_LENGTH: int = 3
MAX_CAP_PER_SOURCE: int = 1_500  # Fair 1,500 slots per source guarantees Source 3 is never starved

NAME_STOPWORDS: Set[str] = {
    # English / Multilingual business designations
    "and", "the", "for", "with", "ltd", "pvt", "inc", "corp", "llc", "llp",
    "limited", "private", "corporation", "company", "co", "enterprises",
    "services", "solutions", "international", "group", "technologies",
    "industries", "trading", "associates", "consulting", "holdings",
    # French generic words
    "de", "la", "le", "les", "et", "du", "des", "sarl", "sa", "sas", "societe",
    # Country / Geographic noise
    "india", "usa", "us", "france", "fr", "north", "south", "east", "west",
    # Common category noise words
    "hotel", "restaurant", "store", "shop", "mart", "agency", "travels"
}

ADDR_STOPWORDS: Set[str] = NAME_STOPWORDS | {
    "road", "street", "st", "rd", "lane", "ave", "avenue", "blvd", "boulevard",
    "floor", "bldg", "building", "near", "opp", "opposite", "behind", "beside",
    "post", "dist", "district", "city", "state", "nagar", "colony", "chowk",
    "marg", "cross", "main", "layout", "phase", "sector", "block", "plot",
    "no", "house", "flat", "apartment", "complex", "plaza", "tower", "towers",
    "rue", "passage", "allee", "route", "zone", "industrial", "area"
}


def extract_name_tokens(normalized_name: str) -> List[str]:
    """Extracts discriminative search tokens from a clean business name."""
    if not normalized_name:
        return []
    return [
        t for t in normalized_name.split()
        if len(t) >= MIN_TOKEN_LENGTH and t not in NAME_STOPWORDS and not t.isdigit()
    ]


def extract_addr_tokens(normalized_addr: str) -> List[str]:
    """Extracts discriminative search tokens from a clean address."""
    if not normalized_addr:
        return []
    return [
        t for t in normalized_addr.split()
        if len(t) >= MIN_TOKEN_LENGTH and t not in ADDR_STOPWORDS and not t.isdigit()
    ]


def extract_postal_code(text: str) -> Optional[str]:
    """Extracts 5/6 digit postal codes (PIN / ZIP) with canonical prefix."""
    m = RE_POSTAL.search(text)
    return f"PIN_{m.group(0)}" if m else None


def build_country_dual_index(
    s23_ids: List[str],
    s23_names: List[str],
    s23_addrs: List[str],
    max_cap_per_source: int = MAX_CAP_PER_SOURCE
) -> Tuple[Dict[str, List[int]], Dict[str, List[int]], Dict[str, List[int]]]:
    """
    Builds three isolated inverted indexes on reference pool (S2 + S3):
    1. name_inv: pure business name tokens
    2. postal_inv: exact postal codes (PIN_XXXXX)
    3. addr_inv: discriminative address locality tokens

    Enforces fair per-source quotas based on entity ID prefix (S2 vs S3).
    """
    name_inv: Dict[str, List[int]] = defaultdict(list)
    postal_inv: Dict[str, List[int]] = defaultdict(list)
    addr_inv: Dict[str, List[int]] = defaultdict(list)

    # Per-source posting trackers: (source_prefix, token) -> count
    src_name_counts: Counter = Counter()
    src_addr_counts: Counter = Counter()

    for idx, (eid, name, addr) in enumerate(zip(s23_ids, s23_names, s23_addrs)):
        src_prefix = eid[:2] if len(eid) >= 2 else "XX"

        # 1. Name tokens
        for tok in set(extract_name_tokens(name)):
            if src_name_counts[(src_prefix, tok)] < max_cap_per_source:
                name_inv[tok].append(idx)
                src_name_counts[(src_prefix, tok)] += 1

        # 2. Postal code
        pin = extract_postal_code(addr)
        if pin:
            postal_inv[pin].append(idx)

        # 3. Address tokens
        for tok in set(extract_addr_tokens(addr)):
            if src_addr_counts[(src_prefix, tok)] < max_cap_per_source:
                addr_inv[tok].append(idx)
                src_addr_counts[(src_prefix, tok)] += 1

    return name_inv, postal_inv, addr_inv


def query_candidates_dual_index(
    s1_name: str,
    s1_addr: str,
    name_inv: Dict[str, List[int]],
    postal_inv: Dict[str, List[int]],
    addr_inv: Dict[str, List[int]],
    max_candidates: int = 5
) -> List[int]:
    """
    Queries candidate entities using a 5-tier high-recall hierarchy:
    Tier 1: Postal code (PIN) + Rarest Name Token exact set intersection.
    Tier 2: Multi-token Name intersection (Top 2 rarest name tokens).
    Tier 3: Locality fallback (Rarest name token + rarest address token).
    Tier 4: Single-token name / fallback rarest name token head.
    Tier 5: Second rarest name token head.
    """
    name_tokens = extract_name_tokens(s1_name)
    pin = extract_postal_code(s1_addr)
    addr_tokens = extract_addr_tokens(s1_addr)

    v_name_tokens = [t for t in name_tokens if t in name_inv]
    if not v_name_tokens:
        return []

    # Sort name tokens by rarity (shortest posting list first)
    v_name_tokens.sort(key=lambda t: len(name_inv[t]))

    cand_indices: List[int] = []
    seen_cand: Set[int] = set()

    # Tier 1: Postal code + rarest name token intersection
    if pin and pin in postal_inv:
        pin_postings = set(postal_inv[pin])
        pin_intersect = pin_postings & set(name_inv[v_name_tokens[0]])
        for idx in pin_intersect:
            if idx not in seen_cand:
                seen_cand.add(idx)
                cand_indices.append(idx)
                if len(cand_indices) >= max_candidates:
                    return cand_indices

    # Tier 2: Multi-token name intersection (Top 2 rarest name tokens)
    if len(cand_indices) < max_candidates and len(v_name_tokens) >= 2:
        t0, t1 = v_name_tokens[0], v_name_tokens[1]
        name_intersect = set(name_inv[t0]) & set(name_inv[t1])
        for idx in name_intersect:
            if idx not in seen_cand:
                seen_cand.add(idx)
                cand_indices.append(idx)
                if len(cand_indices) >= max_candidates:
                    return cand_indices

    # Tier 3: Locality fallback (Rarest name token + rarest address token)
    if len(cand_indices) < max_candidates and addr_tokens:
        v_addr_tokens = [t for t in addr_tokens if t in addr_inv]
        if v_addr_tokens:
            v_addr_tokens.sort(key=lambda t: len(addr_inv[t]))
            name_addr_intersect = set(name_inv[v_name_tokens[0]]) & set(addr_inv[v_addr_tokens[0]])
            for idx in name_addr_intersect:
                if idx not in seen_cand:
                    seen_cand.add(idx)
                    cand_indices.append(idx)
                    if len(cand_indices) >= max_candidates:
                        return cand_indices

    # Tier 4: Rarest name token head (up to 15 candidates)
    if len(cand_indices) < max_candidates:
        for idx in name_inv[v_name_tokens[0]][:15]:
            if idx not in seen_cand:
                seen_cand.add(idx)
                cand_indices.append(idx)
                if len(cand_indices) >= max_candidates:
                    return cand_indices

    # Tier 5: Second rarest name token head
    if len(cand_indices) < max_candidates and len(v_name_tokens) >= 2:
        for idx in name_inv[v_name_tokens[1]][:10]:
            if idx not in seen_cand:
                seen_cand.add(idx)
                cand_indices.append(idx)
                if len(cand_indices) >= max_candidates:
                    return cand_indices

    return cand_indices


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
    Returns DataFrame matching required competition candidate schema.
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
        print(f"\n🏗️ Building Dual Inverted Index for Country '{country_code}' (S23 Records: {n_s23:,})...")

    t_start = time.perf_counter()

    # 1. Normalize S23 entities & extract vectors
    s23_id_col = "entity_id" if "entity_id" in df_s23_country.columns else df_s23_country.columns[0]
    s23_name_col = "business_name" if "business_name" in df_s23_country.columns else "name"
    s23_addr_col = "business_address" if "business_address" in df_s23_country.columns else "address"

    s23_ids = df_s23_country[s23_id_col].astype(str).tolist()
    s23_names = [clean_business_name(name) for name in df_s23_country[s23_name_col].fillna("")]
    s23_addrs = [clean_address(addr) for addr in df_s23_country[s23_addr_col].fillna("")]

    # 2. Build Dual Inverted Indexes
    name_inv, postal_inv, addr_inv = build_country_dual_index(s23_ids, s23_names, s23_addrs)
    idx_build_time = time.perf_counter() - t_start
    if verbose:
        print(f"  ✅ Dual index created: {len(name_inv):,} Name Tokens, {len(postal_inv):,} Postal Codes, {len(addr_inv):,} Addr Tokens in {idx_build_time:.2f}s.")

    # 3. Stream S1 records in batches
    s1_id_col = "entity_id" if "entity_id" in df_s1_country.columns else df_s1_country.columns[0]
    s1_name_col = "business_name" if "business_name" in df_s1_country.columns else "name"
    s1_addr_col = "business_address" if "business_address" in df_s1_country.columns else "address"

    s1_ids = df_s1_country[s1_id_col].astype(str).tolist()
    s1_names = [clean_business_name(name) for name in df_s1_country[s1_name_col].fillna("")]
    s1_addrs = [clean_address(addr) for addr in df_s1_country[s1_addr_col].fillna("")]

    all_pairs: List[Dict[str, str]] = []

    if verbose:
        print(f"🔎 Querying {n_s1:,} Source 1 records against dual index (Batch size: {batch_size:,})...")

    t_query = time.perf_counter()
    for start_idx in range(0, n_s1, batch_size):
        end_idx = min(start_idx + batch_size, n_s1)

        for i in range(start_idx, end_idx):
            s1_id = s1_ids[i]
            s1_clean_name = s1_names[i]
            s1_clean_addr = s1_addrs[i]

            candidate_indices = query_candidates_dual_index(
                s1_clean_name, s1_clean_addr,
                name_inv, postal_inv, addr_inv,
                max_candidates=max_candidates
            )

            for s23_idx in candidate_indices:
                all_pairs.append({
                    "source1_entity_id": s1_id,
                    "candidate_entity_id": s23_ids[s23_idx],
                    "s1_name": s1_clean_name,
                    "s23_name": s23_names[s23_idx],
                    "s1_addr": s1_clean_addr,
                    "s23_addr": s23_addrs[s23_idx],
                    "country": country_code
                })

        gc.collect()
        if verbose and n_s1 > batch_size:
            print(f"  ↳ Processed {end_idx:,}/{n_s1:,} Source 1 records ({len(all_pairs):,} candidates generated)...")

    query_time = time.perf_counter() - t_query
    if verbose:
        candidates_per_s1 = len(all_pairs) / max(n_s1, 1)
        print(f"  🚀 Completed in {query_time:.2f}s! Generated {len(all_pairs):,} pairs ({candidates_per_s1:.2f} candidates/S1).")

    del name_inv, postal_inv, addr_inv
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
    Partitions datasets strictly by country, executes dual index blocking,
    concatenates results, and exports directly to Parquet (S3 or local).
    """
    country_col_s1 = "country" if "country" in df_s1.columns else "Country"
    country_col_s23 = "country" if "country" in df_s23.columns else "Country"

    df_s1_clean = df_s1.copy()
    df_s23_clean = df_s23.copy()
    df_s1_clean[country_col_s1] = df_s1_clean[country_col_s1].fillna("UNKNOWN").str.upper().str.strip()
    df_s23_clean[country_col_s23] = df_s23_clean[country_col_s23].fillna("UNKNOWN").str.upper().str.strip()

    countries = sorted(list(set(df_s1_clean[country_col_s1].unique())))
    print("=" * 78)
    print(f"🌐 DUAL INVERTED INDEX BLOCKING: {len(countries)} COUNTRIES DETECTED: {countries}")
    print("=" * 78)

    country_dfs: List[pd.DataFrame] = []

    for c in countries:
        s1_subset = df_s1_clean[df_s1_clean[country_col_s1] == c]
        s23_subset = df_s23_clean[df_s23_clean[country_col_s23] == c]

        if len(s1_subset) > 0 and len(s23_subset) > 0:
            df_part = block_country_partition(
                s1_subset, s23_subset,
                country_code=c,
                max_candidates=max_candidates,
                batch_size=batch_size
            )
            country_dfs.append(df_part)

    if country_dfs:
        final_df = pd.concat(country_dfs, ignore_index=True)
    else:
        final_df = pd.DataFrame(columns=[
            "source1_entity_id", "candidate_entity_id",
            "s1_name", "s23_name", "s1_addr", "s23_addr", "country"
        ])

    print("\n" + "=" * 78)
    print(f"🏁 BLOCKING SUMMARY: {len(final_df):,} Total Candidate Pairs Generated")
    print("=" * 78)

    if output_parquet_path:
        print(f"💾 Saving candidate pairs to {output_parquet_path}...")
        final_df.to_parquet(
            output_parquet_path,
            index=False,
            engine="pyarrow",
            storage_options=storage_options
        )
        print("✅ Parquet candidate store written successfully.")

    return final_df
