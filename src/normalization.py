"""
src/normalization.py
Ultra-fast, robust string normalization module tailored for the Amazon ML Challenge 2026.
Handles noisy English, Indian English, Hindi Devanagari text, and French accented text.
Standardizes legal entity designations, address street types, and eliminates punctuation spam.
"""

import math
import re
import time
import unicodedata
from typing import Any, Optional


# ==============================================================================
# Pre-compiled Regular Expressions for High-Throughput Normalization
# ==============================================================================

# Combining diacritical marks for Latin / European accent stripping (NFKD)
RE_COMBINING_MARKS = re.compile(r"[\u0300-\u036f]")

# Retain alphanumeric characters, whitespace, and Hindi Devanagari range (\u0900-\u097F)
# Everything else (punctuation, symbols, emojis, noise) gets replaced by a space
RE_ALLOWED_CHARS = re.compile(r"[^\w\s\u0900-\u097f]+", re.UNICODE)

# Collapse consecutive whitespace
RE_MULTISPACE = re.compile(r"\s+")

# Business Legal Suffix Canonicalization Patterns (Ordered by specificity)
# Map: private limited variants -> pvt ltd
RE_PVT_LTD = re.compile(
    r"\b(pvt\.?\s*ltd\.?|private\s+limited|p\.?\s*ltd\.?|pvt\s+limited)\b",
    re.IGNORECASE,
)
# Map: corporation / incorporated variants -> inc
RE_INC = re.compile(
    r"\b(corporation|corp\.?|incorporated|inc\.?)\b",
    re.IGNORECASE,
)
# Map: limited variants -> ltd
RE_LTD = re.compile(
    r"\b(limited|ltd\.?)\b",
    re.IGNORECASE,
)
# Map: llp / llc variants -> llc
RE_LLC = re.compile(
    r"\b(llp|llc)\b",
    re.IGNORECASE,
)
# Map: company / co variants -> co
RE_CO = re.compile(
    r"\b(co\.?|company)\b",
    re.IGNORECASE,
)

# Address Street Type Canonicalization Patterns
RE_ADDR_ROAD = re.compile(r"\b(rd\.?|road)\b", re.IGNORECASE)
RE_ADDR_STREET = re.compile(r"\b(st\.?|street)\b", re.IGNORECASE)
RE_ADDR_AVENUE = re.compile(r"\b(ave\.?|avenue)\b", re.IGNORECASE)
RE_ADDR_BLVD = re.compile(r"\b(blvd\.?|boulevard)\b", re.IGNORECASE)


def clean_business_name(text: Any) -> str:
    """
    Cleans and canonicalizes business entity names.
    - Handles null/NaN/non-string inputs gracefully (returns empty string).
    - Performs Unicode NFKD normalization to strip European accents while preserving Hindi Devanagari.
    - Standardizes legal designations ('private limited' -> 'pvt ltd', 'corporation' -> 'inc', etc.).
    - Removes punctuation spam and collapses multiple spaces.
    
    Target execution speed: < 0.05ms per record.
    """
    if text is None or not isinstance(text, str):
        if isinstance(text, float) and math.isnan(text):
            return ""
        if text is None:
            return ""
        text = str(text)

    # Fast-path for empty or whitespace-only strings
    text = text.strip()
    if not text or text.lower() == "nan":
        return ""

    # 1. Unicode NFKD normalization (decompose accented letters like é -> e + \u0301)
    text = unicodedata.normalize("NFKD", text)
    # Strip combining diacritical marks (leaves base ASCII & preserves Devanagari vowels)
    text = RE_COMBINING_MARKS.sub("", text)

    # 2. Lowercase conversion
    text = text.lower()

    # 3. Canonicalize legal suffixes (before stripping punctuation so "pvt. ltd." is caught)
    text = RE_PVT_LTD.sub("pvt ltd", text)
    text = RE_INC.sub("inc", text)
    text = RE_LTD.sub("ltd", text)
    text = RE_LLC.sub("llc", text)

    # 4. Remove punctuation spam & symbols, keeping alphanumeric, whitespace & Devanagari
    text = RE_ALLOWED_CHARS.sub(" ", text)

    # 5. Re-check standard suffixes after punctuation stripping in case spacing changed
    text = RE_PVT_LTD.sub("pvt ltd", text)
    text = RE_INC.sub("inc", text)
    text = RE_LTD.sub("ltd", text)
    text = RE_LLC.sub("llc", text)

    # 6. Collapse multiple consecutive whitespace characters into a single space
    text = RE_MULTISPACE.sub(" ", text).strip()
    return text


def clean_address(text: Any) -> str:
    """
    Cleans and canonicalizes address strings.
    - Standardizes street types: ('road', 'rd', 'rd.') -> 'road'; ('street', 'st') -> 'street'; etc.
    - Decomposes accents while preserving Devanagari script.
    - Strips noisy punctuation while keeping alphanumeric, whitespace, and Devanagari.
    - Collapses consecutive whitespace.
    """
    if text is None or not isinstance(text, str):
        if isinstance(text, float) and math.isnan(text):
            return ""
        if text is None:
            return ""
        text = str(text)

    text = text.strip()
    if not text or text.lower() == "nan":
        return ""

    # 1. Unicode NFKD normalization & accent stripping
    text = unicodedata.normalize("NFKD", text)
    text = RE_COMBINING_MARKS.sub("", text)

    # 2. Lowercase
    text = text.lower()

    # 3. Standardize address street types (prior to punctuation stripping)
    text = RE_ADDR_ROAD.sub("road", text)
    text = RE_ADDR_STREET.sub("street", text)
    text = RE_ADDR_AVENUE.sub("avenue", text)
    text = RE_ADDR_BLVD.sub("blvd", text)

    # 4. Remove excessive punctuation, commas, hashes, hyphens
    text = RE_ALLOWED_CHARS.sub(" ", text)

    # 5. Re-verify street abbreviations after punctuation removal
    text = RE_ADDR_ROAD.sub("road", text)
    text = RE_ADDR_STREET.sub("street", text)
    text = RE_ADDR_AVENUE.sub("avenue", text)
    text = RE_ADDR_BLVD.sub("blvd", text)

    # 6. Collapse whitespace
    text = RE_MULTISPACE.sub(" ", text).strip()
    return text


# ==============================================================================
# Automated Unit Tests & Latency Benchmark
# ==============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("🧪 Running Automated Unit Tests for src/normalization.py ...")
    print("=" * 70)

    # Test 1: None / Empty / NaN handling
    assert clean_business_name(None) == "", "Test 1 failed: None should return empty string"
    assert clean_business_name("") == "", "Test 1 failed: Empty string should return empty string"
    assert clean_business_name(float("nan")) == "", "Test 1 failed: NaN should return empty string"
    assert clean_address(None) == "", "Test 1 failed: Address None should return empty string"
    print("✅ Test 1 Passed: None, empty, and NaN inputs handled cleanly.")

    # Test 2: Hindi Devanagari script preservation
    hindi_name = "रिलायंस इंडस्ट्रीज लिमिटेड"
    cleaned_hindi = clean_business_name(hindi_name)
    assert "रिलायंस" in cleaned_hindi and "इंडस्ट्रीज" in cleaned_hindi, f"Test 2 failed: Devanagari corrupted: {cleaned_hindi}"
    assert cleaned_hindi == "रिलायंस इंडस्ट्रीज लिमिटेड", f"Test 2 failed: Expected 'रिलायंस इंडस्ट्रीज लिमिटेड', got '{cleaned_hindi}'"
    print(f"✅ Test 2 Passed: Hindi Devanagari preserved -> '{cleaned_hindi}'")

    # Test 3: French / European accent decomposition (NFKD)
    french_name = "Société Générale de Banque - SARL"
    cleaned_french = clean_business_name(french_name)
    assert "societe generale" in cleaned_french, f"Test 3 failed: Accents not decomposed: {cleaned_french}"
    assert "é" not in cleaned_french, f"Test 3 failed: Accents still present in: {cleaned_french}"
    print(f"✅ Test 3 Passed: French accents decomposed -> '{cleaned_french}'")

    # Test 4: Legal suffix canonicalization
    assert clean_business_name("Amazon Private Limited") == "amazon pvt ltd", f"Test 4a failed: {clean_business_name('Amazon Private Limited')}"
    assert clean_business_name("Google, Inc.") == "google inc", f"Test 4b failed: {clean_business_name('Google, Inc.')}"
    assert clean_business_name("Tata Steel Limited") == "tata steel ltd", f"Test 4c failed: {clean_business_name('Tata Steel Limited')}"
    assert clean_business_name("Acme Corp.") == "acme inc", f"Test 4d failed: {clean_business_name('Acme Corp.')}"
    assert clean_business_name("Baker & McKenzie LLP") == "baker mckenzie llc", f"Test 4e failed: {clean_business_name('Baker & McKenzie LLP')}"
    assert clean_business_name("Infosys Pvt. Ltd.") == "infosys pvt ltd", f"Test 4f failed: {clean_business_name('Infosys Pvt. Ltd.')}"
    print("✅ Test 4 Passed: Legal suffixes canonicalized (pvt ltd, inc, ltd, llc).")

    # Test 5: Address abbreviation canonicalization
    raw_addr = "123 MG Rd., Suite #400, Near St. Marks St, 5th Ave."
    cleaned_addr = clean_address(raw_addr)
    assert "road" in cleaned_addr, f"Test 5 failed: 'rd' not converted to 'road' in: {cleaned_addr}"
    assert "street" in cleaned_addr, f"Test 5 failed: 'st' not converted to 'street' in: {cleaned_addr}"
    assert "avenue" in cleaned_addr, f"Test 5 failed: 'ave' not converted to 'avenue' in: {cleaned_addr}"
    print(f"✅ Test 5 Passed: Address abbreviations normalized -> '{cleaned_addr}'")

    # Test 6: Punctuation spam and multi-space collapse
    noisy_name = "  *** Flipkart   India    Pvt.   Ltd. !!! ---  "
    cleaned_noisy = clean_business_name(noisy_name)
    assert cleaned_noisy == "flipkart india pvt ltd", f"Test 6 failed: Expected 'flipkart india pvt ltd', got '{cleaned_noisy}'"
    print(f"✅ Test 6 Passed: Punctuation spam and multi-space collapsed -> '{cleaned_noisy}'")

    # Latency Benchmark: Ensure < 0.05ms (50 microseconds) per record
    print("\n" + "=" * 70)
    print("⚡ Running Speed Benchmark (10,000 iterations)...")
    print("=" * 70)
    test_samples = [
        "Flipkart India Pvt. Ltd.",
        "Société Générale S.A.",
        "रिलायंस इंडस्ट्रीज लिमिटेड",
        "Alphabet Incorporated",
        "123 MG Rd, 4th Ave, St. Louis, MO 63101",
    ]
    iterations = 10_000
    t0 = time.perf_counter()
    for _ in range(iterations // len(test_samples)):
        for sample in test_samples:
            _ = clean_business_name(sample)
            _ = clean_address(sample)
    total_time = time.perf_counter() - t0
    avg_ms = (total_time / (iterations * 2)) * 1000

    print(f"• Total Processed : {iterations * 2:,} calls")
    print(f"• Total Time      : {total_time:.4f}s")
    print(f"• Average Latency : {avg_ms:.4f} ms per record (Limit: 0.0500 ms)")
    assert avg_ms < 0.05, f"Performance check failed: {avg_ms:.4f} ms exceeds 0.05ms threshold!"
    print(f"🚀 Speed benchmark PASSED: {avg_ms:.4f} ms is well below the 0.05ms threshold!")
    print("=" * 70)
