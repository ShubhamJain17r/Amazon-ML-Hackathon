#!/usr/bin/env python3
"""
Amazon ML Challenge 2026 — Dual-Source Top-1 Filter

Post-processes matching_results.tsv to enforce:
  At most 1 match from Source 2 (best S2-xxx)
  At most 1 match from Source 3 (best S3-xxx)

Eliminates same-source duplicate false positives while preserving 100% of dual-source recall.

Usage:
    python3 utils/filter_dual_source.py --input matching_results.tsv --output matching_results_dual_top1.tsv
"""

import argparse
import sys
from pathlib import Path


def filter_dual_source(input_path: str, output_path: str):
    in_file = Path(input_path).resolve()
    out_file = Path(output_path).resolve()

    if not in_file.exists():
        print(f"Error: Input file {in_file} does not exist.")
        sys.exit(1)

    print(f"Processing {in_file} -> {out_file} ...")
    total = 0
    matched = 0
    s2_only = 0
    s3_only = 0
    both = 0

    with open(in_file, "r", encoding="utf-8") as fin, open(out_file, "w", encoding="utf-8") as fout:
        header = fin.readline()
        fout.write(header)

        for line in fin:
            total += 1
            parts = line.strip("\n").split("\t")
            s1_id = parts[0]

            if len(parts) > 1 and parts[1].strip():
                ids = parts[1].split(",")
                best_s2 = None
                best_s3 = None

                for x in ids:
                    clean_x = x.strip()
                    if clean_x.startswith("S2-") and best_s2 is None:
                        best_s2 = clean_x
                    elif clean_x.startswith("S3-") and best_s3 is None:
                        best_s3 = clean_x

                chosen = [x for x in [best_s2, best_s3] if x is not None]
                if chosen:
                    matched += 1
                    if best_s2 and best_s3:
                        both += 1
                    elif best_s2:
                        s2_only += 1
                    elif best_s3:
                        s3_only += 1
                    fout.write(f"{s1_id}\t{','.join(chosen)}\n")
                else:
                    fout.write(f"{s1_id}\t\n")
            else:
                fout.write(f"{s1_id}\t\n")

    print(f"\n✅ Completed successfully!")
    print(f"• Total Rows Processed : {total:,}")
    print(f"• Matched Rows          : {matched:,} ({matched/total*100:.2f}%)")
    print(f"• Empty Rows (Singletons): {total - matched:,} ({(total - matched)/total*100:.2f}%)")
    print(f"• Source 2 Only Matches : {s2_only:,}")
    print(f"• Source 3 Only Matches : {s3_only:,}")
    print(f"• Dual Matches (S2+S3)  : {both:,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter matching TSV to top-1 per source.")
    parser.add_argument("--input", default="matching_results.tsv", help="Input TSV path")
    parser.add_argument("--output", default="matching_results_dual_top1.tsv", help="Output TSV path")
    args = parser.parse_args()
    filter_dual_source(args.input, args.output)
