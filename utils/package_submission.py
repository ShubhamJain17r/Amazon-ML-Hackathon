#!/usr/bin/env python3
"""
Amazon ML Challenge 2026 — Submission Packager

Packages your final artifacts into the official submission ZIP:
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md

Usage:
    python3 utils/package_submission.py --team-name <team_name> \
        --matching output/matching_results.tsv \
        --candidate output/candidate_pairs.tsv \
        --doc docs/Documentation_template.md
"""

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path


def package_submission(team_name: str, matching_path: str, candidate_path: str, doc_path: str, output_dir: str = "."):
    repo_root = Path(__file__).resolve().parent.parent

    matching_file = Path(matching_path).resolve()
    candidate_file = Path(candidate_path).resolve()
    doc_file = Path(doc_path).resolve()

    if not matching_file.exists():
        print(f"Error: Matching results file not found at {matching_file}")
        sys.exit(1)
    if not candidate_file.exists():
        print(f"Error: Candidate pairs file not found at {candidate_file}")
        sys.exit(1)
    if not doc_file.exists():
        print(f"Error: Documentation file not found at {doc_file}")
        sys.exit(1)

    zip_filename = f"{team_name}_submission.zip"
    zip_out_path = Path(output_dir).resolve() / zip_filename

    print(f"📦 Packaging submission into {zip_out_path} ...")

    with zipfile.ZipFile(zip_out_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        # 1. Output files
        print("  • Adding output/matching_results.tsv")
        zipf.write(matching_file, arcname="output/matching_results.tsv")

        print("  • Adding output/candidate_pairs.tsv")
        zipf.write(candidate_file, arcname="output/candidate_pairs.tsv")

        # 2. Documentation template
        print(f"  • Adding Documentation_template.md (from {doc_file.name})")
        zipf.write(doc_file, arcname="Documentation_template.md")

        # 3. Code folder: code/business_entity_resolution/
        code_prefix = "code/business_entity_resolution"

        readme_file = repo_root / "README.md"
        if readme_file.exists():
            print("  • Adding code/business_entity_resolution/README.md")
            zipf.write(readme_file, arcname=f"{code_prefix}/README.md")

        req_file = repo_root / "requirements.txt"
        if req_file.exists():
            print("  • Adding code/business_entity_resolution/requirements.txt")
            zipf.write(req_file, arcname=f"{code_prefix}/requirements.txt")

        # Add src files
        src_dir = repo_root / "src"
        for py_file in src_dir.rglob("*.py"):
            if "__pycache__" not in str(py_file):
                rel_path = py_file.relative_to(src_dir)
                arcname = f"{code_prefix}/src/{rel_path}"
                print(f"  • Adding {arcname}")
                zipf.write(py_file, arcname=arcname)

    print(f"\n✅ Successfully generated: {zip_out_path} ({zip_out_path.stat().st_size / (1024*1024):.2f} MB)")
    print("Archive Contents:")
    with zipfile.ZipFile(zip_out_path, "r") as zipf:
        for info in zipf.infolist():
            print(f"  - {info.filename} ({info.file_size:,} bytes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Package Amazon ML Challenge submission.")
    parser.add_argument("--team-name", required=True, help="Your official team name")
    parser.add_argument("--matching", default="output/matching_results.tsv", help="Path to matching_results.tsv")
    parser.add_argument("--candidate", default="output/candidate_pairs.tsv", help="Path to candidate_pairs.tsv")
    parser.add_argument("--doc", default="docs/Documentation_template.md", help="Path to filled Documentation_template.md")
    parser.add_argument("--output-dir", default=".", help="Directory to save the zip file")

    args = parser.parse_args()
    package_submission(args.team_name, args.matching, args.candidate, args.doc, args.output_dir)
