#!/usr/bin/env python3
"""Inspect the three expected raw-data folders and write reports."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ml.data.audit import audit_directory, write_audit_report

DATASETS = {
    "AMPds2": "ampds2",
    "Electricity Theft Detection": "electricity_theft",
    "WDSEventDB": "wdseventdb",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit local raw utility datasets without changing them.")
    parser.add_argument("--full-scan", action="store_true", help="Scan every row in CSV/TSV/TXT files for row and missing-value counts.")
    parser.add_argument("--sample-rows", type=int, default=5000, help="Rows to inspect per file in sample mode (default: 5000).")
    parser.add_argument("--data-root", default=str(ROOT / "data" / "raw"), help="Root folder containing dataset subfolders.")
    args = parser.parse_args()
    if args.sample_rows < 1:
        parser.error("--sample-rows must be at least 1")

    data_root = Path(args.data_root)
    audit = {"data_root": str(data_root), "full_scan_requested": args.full_scan, "datasets": {}}
    for display_name, folder in DATASETS.items():
        audit["datasets"][display_name] = audit_directory(
            data_root / folder, sample_rows=args.sample_rows, full_scan=args.full_scan
        )

    reports_dir = ROOT / "reports"
    write_audit_report(audit, reports_dir / "dataset_audit.json", reports_dir / "dataset_audit.md")
    for name, entry in audit["datasets"].items():
        print(f"{name}: {entry['files_found']} tabular file(s) in {entry['root']}")
    print(f"Wrote {reports_dir / 'dataset_audit.md'} and {reports_dir / 'dataset_audit.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
