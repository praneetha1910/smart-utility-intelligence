"""Schema and sample-based quality audits for raw utility datasets."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import pandas as pd

from .loaders import discover_data_files, load_table, read_csv_chunks

TIMESTAMP_TOKENS = ("timestamp", "datetime", "date", "time", "unix_ts", "^ts$")
LABEL_TOKENS = ("label", "target", "class", "flag", "theft", "leak", "event", "attack", "status", "fault")
ID_TOKENS = ("id", "meter", "customer", "consumer", "pipe", "sensor")


def _candidates(columns: list[str], tokens: tuple[str, ...]) -> list[str]:
    out = []
    for col in columns:
        normalized = str(col).strip().lower().replace(" ", "_")
        if any((token.startswith("^") and normalized == token.strip("^")) or (not token.startswith("^") and token in normalized) for token in tokens):
            out.append(str(col))
    return out


def _safe_json(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            pass
    if pd.isna(value):
        return None
    return str(value) if not isinstance(value, (str, int, float, bool, type(None))) else value


def audit_dataframe(df: pd.DataFrame, *, source_path: str, sample_limit: int = 5) -> dict[str, Any]:
    """Summarize a DataFrame without changing it."""
    columns = [str(c) for c in df.columns]
    numeric = df.select_dtypes(include="number")
    sample = df.head(sample_limit).to_dict(orient="records")
    return {
        "file": source_path,
        "rows_in_sample": int(len(df)),
        "columns_count": int(len(columns)),
        "columns": [
            {
                "name": str(col),
                "dtype": str(df[col].dtype),
                "missing_in_sample": int(df[col].isna().sum()),
                "missing_pct_in_sample": round(float(df[col].isna().mean() * 100), 3),
                "unique_in_sample": int(df[col].nunique(dropna=True)),
            }
            for col in df.columns
        ],
        "duplicate_rows_in_sample": int(df.duplicated().sum()),
        "numeric_columns": [str(c) for c in numeric.columns],
        "constant_columns_in_sample": [str(c) for c in df.columns if df[c].nunique(dropna=False) <= 1],
        "timestamp_candidates": _candidates(columns, TIMESTAMP_TOKENS),
        "label_candidates": _candidates(columns, LABEL_TOKENS),
        "id_candidates": _candidates(columns, ID_TOKENS),
        "sample_rows": [{str(k): _safe_json(v) for k, v in row.items()} for row in sample],
    }


def audit_file(path: str | Path, *, sample_rows: int = 5000, full_scan: bool = False) -> dict[str, Any]:
    """Audit one supported file; full_scan scans row counts and missing values for delimited files."""
    file_path = Path(path)
    size_bytes = file_path.stat().st_size
    sample = load_table(file_path, nrows=sample_rows if file_path.suffix.lower() in {".csv", ".tsv", ".txt", ".xlsx", ".xls", ".jsonl", ".ndjson"} else None)
    result = audit_dataframe(sample, source_path=str(file_path))
    result["size_bytes"] = size_bytes
    result["mode"] = "full_scan" if full_scan and file_path.suffix.lower() in {".csv", ".tsv", ".txt"} else "sample"
    result["rows_scanned"] = int(len(sample))
    result["full_row_count"] = None
    result["full_missing_counts"] = None

    if full_scan and file_path.suffix.lower() in {".csv", ".tsv", ".txt"}:
        total = 0
        missing = None
        for chunk in read_csv_chunks(file_path):
            total += len(chunk)
            current = chunk.isna().sum()
            missing = current if missing is None else missing.add(current, fill_value=0)
        result["full_row_count"] = int(total)
        result["full_missing_counts"] = {str(k): int(v) for k, v in (missing if missing is not None else pd.Series(dtype=int)).items()}
    return result


def audit_directory(root: str | Path, *, sample_rows: int = 5000, full_scan: bool = False) -> dict[str, Any]:
    """Audit every supported tabular file below a directory."""
    base = Path(root)
    files = discover_data_files(base)
    return {
        "root": str(base),
        "directory_exists": base.exists(),
        "files_found": len(files),
        "files": [audit_file(p, sample_rows=sample_rows, full_scan=full_scan) for p in files],
    }


def write_audit_report(audit: dict[str, Any], json_path: str | Path, markdown_path: str | Path) -> None:
    """Write JSON and concise Markdown audit reports."""
    json_file, md_file = Path(json_path), Path(markdown_path)
    json_file.parent.mkdir(parents=True, exist_ok=True)
    json_file.write_text(json.dumps(audit, indent=2, default=str), encoding="utf-8")

    lines = ["# Dataset audit report", "", f"Generated from data root: `{audit.get('data_root', '')}`", ""]
    for name, entry in audit.get("datasets", {}).items():
        lines += [f"## {name}", "", f"- Directory: `{entry['root']}`", f"- Directory exists: **{entry['directory_exists']}**", f"- Tabular files found: **{entry['files_found']}**", ""]
        if not entry["files"]:
            lines += ["No supported data files found yet. Add the downloaded/extracted files and rerun the script.", ""]
            continue
        for file in entry["files"]:
            row_count = file.get("full_row_count")
            row_label = str(row_count) if row_count is not None else f"{file['rows_scanned']} sampled"
            lines += [f"### `{Path(file['file']).name}`", "", f"- Size: {file['size_bytes']:,} bytes", f"- Rows: {row_label}", f"- Columns ({file['columns_count']}): {', '.join(file['columns'])}", f"- Timestamp candidates: {', '.join(file['timestamp_candidates']) or 'none identified'}", f"- Label candidates: {', '.join(file['label_candidates']) or 'none identified'}", f"- ID candidates: {', '.join(file['id_candidates']) or 'none identified'}", f"- Duplicate rows in sample: {file['duplicate_rows_in_sample']}", "", "| Column | Type | Missing in sample | Unique in sample |", "|---|---|---:|---:|"]
            for col in file["columns"]:
                lines.append(f"| `{col['name']}` | `{col['dtype']}` | {col['missing_in_sample']} ({col['missing_pct_in_sample']}%) | {col['unique_in_sample']} |")
            if file.get("full_missing_counts") is not None:
                lines += ["", "Full-scan missing values:", ""]
                lines += [f"- `{k}`: {v}" for k, v in file["full_missing_counts"].items()]
            lines += ["", "Sample rows (first few rows only; inspect the JSON report for values):", "", "```json", json.dumps(file["sample_rows"], indent=2, default=str), "```", ""]
    lines += ["## Interpretation cautions", "", "- The default audit is a sample, not proof that every row is clean.", "- Column-name candidates are heuristics; manually confirm meaning and units.", "- No missing readings are automatically filled or removed by this audit.", "- Do not train models until label semantics, time order, units, and data leakage risks are understood.", ""]
    md_file.write_text("\n".join(lines), encoding="utf-8")
