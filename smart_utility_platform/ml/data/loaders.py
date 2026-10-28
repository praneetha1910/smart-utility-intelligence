"""Conservative readers for common tabular formats."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import pandas as pd

SUPPORTED_SUFFIXES = {".csv", ".tsv", ".txt", ".xlsx", ".xls", ".parquet", ".json", ".jsonl", ".ndjson"}


def discover_data_files(root: str | Path) -> list[Path]:
    """Return supported data files under *root*, ordered by path."""
    base = Path(root)
    if not base.exists():
        return []
    return sorted(
        p for p in base.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES and not p.name.startswith(".")
    )


def load_table(path: str | Path, *, nrows: int | None = None) -> pd.DataFrame:
    """Load one table. Delimiter detection is used for text-delimited files.

    This function never writes to or modifies the source file.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Data file does not exist: {file_path}")
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(file_path, nrows=nrows, low_memory=False)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(file_path, sep=None, engine="python", nrows=nrows, low_memory=False)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(file_path, nrows=nrows)
    if suffix == ".parquet":
        frame = pd.read_parquet(file_path)
        return frame.head(nrows) if nrows is not None else frame
    if suffix == ".jsonl" or suffix == ".ndjson":
        return pd.read_json(file_path, lines=True, nrows=nrows)
    if suffix == ".json":
        frame = pd.read_json(file_path)
        return frame.head(nrows) if nrows is not None else frame
    raise ValueError(f"Unsupported file format: {suffix}")


def read_csv_chunks(path: str | Path, *, chunksize: int = 50_000):
    """Yield CSV chunks for files too large to load into memory at once."""
    file_path = Path(path)
    if file_path.suffix.lower() == ".csv":
        yield from pd.read_csv(file_path, chunksize=chunksize, low_memory=False)
    elif file_path.suffix.lower() in {".tsv", ".txt"}:
        yield from pd.read_csv(file_path, sep=None, engine="python", chunksize=chunksize, low_memory=False)
    else:
        raise ValueError("Chunked reading supports CSV, TSV, and TXT files only.")
