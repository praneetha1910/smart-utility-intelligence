"""Dataset-specific column helpers based on candidate names, not hardcoded schemas."""
from __future__ import annotations

import re
import pandas as pd

AMPDS_TIMESTAMP_CANDIDATES = ("unix_ts", "timestamp", "ts", "datetime", "date", "time")
THEFT_LABEL_CANDIDATES = ("flag", "label", "theft", "theft_label", "is_theft")
THEFT_ID_CANDIDATES = ("cons_no", "customer_id", "consumer_id", "meter_id", "id")
WATER_LABEL_CANDIDATES = ("label", "event", "class", "status", "target", "fault", "scenario")


def find_candidate_columns(columns: list[str], candidates: tuple[str, ...]) -> list[str]:
    """Return columns matching explicit candidates after case/space normalization."""
    normalize = lambda s: re.sub(r"[^a-z0-9]+", "_", str(s).strip().lower()).strip("_")
    normalized = {normalize(col): str(col) for col in columns}
    matches = []
    for candidate in candidates:
        key = normalize(candidate)
        if key in normalized:
            matches.append(normalized[key])
    return list(dict.fromkeys(matches))


def resolve_one_column(columns: list[str], candidates: tuple[str, ...], *, purpose: str) -> str:
    """Resolve a candidate when exactly one match exists; force a manual choice otherwise."""
    matches = find_candidate_columns(columns, candidates)
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(f"Could not infer {purpose} column from {columns}. Pass the confirmed name explicitly.")
    raise ValueError(f"Ambiguous {purpose} columns {matches}. Choose one explicitly after inspecting the data.")


def split_customer_level_theft_data(
    df: pd.DataFrame, *, label_col: str | None = None, id_cols: list[str] | None = None
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Prepare a customer-row theft dataset into features, labels, and identifiers.

    Expected only when inspection confirms that each row is a customer and the selected label
    represents a theft/normal class. Daily consumption feature selection stays numeric and does
    not silently encode categorical columns.
    """
    from .preprocessing import require_columns

    if label_col is None:
        label_col = resolve_one_column(list(df.columns), THEFT_LABEL_CANDIDATES, purpose="theft label")
    ids = id_cols or find_candidate_columns(list(df.columns), THEFT_ID_CANDIDATES)
    require_columns(df, [label_col, *ids])
    y = df[label_col].copy()
    X = df.drop(columns=[label_col, *ids]).select_dtypes(include="number").copy()
    identifiers = df[ids].copy() if ids else pd.DataFrame(index=df.index)
    if X.empty:
        raise ValueError("No numeric consumption feature columns found; inspect the dataset schema first.")
    return X, y, identifiers
