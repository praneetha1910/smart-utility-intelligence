"""Safe preprocessing primitives. All functions return new objects."""
from __future__ import annotations

from typing import Literal
import pandas as pd

MissingStrategy = Literal["preserve", "drop_rows", "numeric_median"]


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Trim surrounding whitespace in column names, returning a copy.

    If trimming would create duplicate names, raise instead of silently merging fields.
    """
    result = df.copy(deep=True)
    normalized = [str(name).strip() for name in result.columns]
    if len(set(normalized)) != len(normalized):
        raise ValueError("Trimming whitespace would create duplicate column names; resolve manually.")
    result.columns = normalized
    return result


def require_columns(df: pd.DataFrame, required: list[str]) -> None:
    """Raise a readable error if one or more required columns are absent."""
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}. Available: {list(df.columns)}")


def parse_timestamp_column(
    df: pd.DataFrame, *, column: str, unit: str | None = None, utc: bool = True, errors: str = "raise"
) -> pd.DataFrame:
    """Parse a known timestamp column on a copy.

    For numeric Unix timestamps, `unit` must be explicitly supplied (for example `s` or `ms`).
    """
    require_columns(df, [column])
    result = df.copy(deep=True)
    values = result[column]
    if pd.api.types.is_numeric_dtype(values):
        if unit is None:
            raise ValueError(f"Column {column!r} is numeric; pass the confirmed Unix timestamp unit explicitly.")
        result[column] = pd.to_datetime(values, unit=unit, utc=utc, errors=errors)
    else:
        result[column] = pd.to_datetime(values, utc=utc, errors=errors)
    return result


def remove_exact_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove exact duplicate rows from a copy and return (cleaned_frame, rows_removed).

    Use only after checking that repeated rows are not meaningful repeated measurements.
    """
    result = df.copy(deep=True)
    before = len(result)
    result = result.drop_duplicates().reset_index(drop=True)
    return result, before - len(result)


def handle_missing_values(
    df: pd.DataFrame,
    *,
    strategy: MissingStrategy = "preserve",
    columns: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply an explicitly selected missing-value strategy to selected columns.

    `preserve` is the default. `numeric_median` fills only selected numeric columns and should
    normally be fitted on training data only. Never apply it blindly to target/label columns.
    """
    result = df.copy(deep=True)
    selected = columns if columns is not None else list(result.columns)
    require_columns(result, selected)
    missing_before = {col: int(result[col].isna().sum()) for col in selected}

    if strategy == "preserve":
        return result, missing_before
    if strategy == "drop_rows":
        result = result.dropna(subset=selected).reset_index(drop=True)
        return result, missing_before
    if strategy == "numeric_median":
        non_numeric = [col for col in selected if not pd.api.types.is_numeric_dtype(result[col])]
        if non_numeric:
            raise ValueError(f"numeric_median only supports numeric columns: {non_numeric}")
        for col in selected:
            median = result[col].median()
            if pd.notna(median):
                result[col] = result[col].fillna(median)
        return result, missing_before
    raise ValueError(f"Unknown missing-value strategy: {strategy}")


def aggregate_power_to_daily_kwh(
    df: pd.DataFrame, *, timestamp_col: str, power_col: str, sampling_minutes: float
) -> pd.DataFrame:
    """Aggregate sampled power readings (watts) into daily estimated energy (kWh).

    This formula assumes the power readings represent average watts over each sampling interval.
    Confirm the source units and sampling interval before using it. It does not handle irregular
    intervals or meter resets; those require dataset-specific logic.
    """
    if sampling_minutes <= 0:
        raise ValueError("sampling_minutes must be greater than zero")
    require_columns(df, [timestamp_col, power_col])
    work = df[[timestamp_col, power_col]].copy(deep=True)
    if not pd.api.types.is_datetime64_any_dtype(work[timestamp_col]):
        raise ValueError(f"{timestamp_col!r} must be parsed to datetime first")
    work[power_col] = pd.to_numeric(work[power_col], errors="raise")
    work["estimated_kwh"] = work[power_col] * (sampling_minutes / 60.0) / 1000.0
    work["date"] = work[timestamp_col].dt.floor("D")
    return work.groupby("date", as_index=False)["estimated_kwh"].sum()
