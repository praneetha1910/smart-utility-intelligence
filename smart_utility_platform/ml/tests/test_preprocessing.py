import pandas as pd
import pytest

from ml.data.preprocessing import (
    aggregate_power_to_daily_kwh,
    handle_missing_values,
    normalize_column_names,
    parse_timestamp_column,
    remove_exact_duplicates,
    require_columns,
)
from ml.data.dataset_profiles import find_candidate_columns, split_customer_level_theft_data


def test_normalize_column_names_returns_copy():
    original = pd.DataFrame({" value ": [1, 2]})
    cleaned = normalize_column_names(original)
    assert cleaned.columns.tolist() == ["value"]
    assert original.columns.tolist() == [" value "]


def test_normalize_duplicate_names_raises():
    with pytest.raises(ValueError):
        normalize_column_names(pd.DataFrame([[1, 2]], columns=["x", " x "]))


def test_require_columns_reports_missing():
    with pytest.raises(ValueError, match="missing"):
        require_columns(pd.DataFrame({"a": [1]}), ["missing"])


def test_numeric_timestamp_requires_unit():
    frame = pd.DataFrame({"unix_ts": [1_700_000_000]})
    with pytest.raises(ValueError, match="unit"):
        parse_timestamp_column(frame, column="unix_ts")
    parsed = parse_timestamp_column(frame, column="unix_ts", unit="s")
    assert pd.api.types.is_datetime64_any_dtype(parsed["unix_ts"])


def test_remove_duplicates_returns_count_and_copy():
    frame = pd.DataFrame({"x": [1, 1, 2]})
    cleaned, removed = remove_exact_duplicates(frame)
    assert removed == 1
    assert len(cleaned) == 2
    assert len(frame) == 3


def test_missing_values_preserved_by_default():
    frame = pd.DataFrame({"x": [1.0, None]})
    result, missing = handle_missing_values(frame)
    assert result["x"].isna().sum() == 1
    assert missing == {"x": 1}


def test_daily_kwh_aggregation():
    frame = pd.DataFrame({
        "timestamp": pd.to_datetime(["2024-01-01 00:00:00", "2024-01-01 00:01:00"]),
        "power_w": [1000.0, 1000.0],
    })
    result = aggregate_power_to_daily_kwh(frame, timestamp_col="timestamp", power_col="power_w", sampling_minutes=1)
    assert result.loc[0, "estimated_kwh"] == pytest.approx(2 / 60)


def test_candidate_column_matching():
    assert find_candidate_columns(["CONS_NO", "FLAG", "0", "1"], ("flag", "label")) == ["FLAG"]


def test_split_customer_theft_rows():
    frame = pd.DataFrame({"CONS_NO": [1, 2], "day_1": [10.0, 20.0], "day_2": [12.0, 15.0], "FLAG": [0, 1]})
    X, y, ids = split_customer_level_theft_data(frame, label_col="FLAG")
    assert X.columns.tolist() == ["day_1", "day_2"]
    assert y.tolist() == [0, 1]
    assert ids.columns.tolist() == ["CONS_NO"]


def test_audit_full_scan_counts_rows_and_missing_values(tmp_path):
    from ml.data.audit import audit_file

    path = tmp_path / "sample.csv"
    pd.DataFrame({"reading": [1.0, None, 3.0], "FLAG": [0, 1, 0]}).to_csv(path, index=False)
    audit = audit_file(path, sample_rows=2, full_scan=True)
    assert audit["full_row_count"] == 3
    assert audit["full_missing_counts"]["reading"] == 1
    assert audit["label_candidates"] == ["FLAG"]
