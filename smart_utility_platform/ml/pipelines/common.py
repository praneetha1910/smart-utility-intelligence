"""Shared helpers for the three dataset-cleaning pipelines.

Design rules used everywhere in this package:
  1. Nothing learned from data (scaler statistics, class weights, thresholds
     derived from the whole table) is fitted on validation/test rows.
  2. Splits are made on the unit that must not leak: time order for AMPds2,
     customers for SGCC, whole recordings for WDSEventDB.
  3. Every cleaning step reports how many values it touched, so the cleaning
     is auditable rather than silent.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler


class CleaningReport:
    """Collects a human-readable audit trail and writes it as JSON."""

    def __init__(self, name: str, verbose: bool = True):
        self.name = name
        self.verbose = verbose
        self.steps: list[dict] = []

    def add(self, step: str, **details) -> None:
        self.steps.append({"step": step, **details})
        if self.verbose:
            info = ", ".join(f"{k}={v}" for k, v in details.items())
            print(f"[{self.name}] {step}" + (f": {info}" if info else ""))

    def warn(self, message: str) -> None:
        self.steps.append({"step": "WARNING", "message": message})
        print(f"[{self.name}] WARNING: {message}")

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.steps, indent=2, default=str))


def save_json(obj, path: str | Path) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, default=str))


# --------------------------------------------------------------------------
# Time-series helpers
# --------------------------------------------------------------------------
def longest_true_run(mask: np.ndarray) -> int:
    """Length of the longest run of True in a 1-D boolean array."""
    best = run = 0
    for v in mask:
        run = run + 1 if v else 0
        best = max(best, run)
    return best


def flatline_run_lengths(s: pd.Series) -> pd.Series:
    """For each sample, how many consecutive identical values end at it.

    A long run of identical readings from a normally noisy sensor is the
    classic signature of a stuck / failed sensor.
    """
    changed = s.ne(s.shift()) | s.isna()
    grp = changed.cumsum()
    return s.groupby(grp).cumcount().add(1).where(s.notna(), 0)


def hampel_mask(s: pd.Series, window: int = 11, n_sigma: float = 4.0) -> pd.Series:
    """Rolling-median / MAD outlier mask (centred window)."""
    minp = max(3, window // 2)
    med = s.rolling(window, center=True, min_periods=minp).median()
    mad = (s - med).abs().rolling(window, center=True, min_periods=minp).median()
    global_mad = (s - s.median()).abs().median()
    floor = 0.05 * 1.4826 * global_mad if global_mad > 0 else 1e-9
    thresh = np.maximum(n_sigma * 1.4826 * mad, n_sigma * floor)
    return (s - med).abs() > thresh


# --------------------------------------------------------------------------
# Splitting & scaling (leak-free)
# --------------------------------------------------------------------------
def chronological_split(
    df: pd.DataFrame, train: float = 0.70, val: float = 0.15
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by time order (df must already be sorted by its index)."""
    n = len(df)
    i_train = int(n * train)
    i_val = int(n * (train + val))
    return df.iloc[:i_train], df.iloc[i_train:i_val], df.iloc[i_val:]


def fit_scaler(train_df: pd.DataFrame, cols: Iterable[str], kind: str = "standard"):
    """Fit a scaler on TRAIN rows only."""
    scaler = {"standard": StandardScaler, "robust": RobustScaler}[kind]()
    scaler.fit(train_df[list(cols)].to_numpy(dtype=float))
    return scaler


def apply_scaler(df: pd.DataFrame, cols: Iterable[str], scaler) -> pd.DataFrame:
    cols = list(cols)
    out = df.copy()
    out[cols] = scaler.transform(df[cols].to_numpy(dtype=float))
    return out


def safe_class_weights(y: pd.Series) -> dict:
    """'Balanced' class weights computed from the training labels only."""
    counts = y.value_counts()
    n, k = len(y), len(counts)
    return {str(c): float(n / (k * cnt)) for c, cnt in counts.items()}
