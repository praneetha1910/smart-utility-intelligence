"""SGCC Electricity Theft Detection cleaning + feature pipeline.

Input: the SGCC `data set.csv` (Kaggle / Zheng et al. 2018 release):
    CONS_NO, FLAG, <one column per day>   ~42,372 customers x ~1,035 days
    FLAG: 1 = theft, 0 = normal (about 8.5 % theft -> imbalanced).

Known problems this pipeline handles:
  * missing and non-numeric daily readings
  * date headers (MM/DD/YYYY) that are not guaranteed to be in chronological order
  * negative readings (impossible for consumption)
  * customers that are entirely empty / entirely zero
  * duplicated customer IDs or duplicated consumption series (which would
    otherwise sit in both train and test and inflate the score)
  * extreme spikes

Order of operations matters for leakage:
  every cleaning step below is PER CUSTOMER (uses only that customer's own
  row), so doing them before the split cannot leak information between rows.
  The split is stratified by label and happens BEFORE any statistic is shared
  across customers (feature scaler, class weights).

Outputs
  theft_sequences.npz   per-customer min-max scaled daily series (train/val/test)
  theft_features_*.csv  engineered per-customer features, RobustScaler fitted on train
  theft_splits.csv      CONS_NO, FLAG, split
  theft_class_weights.json, theft_feature_scaler.joblib, theft_cleaning_report.json
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from common import CleaningReport, safe_class_weights, save_json


@dataclass
class TheftConfig:
    csv_path: Path
    out_dir: Path
    id_col: str = "CONS_NO"
    label_col: str = "FLAG"
    max_missing_frac: float = 0.50     # drop customers with more than this NaN share
    drop_all_zero_rows: bool = True    # customers with no consumption at all
    max_interp_gap_days: int = 30      # longer gaps are filled with the customer's median
    winsor_k: float = 3.0              # clip at mean + k*std of the customer's own series (None = off)
    val_frac: float = 0.15
    test_frac: float = 0.15
    seed: int = 42


# --------------------------------------------------------------------------
# Loading / structure
# --------------------------------------------------------------------------
def load_raw(cfg: TheftConfig, rep: CleaningReport) -> tuple[pd.DataFrame, pd.Series, pd.DatetimeIndex]:
    df = pd.read_csv(cfg.csv_path, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]

    lower = {c.lower(): c for c in df.columns}
    id_col = lower.get(cfg.id_col.lower())
    lab_col = lower.get(cfg.label_col.lower())
    if id_col is None or lab_col is None:
        raise ValueError(f"Need columns {cfg.id_col!r} and {cfg.label_col!r}; found {list(df.columns)[:6]}...")

    # Separate date columns from everything else, parse their headers
    meta = {id_col, lab_col}
    parsed = {}
    for c in df.columns:
        if c in meta:
            continue
        d = pd.to_datetime(c, format="%m/%d/%Y", errors="coerce")
        if pd.isna(d):
            d = pd.to_datetime(c, errors="coerce")
        parsed[c] = d
    date_cols = [c for c, d in parsed.items() if pd.notna(d)]
    junk_cols = [c for c, d in parsed.items() if pd.isna(d)]
    if not date_cols:
        raise ValueError("No date columns could be parsed from the header.")

    # Sort chronologically; drop repeated dates
    order = sorted(date_cols, key=lambda c: parsed[c])
    was_sorted = order == date_cols
    dates = pd.DatetimeIndex([parsed[c] for c in order])
    dup_dates = int(dates.duplicated().sum())
    keep_mask = ~dates.duplicated()
    order = [c for c, k in zip(order, keep_mask) if k]
    dates = dates[keep_mask]

    ids = df[id_col].astype(str).str.strip()
    y = pd.to_numeric(df[lab_col], errors="coerce")
    values = df[order].apply(pd.to_numeric, errors="coerce").astype("float32")
    values.columns = dates
    values.index = ids.values

    gaps = int((pd.Series(dates).diff().dt.days.dropna() != 1).sum())
    rep.add("load", customers=len(df), day_columns=len(order), date_range=f"{dates[0].date()}..{dates[-1].date()}",
            header_was_chronological=was_sorted, duplicate_date_columns_dropped=dup_dates,
            non_date_columns_ignored=junk_cols, non_consecutive_day_steps=gaps)
    return values, y.set_axis(values.index), dates


# --------------------------------------------------------------------------
# Row-wise cleaning
# --------------------------------------------------------------------------
def clean_rows(values: pd.DataFrame, y: pd.Series, cfg: TheftConfig, rep: CleaningReport):
    # 1. label sanity
    bad_label = ~y.isin([0, 1])
    if bad_label.any():
        rep.add("drop_rows_without_valid_label", dropped=int(bad_label.sum()))
    values, y = values[~bad_label.values], y[~bad_label.values].astype(int)

    # 2. duplicate customer ids
    dup_id = values.index.duplicated(keep="first")
    rep.add("drop_duplicate_customer_ids", dropped=int(dup_id.sum()))
    values, y = values[~dup_id], y[~dup_id]

    # 3. negative readings are impossible
    neg = (values < 0).to_numpy()
    rep.add("negative_to_nan", cells=int(neg.sum()))
    values = values.mask(values < 0)

    # 4. customers with too many missing days / no data at all
    miss_frac = values.isna().mean(axis=1)
    all_nan = miss_frac >= 1.0
    too_missing = miss_frac > cfg.max_missing_frac
    rep.add("drop_mostly_missing_customers", threshold=cfg.max_missing_frac, dropped=int(too_missing.sum()),
            theft_among_dropped=int(y[too_missing].sum()), fully_empty=int(all_nan.sum()))
    values, y = values[~too_missing.to_numpy()], y[~too_missing.to_numpy()]
    miss_frac = miss_frac[~too_missing.to_numpy()]

    if cfg.drop_all_zero_rows:
        dead = ((values.fillna(0) <= 0).all(axis=1)).to_numpy()
        rep.add("drop_all_zero_customers", dropped=int(dead.sum()), theft_among_dropped=int(y[dead].sum()),
                note="no signal to learn from; set drop_all_zero_rows=False to keep them")
        values, y, miss_frac = values[~dead], y[~dead], miss_frac[~dead]

    # 5. duplicated consumption series (same numbers, different customer) -> leakage risk.
    #    Done after empty/all-zero rows are gone so only genuine copies are counted.
    h = pd.util.hash_pandas_object(values.fillna(-1), index=False)
    dup_series = h.duplicated(keep="first").to_numpy()
    # copies with different labels are ambiguous: drop every member of the group
    conflicting = pd.Series(y.to_numpy(), index=h.to_numpy()).groupby(level=0).nunique().gt(1)
    conflict_mask = pd.Series(h.to_numpy()).map(conflicting).fillna(False).to_numpy().astype(bool)
    drop = dup_series | conflict_mask
    rep.add("drop_duplicate_series", dropped=int(drop.sum()), of_which_label_conflicts=int(conflict_mask.sum()))
    values, y, miss_frac = values[~drop], y[~drop], miss_frac[~drop]

    # 6. fill gaps per customer (time interpolation, then own median)
    n_nan = int(values.isna().to_numpy().sum())
    filled = values.T.interpolate(method="linear", limit=cfg.max_interp_gap_days,
                                  limit_direction="both", limit_area=None).T
    med = filled.median(axis=1)
    filled = filled.apply(lambda col: col.fillna(med))
    rep.add("impute_missing_days", cells_missing=n_nan,
            method=f"linear interpolation (gaps <= {cfg.max_interp_gap_days} d) then customer median",
            remaining_nan=int(filled.isna().to_numpy().sum()))
    values = filled.astype("float32")

    # 7. winsorise spikes using the customer's OWN mean/std
    if cfg.winsor_k is not None:
        arr = values.to_numpy()
        mu, sd = arr.mean(axis=1, keepdims=True), arr.std(axis=1, keepdims=True)
        upper = mu + cfg.winsor_k * sd
        clipped = arr > upper
        rep.add("winsorise_upper_tail", k=cfg.winsor_k, cells_clipped=int(clipped.sum()),
                share_of_cells=round(float(clipped.mean()), 5))
        values = pd.DataFrame(np.minimum(arr, upper), index=values.index, columns=values.columns).astype("float32")

    rep.add("class_balance_after_cleaning", customers=len(y), theft=int(y.sum()),
            theft_rate=round(float(y.mean()), 4))
    return values, y, miss_frac


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------
def _row_autocorr(x: np.ndarray, lag: int) -> np.ndarray:
    a, b = x[:, :-lag], x[:, lag:]
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    den = np.sqrt((a ** 2).sum(axis=1) * (b ** 2).sum(axis=1))
    return np.divide((a * b).sum(axis=1), den, out=np.zeros(len(x)), where=den > 0)


def engineer_features(values: pd.DataFrame, miss_frac: pd.Series, include_missing_frac: bool = True) -> pd.DataFrame:
    x = values.to_numpy(dtype=np.float64)
    n, d = x.shape
    dates = values.columns
    eps = 1e-9
    mean = x.mean(axis=1)
    std = x.std(axis=1)
    med = np.median(x, axis=1)

    f = pd.DataFrame(index=values.index)
    f["mean"], f["std"], f["median"] = mean, std, med
    f["min"], f["max"] = x.min(axis=1), x.max(axis=1)
    f["p10"], f["p90"] = np.quantile(x, 0.10, axis=1), np.quantile(x, 0.90, axis=1)
    f["cv"] = std / (mean + eps)
    f["skew"] = np.nan_to_num(stats.skew(x, axis=1))
    f["kurtosis"] = np.nan_to_num(stats.kurtosis(x, axis=1))
    f["max_over_mean"] = f["max"] / (mean + eps)
    f["zero_frac"] = (x <= 0).mean(axis=1)
    f["low_day_frac"] = (x < 0.1 * (med[:, None] + eps)).mean(axis=1)

    run = np.zeros(n)
    best = np.zeros(n)
    for j in range(d):
        run = np.where(x[:, j] <= 0, run + 1, 0)
        best = np.maximum(best, run)
    f["longest_zero_run"] = best

    t = np.arange(d, dtype=np.float64)
    tc = t - t.mean()
    f["trend_slope"] = ((x - mean[:, None]) * tc).sum(axis=1) / (tc ** 2).sum()
    third = d // 3
    f["first_third_mean"], f["last_third_mean"] = x[:, :third].mean(axis=1), x[:, -third:].mean(axis=1)
    f["last_over_first_third"] = f["last_third_mean"] / (f["first_third_mean"] + eps)
    f["last_90d_over_mean"] = x[:, -90:].mean(axis=1) / (mean + eps)

    # sustained drop: lowest 30-day average relative to the customer's own average
    k = 30
    cs = np.cumsum(np.pad(x, ((0, 0), (1, 0))), axis=1)
    roll30 = (cs[:, k:] - cs[:, :-k]) / k
    f["min_30d_mean_over_mean"] = roll30.min(axis=1) / (mean + eps)
    f["max_30d_mean_over_mean"] = roll30.max(axis=1) / (mean + eps)

    dx = np.diff(x, axis=1)
    f["mean_abs_daily_change"] = np.abs(dx).mean(axis=1)
    f["max_daily_drop"] = -dx.min(axis=1)
    f["autocorr_lag1"] = _row_autocorr(x, 1)
    f["autocorr_lag7"] = _row_autocorr(x, 7)

    wk = np.asarray(dates.dayofweek >= 5)
    f["weekend_over_weekday"] = x[:, wk].mean(axis=1) / (x[:, ~wk].mean(axis=1) + eps)
    month = np.asarray(dates.month)
    month_means = np.stack([x[:, month == m].mean(axis=1) for m in range(1, 13) if (month == m).any()], axis=1)
    f["monthly_mean_cv"] = month_means.std(axis=1) / (month_means.mean(axis=1) + eps)

    if include_missing_frac:
        # may carry data-collection artefacts rather than behaviour: drop if you
        # see it dominating feature importance
        f["missing_frac_before_imputation"] = miss_frac.to_numpy()
    return f.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float32")


def minmax_rows(x: np.ndarray) -> np.ndarray:
    """Per-customer min-max scaling (as in Zheng et al.). Uses only the row itself."""
    lo, hi = x.min(axis=1, keepdims=True), x.max(axis=1, keepdims=True)
    rng = hi - lo
    return np.divide(x - lo, rng, out=np.zeros_like(x), where=rng > 0).astype("float32")


def to_weekly(x: np.ndarray, dates: pd.DatetimeIndex) -> np.ndarray:
    """(N, days) -> (N, weeks, 7) starting on the first Monday, for 2-D CNNs."""
    start = int(np.argmax(np.asarray(dates.dayofweek) == 0))
    weeks = (x.shape[1] - start) // 7
    return x[:, start:start + weeks * 7].reshape(len(x), weeks, 7)


# --------------------------------------------------------------------------
def run(cfg: TheftConfig) -> dict:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    rep = CleaningReport("SGCC-theft")

    values, y, dates = load_raw(cfg, rep)
    values, y, miss = clean_rows(values, y, cfg, rep)

    # ---- stratified split by CUSTOMER, before any shared statistic -------
    idx = np.arange(len(y))
    tr_idx, te_idx = train_test_split(idx, test_size=cfg.test_frac, stratify=y.to_numpy(), random_state=cfg.seed)
    rel_val = cfg.val_frac / (1 - cfg.test_frac)
    tr_idx, va_idx = train_test_split(tr_idx, test_size=rel_val, stratify=y.to_numpy()[tr_idx], random_state=cfg.seed)
    split = np.empty(len(y), dtype=object)
    split[tr_idx], split[va_idx], split[te_idx] = "train", "val", "test"
    pd.DataFrame({"CONS_NO": y.index, "FLAG": y.to_numpy(), "split": split}).to_csv(
        cfg.out_dir / "theft_splits.csv", index=False)
    rep.add("split", train=len(tr_idx), val=len(va_idx), test=len(te_idx),
            theft_rate_train=round(float(y.iloc[tr_idx].mean()), 4),
            theft_rate_val=round(float(y.iloc[va_idx].mean()), 4),
            theft_rate_test=round(float(y.iloc[te_idx].mean()), 4))

    # ---- sequences for deep models ---------------------------------------
    seq = minmax_rows(values.to_numpy(dtype=np.float32))
    np.savez_compressed(
        cfg.out_dir / "theft_sequences.npz",
        X_train=seq[tr_idx], y_train=y.to_numpy()[tr_idx],
        X_val=seq[va_idx], y_val=y.to_numpy()[va_idx],
        X_test=seq[te_idx], y_test=y.to_numpy()[te_idx],
        dates=np.asarray(dates.strftime("%Y-%m-%d"), dtype="U10"),  # plain unicode: loads without allow_pickle
        first_monday_index=int(np.argmax(np.asarray(dates.dayofweek) == 0)),
    )

    # ---- engineered features for classical models ------------------------
    feats = engineer_features(values, miss)
    scaler = RobustScaler().fit(feats.iloc[tr_idx])
    scaled = pd.DataFrame(scaler.transform(feats), index=feats.index, columns=feats.columns).astype("float32")
    for name, ix in (("train", tr_idx), ("val", va_idx), ("test", te_idx)):
        out = scaled.iloc[ix].copy()
        out["FLAG"] = y.to_numpy()[ix]
        out.index.name = "CONS_NO"
        out.to_csv(cfg.out_dir / f"theft_features_{name}.csv")
    joblib.dump(scaler, cfg.out_dir / "theft_feature_scaler.joblib")

    cw = safe_class_weights(y.iloc[tr_idx])
    save_json({"class_weight": cw, "pos_weight_for_xgboost_or_bce":
               float((y.iloc[tr_idx] == 0).sum() / max((y.iloc[tr_idx] == 1).sum(), 1)),
               "note": "computed from the training split only"}, cfg.out_dir / "theft_class_weights.json")
    rep.add("outputs_written", directory=str(cfg.out_dir), features=feats.shape[1], class_weights=cw)
    rep.save(cfg.out_dir / "theft_cleaning_report.json")
    return {"values": values, "y": y, "features": feats, "split": split}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, type=Path, help='path to the SGCC "data set.csv"')
    ap.add_argument("--out-dir", default=Path("out/theft"), type=Path)
    ap.add_argument("--keep-all-zero", action="store_true")
    ap.add_argument("--no-winsor", action="store_true")
    a = ap.parse_args()
    run(TheftConfig(a.csv, a.out_dir, drop_all_zero_rows=not a.keep_all_zero,
                    winsor_k=None if a.no_winsor else 3.0))


if __name__ == "__main__":
    main()
