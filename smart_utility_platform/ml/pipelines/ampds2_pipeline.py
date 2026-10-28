"""AMPds2 cleaning + feature pipeline  ->  hourly electricity demand forecasting.

Input files (from the AMPds2 release, CSV flavour):
    Electricity_WHE.csv            whole-house meter, 1 reading / minute
    Climate_HourlyWeather.csv      Environment Canada YVR hourly weather (LST)
    (optional) Electricity_<XXX>.csv sub-meters, used only through lagged values

Facts about the raw data that this pipeline is built around (Makonin et al.,
Sci. Data 2016):
  * `unix_ts` is seconds since epoch; a leading '+' marks a record that was
    algorithmically filled because the reading was missing.
  * Timestamps can be off by +-10 s -> floored to the minute.
  * The mains meter was swapped on 2012-05-04, producing zero readings.
  * Weather timestamps are Local STANDARD Time (UTC-8, no DST) and contain the
    markers M / E / NA / ** instead of numbers.
  * The family was away 1-7 May 2012, 9-18 Jun 2012, 31 Jul-8 Aug 2013.
  * The house has a dual-fuel HVAC that switches heat-pump -> gas at <= 2 C.

Leakage control: features for hour t only use electricity data up to hour
t-horizon. Weather and calendar for hour t are allowed (a weather forecast and
the calendar are known in advance). The scaler is fitted on the training
block only and the split is chronological.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from dateutil.easter import easter

from common import (
    CleaningReport,
    apply_scaler,
    chronological_split,
    fit_scaler,
    save_json,
)

LOCAL_TZ = "America/Vancouver"
WEATHER_TZ = "Etc/GMT+8"  # fixed UTC-8 == Local Standard Time, no DST

VACATIONS = [  # inclusive local dates, from the dataset paper
    ("2012-05-01", "2012-05-07"),
    ("2012-06-09", "2012-06-18"),
    ("2013-07-31", "2013-08-08"),
]


@dataclass
class AMPdsConfig:
    data_dir: Path
    out_dir: Path
    target_meter: str = "WHE"
    horizon: int = 1                      # forecast horizon in hours
    lags: tuple = (1, 2, 3, 24, 48, 168)  # lags >= horizon are kept
    roll_windows: tuple = (3, 6, 24, 168)
    submeters: tuple = ()                 # e.g. ("HPE", "FGE"); lagged only
    dead_meter_minutes: int = 10          # all-zero run treated as outage
    min_valid_frac_per_hour: float = 0.75
    max_hourly_gap: int = 3               # hours interpolated linearly
    max_weather_gap: int = 6
    p_bounds_w: tuple = (0.0, 30_000.0)   # physical bound for a 200 A service
    train_frac: float = 0.70
    val_frac: float = 0.15
    scaler: str = "standard"


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_power_meter(path: Path, rep: CleaningReport, name: str) -> pd.DataFrame:
    raw = pd.read_csv(path, dtype={"unix_ts": str})
    raw.columns = [c.strip() for c in raw.columns]
    if "unix_ts" not in raw.columns:
        raise ValueError(f"{path.name}: no 'unix_ts' column (found {list(raw.columns)})")

    ts_str = raw["unix_ts"].astype(str).str.strip()
    imputed = ts_str.str.startswith("+")
    ts = pd.to_numeric(ts_str.str.lstrip("+"), errors="coerce")
    bad_ts = int(ts.isna().sum())
    raw = raw.assign(_ts=ts, imputed=imputed.astype(float)).dropna(subset=["_ts"])

    idx = pd.to_datetime(raw["_ts"].astype("int64"), unit="s", utc=True).dt.floor("min")
    raw.index = idx
    raw = raw.drop(columns=["unix_ts", "_ts"])

    n_before = len(raw)
    raw = raw[~raw.index.duplicated(keep="first")].sort_index()
    n_dups = n_before - len(raw)
    for c in raw.columns:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")

    full = pd.date_range(raw.index.min(), raw.index.max(), freq="min", tz="UTC")
    n_gap = len(full) - len(raw)
    raw = raw.reindex(full)
    raw["imputed"] = raw["imputed"].fillna(1.0)  # rows we had to create ourselves
    rep.add(
        f"load_{name}",
        rows=n_before,
        bad_timestamps=bad_ts,
        duplicate_minutes_dropped=n_dups,
        minutes_added_to_close_gaps=n_gap,
        flagged_algorithmically_filled=int(imputed.sum()),
    )
    return raw


def load_weather(path: Path, cfg: AMPdsConfig, rep: CleaningReport) -> pd.DataFrame:
    w = pd.read_csv(path, dtype=str)
    w.columns = [c.strip() for c in w.columns]
    time_col = next((c for c in w.columns if "date" in c.lower() or "time" in c.lower()), None)
    if time_col is None:
        raise ValueError(f"{path.name}: cannot find a date/time column in {list(w.columns)}")

    ts = pd.to_datetime(w[time_col], errors="coerce")
    w = w.loc[ts.notna()].copy()
    w.index = ts[ts.notna()].dt.tz_localize(WEATHER_TZ).dt.tz_convert("UTC")
    w = w[~w.index.duplicated(keep="first")].sort_index()

    keep = {}
    for c in w.columns:
        lc = c.lower()
        if c == time_col or "flag" in lc or "quality" in lc or "weather" == lc:
            continue
        if any(k in lc for k in ("temp", "dew", "hum", "wind", "visib", "press", "hmdx", "chill")):
            keep[c] = pd.to_numeric(w[c], errors="coerce")  # M / E / NA / ** -> NaN
    out = pd.DataFrame(keep)
    # shorten names
    ren = {}
    for c in out.columns:
        lc = c.lower()
        if "dew" in lc: ren[c] = "dew_point_c"
        elif "temp" in lc and "dew" not in lc: ren[c] = "temp_c"
        elif "hum" in lc and "hmdx" not in lc: ren[c] = "rel_hum_pct"
        elif "wind dir" in lc: ren[c] = "wind_dir_10deg"
        elif "wind spd" in lc or "wind speed" in lc: ren[c] = "wind_spd_kmh"
        elif "visib" in lc: ren[c] = "visibility_km"
        elif "press" in lc: ren[c] = "stn_press_kpa"
        elif "hmdx" in lc: ren[c] = "humidex"
        elif "chill" in lc: ren[c] = "wind_chill"
    out = out.rename(columns=ren)

    # columns that are mostly empty (Humidex / Wind Chill) are dropped
    miss = out.isna().mean()
    sparse = miss[miss > 0.5].index.tolist()
    out = out.drop(columns=sparse)

    bounds = {"temp_c": (-40, 45), "dew_point_c": (-50, 35), "rel_hum_pct": (0, 100),
              "wind_spd_kmh": (0, 200), "stn_press_kpa": (80, 110), "visibility_km": (0, 200)}
    clipped = 0
    for c, (lo, hi) in bounds.items():
        if c in out:
            bad = (out[c] < lo) | (out[c] > hi)
            clipped += int(bad.sum())
            out.loc[bad, c] = np.nan

    n_missing = int(out.isna().sum().sum())
    out = out.resample("1h").mean()
    out = out.interpolate(method="time", limit=cfg.max_weather_gap, limit_area="inside")
    out = out.ffill(limit=cfg.max_weather_gap).bfill(limit=cfg.max_weather_gap)
    rep.add(
        "load_weather",
        rows=len(out),
        columns=list(out.columns),
        dropped_sparse_columns=sparse,
        out_of_range_set_to_nan=clipped,
        missing_before_interpolation=n_missing,
        missing_after=int(out.isna().sum().sum()),
    )
    return out


# --------------------------------------------------------------------------
# Cleaning the electricity stream
# --------------------------------------------------------------------------
def clean_minute_power(m: pd.DataFrame, cfg: AMPdsConfig, rep: CleaningReport, name: str) -> pd.DataFrame:
    p = m["P"].astype(float)

    lo, hi = cfg.p_bounds_w
    out_of_range = (p < lo) | (p > hi)
    p = p.mask(out_of_range)

    # Long all-zero runs on the whole-house meter are metering gaps, not "no demand"
    # (fridge, furnace fan etc. never let a house go to 0 W). Applies to the mains only.
    dead = pd.Series(False, index=p.index)
    if name == cfg.target_meter:
        zero = (p == 0).to_numpy()
        run_id = np.cumsum(np.r_[True, zero[1:] != zero[:-1]])
        run_len = pd.Series(zero).groupby(run_id).transform("size").to_numpy()
        dead = pd.Series(zero & (run_len >= cfg.dead_meter_minutes), index=p.index)
        p = p.mask(dead)

    m = m.copy()
    m["P"] = p
    rep.add(
        f"clean_minutes_{name}",
        out_of_range_to_nan=int(out_of_range.sum()),
        dead_meter_minutes_to_nan=int(dead.sum()),
        minutes_still_missing=int(p.isna().sum()),
    )
    return m


def to_hourly(m: pd.DataFrame, cfg: AMPdsConfig, rep: CleaningReport, name: str) -> pd.DataFrame:
    g = m.resample("1h")
    hourly = pd.DataFrame({
        f"{name}_P_mean": g["P"].mean(),
        f"{name}_P_max": g["P"].max(),
        f"{name}_P_std": g["P"].std(),
    })
    if "Q" in m:
        hourly[f"{name}_Q_mean"] = g["Q"].mean()
    valid_frac = g["P"].apply(lambda s: s.notna().mean())
    hourly.loc[valid_frac < cfg.min_valid_frac_per_hour, :] = np.nan
    if name == cfg.target_meter:
        hourly["imputed_frac"] = g["imputed"].mean()

    n_nan = int(hourly[f"{name}_P_mean"].isna().sum())
    hourly = hourly.interpolate(method="time", limit=cfg.max_hourly_gap, limit_area="inside")
    rep.add(
        f"hourly_{name}",
        hours=len(hourly),
        hours_below_valid_fraction=n_nan,
        hours_still_missing_after_short_gap_fill=int(hourly[f"{name}_P_mean"].isna().sum()),
    )
    return hourly


# --------------------------------------------------------------------------
# Calendar
# --------------------------------------------------------------------------
def _nth_weekday(year: int, month: int, weekday: int, n: int) -> pd.Timestamp:
    d = pd.Timestamp(year=year, month=month, day=1)
    d += pd.Timedelta(days=(weekday - d.weekday()) % 7)
    return d + pd.Timedelta(weeks=n - 1)


def bc_holidays(years: Iterable[int]) -> set:
    h = set()
    for y in years:
        h |= {
            pd.Timestamp(y, 1, 1),
            _nth_weekday(y, 2, 0, 3),                       # Family Day
            pd.Timestamp(easter(y)) - pd.Timedelta(days=2),  # Good Friday
            pd.Timestamp(y, 5, 24) - pd.Timedelta(days=pd.Timestamp(y, 5, 24).weekday()),  # Victoria Day: Monday on/before May 24
            pd.Timestamp(y, 7, 1),
            _nth_weekday(y, 8, 0, 1),                       # BC Day
            _nth_weekday(y, 9, 0, 1),                       # Labour Day
            _nth_weekday(y, 10, 0, 2),                      # Thanksgiving
            pd.Timestamp(y, 11, 11),
            pd.Timestamp(y, 12, 25),
        }
    return h


def calendar_features(idx_utc: pd.DatetimeIndex) -> pd.DataFrame:
    loc = idx_utc.tz_convert(LOCAL_TZ)
    f = pd.DataFrame(index=idx_utc)
    hour, dow, month = loc.hour, loc.dayofweek, loc.month
    f["hour_sin"], f["hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    f["dow_sin"], f["dow_cos"] = np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)
    f["month_sin"], f["month_cos"] = np.sin(2 * np.pi * (month - 1) / 12), np.cos(2 * np.pi * (month - 1) / 12)
    f["is_weekend"] = (dow >= 5).astype(int)

    days = loc.tz_localize(None).normalize()
    hol = bc_holidays(sorted(set(days.year)))
    f["is_holiday"] = days.isin(list(hol)).astype(int)

    away = np.zeros(len(f), dtype=int)
    for a, b in VACATIONS:
        away |= ((days >= pd.Timestamp(a)) & (days <= pd.Timestamp(b))).astype(int)
    f["family_away"] = away
    return f


# --------------------------------------------------------------------------
# Feature table
# --------------------------------------------------------------------------
def build_features(hourly: pd.DataFrame, weather: pd.DataFrame, cfg: AMPdsConfig,
                   rep: CleaningReport) -> pd.DataFrame:
    t = cfg.target_meter
    target = f"{t}_P_mean"
    h = cfg.horizon
    df = pd.DataFrame(index=hourly.index)
    df["target_W"] = hourly[target]

    # --- everything below derives from data at or before t - horizon -------
    for col in [c for c in hourly.columns if c.startswith(f"{t}_") and c != "imputed_frac"]:
        for lag in cfg.lags:
            if lag >= h:
                df[f"{col}_lag{lag}"] = hourly[col].shift(lag)
    base = hourly[target].shift(h)
    # a window needs >= 75 % of its hours, so one metering gap does not wipe out the
    # next week of rows (a strict full-window rule would drop ~168 rows per missing hour)
    mp = lambda w: max(2, int(np.ceil(0.75 * w)))
    for w in cfg.roll_windows:
        df[f"roll_mean_{w}h"] = base.rolling(w, min_periods=mp(w)).mean()
        df[f"roll_std_{w}h"] = base.rolling(w, min_periods=mp(w)).std()
    df["roll_max_24h"] = base.rolling(24, min_periods=mp(24)).max()
    df["roll_min_24h"] = base.rolling(24, min_periods=mp(24)).min()
    df["diff_lag_h"] = base - hourly[target].shift(h + 1)
    if "imputed_frac" in hourly:
        df["imputed_frac_lag_h"] = hourly["imputed_frac"].shift(h)
    for sm in cfg.submeters:
        c = f"{sm}_P_mean"
        if c in hourly:
            for lag in (l for l in cfg.lags if l >= h):
                df[f"{c}_lag{lag}"] = hourly[c].shift(lag)

    # --- known-in-advance: weather at t and calendar -----------------------
    wx = weather.reindex(df.index)
    for c in wx.columns:
        df[c] = wx[c]
    if "temp_c" in df:
        df["hdd_18"] = (18.0 - df["temp_c"]).clip(lower=0)
        df["cdd_24"] = (df["temp_c"] - 24.0).clip(lower=0)
        df["below_heatpump_cutoff"] = (df["temp_c"] <= 2.0).astype(int)  # HVAC fuel switch
        df["temp_lag24"] = df["temp_c"].shift(24)
        df["temp_change_24h"] = df["temp_c"] - df["temp_lag24"]
    df = df.join(calendar_features(df.index))
    return df


def run(cfg: AMPdsConfig) -> dict:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    rep = CleaningReport("AMPds2")

    t = cfg.target_meter
    f = cfg.data_dir / f"Electricity_{t}.csv"
    m = clean_minute_power(load_power_meter(f, rep, t), cfg, rep, t)
    hourly = to_hourly(m, cfg, rep, t)
    for sm in cfg.submeters:
        sf = cfg.data_dir / f"Electricity_{sm}.csv"
        sm_min = clean_minute_power(load_power_meter(sf, rep, sm), cfg, rep, sm)
        hourly = hourly.join(to_hourly(sm_min, cfg, rep, sm), how="left")

    weather = load_weather(cfg.data_dir / "Climate_HourlyWeather.csv", cfg, rep)
    feats = build_features(hourly, weather, cfg, rep)

    n0 = len(feats)
    feats = feats.dropna()
    rep.add("drop_rows_with_any_missing_after_short_gap_fill",
            before=n0, after=len(feats), dropped=n0 - len(feats))
    if len(feats) < 100:
        raise ValueError("Fewer than 100 usable hourly rows after cleaning - check the input files.")

    feats.index.name = "timestamp_utc"
    feats.to_csv(cfg.out_dir / "ampds2_hourly_clean_unscaled.csv")

    # ---- chronological split, scaler fitted on train only -----------------
    train, val, test = chronological_split(feats, cfg.train_frac, cfg.val_frac)
    feature_cols = [c for c in feats.columns if c != "target_W"]
    binary_or_cyclic = {c for c in feature_cols
                        if c.endswith(("_sin", "_cos")) or feats[c].nunique() <= 2}
    scale_cols = [c for c in feature_cols if c not in binary_or_cyclic]

    scaler = fit_scaler(train, scale_cols, cfg.scaler)
    for name, part in (("train", train), ("val", val), ("test", test)):
        apply_scaler(part, scale_cols, scaler).to_csv(cfg.out_dir / f"ampds2_{name}.csv")

    joblib.dump(scaler, cfg.out_dir / "ampds2_scaler.joblib")
    save_json({"features": feature_cols, "scaled_columns": scale_cols,
               "unscaled_columns": sorted(binary_or_cyclic), "target": "target_W (raw watts, unscaled)",
               "horizon_hours": cfg.horizon}, cfg.out_dir / "ampds2_features.json")
    rep.add("split", train=len(train), val=len(val), test=len(test),
            train_end=str(train.index[-1]), val_end=str(val.index[-1]),
            note="chronological; scaler fitted on train only")
    rep.save(cfg.out_dir / "ampds2_cleaning_report.json")
    return {"train": train, "val": val, "test": test, "features": feature_cols}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--out-dir", default=Path("out/ampds2"), type=Path)
    ap.add_argument("--target-meter", default="WHE")
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--submeters", nargs="*", default=[])
    a = ap.parse_args()
    run(AMPdsConfig(a.data_dir, a.out_dir, a.target_meter, a.horizon, submeters=tuple(a.submeters)))


if __name__ == "__main__":
    main()
