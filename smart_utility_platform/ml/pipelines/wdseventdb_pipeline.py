"""WDSEventDB cleaning + feature pipeline  ->  leakage / cyber-attack / sensor-failure diagnosis.

WDSEventDB (Raza et al., Lehigh University, Zenodo 10.5281/zenodo.17547955) is a
set of recordings from a sensor-instrumented water-distribution testbed, shipped
as spreadsheets, with leakage, cyber-attack and sensor-failure events.

IMPORTANT - what this pipeline does and does not assume
  The public record does not document the column layout, so nothing here is
  hard-coded to column names. Run `inspect` first; the pipeline detects the time
  column, the label / ground-truth columns and the sensor channels from the files
  and writes what it found to wds_schema.json. Override any detection through
  WDSConfig (time_col, label_cols, drop_cols, keep_cols, label_map).

Cleaning philosophy for an EVENT-DETECTION dataset
  Spikes, flat lines and sudden changes ARE the events (attack, burst, stuck
  sensor). So unlike a forecasting dataset this pipeline does NOT despike or
  smooth. It only repairs what is certainly broken (types, duplicates, NaN/inf,
  impossible values) and it records how it repaired things (`*_was_missing`
  flags, flat-line run lengths computed BEFORE gap filling so forward-fill cannot
  fake a stuck sensor).

Leakage control
  * ground-truth / label / scenario columns and file names never enter the features
  * all rolling features are causal (past samples only) and never cross a
    recording boundary
  * the train/test split is by WHOLE RECORDING (neighbouring samples of one
    recording are near-duplicates, a random row split would be meaningless)
  * the scaler is fitted on training recordings only
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from common import CleaningReport, flatline_run_lengths, safe_class_weights, save_json

TIME_RE = re.compile(r"(time|date|timestamp|datetime)")
# deliberately strict: 'status' / 'flag' are NOT here because pump/valve status are legitimate sensors
LABEL_RE = re.compile(r"(^|_)(labels?|class|classes|event|events|attack|attacks|leak|leakage|fault|failure|"
                      r"anomaly|anomalies|ground_?truth|target|scenario)(_|$)")
INDEX_NAMES = {"index", "unnamed_0", "id", "row", "sr_no", "s_no", "no"}
NORMAL_TOKENS = {"", "0", "0.0", "normal", "none", "nan", "no", "false", "nominal", "baseline", "no_event"}
FILENAME_RULES = [  # used ONLY when no label column exists (weak, whole-recording labels)
    (re.compile(r"attack|cyber|spoof|dos"), "cyberattack"),
    (re.compile(r"leak|burst"), "leakage"),
    (re.compile(r"fail|fault|stuck|drift"), "sensor_failure"),
    (re.compile(r"normal|baseline|nominal|benign"), "normal"),
]


@dataclass
class WDSConfig:
    data_dir: Path
    out_dir: Path
    time_col: str | None = None
    label_cols: list[str] | None = None        # None -> auto-detect
    drop_cols: tuple = ()                      # extra columns to exclude (metadata etc.)
    keep_cols: tuple = ()                      # force-keep columns the label regex would exclude
    label_map: dict | None = None             # raw label value -> class name
    binary: bool = False                       # collapse every event class into "event"
    sheet: str | int | None = None             # None -> sheet with the most numeric cells
    valid_ranges: dict = field(default_factory=dict)   # {"column": (lo, hi)} physical limits
    windows: tuple = (5, 20)
    max_ffill: int = 3
    max_interp: int = 10
    max_missing_frac: float = 0.5              # drop a channel missing more than this anywhere
    test_recordings: tuple | None = None       # names (file stems); None -> automatic
    test_frac: float = 0.25
    seed: int = 42


def norm_name(c) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", str(c).strip().lower()).strip("_") or "col"


def _unique(names: list[str]) -> list[str]:
    seen, out = {}, []
    for n in names:
        seen[n] = seen.get(n, 0) + 1
        out.append(n if seen[n] == 1 else f"{n}_{seen[n]}")
    return out


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------
def list_recordings(data_dir: Path) -> list[Path]:
    files = [p for p in sorted(data_dir.rglob("*"))
             if p.suffix.lower() in {".xlsx", ".xls", ".xlsm", ".csv"} and not p.name.startswith(("~$", "."))]
    if not files:
        raise FileNotFoundError(f"No .xlsx/.xls/.csv recordings found under {data_dir}")
    return files


def read_recording(path: Path, sheet=None) -> tuple[pd.DataFrame, str]:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path), "csv"
    sheets = pd.read_excel(path, sheet_name=sheet if sheet is not None else None)
    if isinstance(sheets, pd.DataFrame):
        return sheets, str(sheet)
    best = max(sheets.items(), key=lambda kv: int(kv[1].apply(pd.to_numeric, errors="coerce").notna().sum().sum()))
    return best[1], str(best[0])


def inspect_dataset(data_dir: Path) -> None:
    """Print what is in every recording so the config can be set deliberately."""
    for p in list_recordings(data_dir):
        df, sheet = read_recording(p)
        df.columns = _unique([norm_name(c) for c in df.columns])
        print(f"\n=== {p.name}  (sheet={sheet})  rows={len(df)}  cols={df.shape[1]}")
        for c in df.columns:
            num = pd.to_numeric(df[c], errors="coerce")
            tag = []
            if TIME_RE.search(c): tag.append("TIME?")
            if LABEL_RE.search(c): tag.append("LABEL?")
            rng = f"[{num.min():.4g} .. {num.max():.4g}]" if num.notna().any() else ""
            print(f"  {c:32s} dtype={str(df[c].dtype):10s} null={df[c].isna().mean():6.1%} "
                  f"unique={df[c].nunique():6d} {rng} {' '.join(tag)}")


# --------------------------------------------------------------------------
# One recording
# --------------------------------------------------------------------------
def _is_row_index(s: pd.Series) -> bool:
    v = pd.to_numeric(s, errors="coerce")
    if v.isna().any() or len(v) < 3:
        return False
    d = v.diff().dropna()
    return bool((d == 1).all() and v.iloc[0] in (0, 1))


def _is_binary(s: pd.Series) -> bool:
    num = pd.to_numeric(s, errors="coerce")
    return bool(num.notna().mean() > 0.95 and set(num.dropna().unique()) <= {0, 1})


def derive_label(df: pd.DataFrame, label_cols: list[str], label_map: dict | None) -> pd.Series:
    def one(col: str) -> pd.Series:
        raw = df[col]
        num = pd.to_numeric(raw, errors="coerce")
        if _is_binary(raw) and not label_map:
            return num.map({0: "normal", 1: col}).where(num.notna())
        if num.notna().mean() > 0.95 and (num.dropna() % 1 == 0).all():
            raw = num.astype("Int64")                       # integer class ids: "2.0" -> "2"
        s = raw.astype(str).str.strip().str.lower()
        if label_map:
            lm = {str(k).strip().lower(): v for k, v in label_map.items()}
            s = s.map(lambda v: lm.get(v, v))
        return s.map(lambda v: "normal" if v in NORMAL_TOKENS else v).where(raw.notna())

    if len(label_cols) == 1:
        return one(label_cols[0])
    parts = pd.concat([one(c) for c in label_cols], axis=1)
    def combine(row):
        act = [v for v in row if isinstance(v, str) and v != "normal"]
        return "+".join(sorted(set(act))) if act else ("normal" if row.notna().any() else np.nan)
    return parts.apply(combine, axis=1)


def prepare_recording(path: Path, cfg: WDSConfig, rep: CleaningReport) -> dict:
    df, sheet = read_recording(path, cfg.sheet)
    df.columns = _unique([norm_name(c) for c in df.columns])
    name = path.stem
    n0 = len(df)
    df = df.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)
    unnamed = [c for c in df.columns if c.startswith("unnamed")]
    if len(unnamed) > 0.5 * df.shape[1]:
        rep.warn(f"{name}: most columns are unnamed - the header row may not be the first row")

    # ---- time column ------------------------------------------------------
    time_col, time_kind = None, None
    cands = [norm_name(cfg.time_col)] if cfg.time_col else [c for c in df.columns if TIME_RE.search(c)]
    for c in cands:
        if c not in df.columns:
            continue
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            time_col, time_kind = c, "datetime"; break
        parsed = pd.to_datetime(df[c], errors="coerce") if df[c].dtype == object else None
        if parsed is not None and parsed.notna().mean() >= 0.9:
            df[c] = parsed; time_col, time_kind = c, "datetime"; break
        num = pd.to_numeric(df[c], errors="coerce")
        if num.notna().mean() >= 0.9:
            df[c] = num; time_col, time_kind = c, "numeric"; break

    dup_t = reorder = 0
    if time_col:
        reorder = int((df[time_col].diff().dropna() < pd.Timedelta(0) if time_kind == "datetime"
                       else df[time_col].diff().dropna() < 0).sum())
        df = df.dropna(subset=[time_col]).sort_values(time_col, kind="stable")
        dup_t = int(df[time_col].duplicated().sum())
        df = df[~df[time_col].duplicated(keep="first")].reset_index(drop=True)

    # ---- label columns ----------------------------------------------------
    keep_norm = {norm_name(c) for c in cfg.keep_cols}
    if cfg.label_cols is not None:
        label_cols = [norm_name(c) for c in cfg.label_cols if norm_name(c) in df.columns]
        candidates = list(label_cols)
    else:
        candidates = [c for c in df.columns if LABEL_RE.search(c) and c != time_col and c not in keep_norm]
        label_cols = list(candidates)
        if len(candidates) > 1:
            text = [c for c in candidates if pd.to_numeric(df[c], errors="coerce").notna().mean() < 0.5]
            binary = [c for c in candidates if _is_binary(df[c])]
            if len(text) == 1:
                label_cols = text   # the other label-like columns are ground-truth extras: excluded, not combined
                rep.warn(f"{name}: several label-like columns {candidates}; using text column {text[0]!r} as the label "
                         "and excluding the rest from the features")
            elif len(binary) == len(candidates):
                pass                # independent 0/1 event flags: combined into one class column
            else:
                raise ValueError(f"{name}: several label-like columns {candidates} and no unambiguous way to combine "
                                 "them. Run `inspect`, then set WDSConfig.label_cols (and drop_cols for the rest).")
    label_source = "column" if label_cols else None
    if label_cols:
        label = derive_label(df, label_cols, cfg.label_map)
        n_na = int(label.isna().sum())
        label = label.ffill().bfill()
        if label.isna().all():
            raise ValueError(f"{name}: label column(s) {label_cols} contain no usable values")
    else:
        label = None
        for rx, cls in FILENAME_RULES:
            if rx.search(name.lower()):
                label, label_source = pd.Series(cls, index=df.index), "filename"
                break
        n_na = 0
        if label is None:
            label, label_source = pd.Series("unlabelled", index=df.index), "none"
            rep.warn(f"{name}: no label column and no recognisable keyword in the file name")
        else:
            rep.warn(f"{name}: no label column found; label '{label.iloc[0]}' taken from the FILE NAME "
                     "(whole-recording label, event timing unknown)")

    # ---- sensor channels --------------------------------------------------
    exclude = set(candidates) | set(label_cols) | {norm_name(c) for c in cfg.drop_cols}
    if time_col:
        exclude.add(time_col)
    channels, dropped_nonnum, dropped_index = {}, [], []
    for c in df.columns:
        if c in exclude:
            continue
        num = pd.to_numeric(df[c], errors="coerce")
        if num.notna().mean() < 0.5:
            dropped_nonnum.append(c); continue
        if c in INDEX_NAMES or _is_row_index(df[c]):
            dropped_index.append(c); continue
        channels[c] = num.replace([np.inf, -np.inf], np.nan)
    ch = pd.DataFrame(channels)

    clipped = 0
    for col, (lo, hi) in cfg.valid_ranges.items():
        k = norm_name(col)
        if k in ch:
            bad = (ch[k] < lo) | (ch[k] > hi)
            clipped += int(bad.sum()); ch.loc[bad, k] = np.nan

    dup_rows = int(ch.duplicated().sum())  # reported only: repeated rows can be legitimate for steady sensors
    rep.add(f"read_{name}", sheet=sheet, rows_raw=n0, rows=len(ch), channels=ch.shape[1],
            time_column=time_col, label_columns=label_cols, label_source=label_source,
            duplicate_timestamps_dropped=dup_t, out_of_order_timestamps_sorted=reorder,
            label_cells_missing_filled=n_na, nonnumeric_columns_dropped=dropped_nonnum,
            row_index_columns_dropped=dropped_index, out_of_range_to_nan=clipped,
            identical_consecutive_values_rows_info_only=dup_rows)
    return {"name": name, "ch": ch, "label": label.reset_index(drop=True),
            "time": df[time_col].reset_index(drop=True) if time_col else None, "time_kind": time_kind,
            "label_cols": label_cols, "label_source": label_source}


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------
def build_recording_features(rec: dict, channels: list[str], cfg: WDSConfig, rep: CleaningReport) -> pd.DataFrame:
    raw = rec["ch"][channels].reset_index(drop=True)
    out = pd.DataFrame(index=raw.index)

    # computed BEFORE filling so forward-fill cannot masquerade as a stuck sensor
    flat = pd.DataFrame({f"{c}_flat_run": flatline_run_lengths(raw[c]) for c in channels})
    was_missing = pd.DataFrame({f"{c}_was_missing": raw[c].isna().astype(int) for c in channels})

    filled = raw.ffill(limit=cfg.max_ffill).interpolate(method="linear", limit=cfg.max_interp, limit_area="inside")
    for c in channels:
        out[c] = filled[c]
    out = pd.concat([out, was_missing, flat], axis=1)

    feat = {}
    for c in channels:
        s = filled[c]
        feat[f"{c}_diff1"] = s.diff()
        for w in cfg.windows:
            r = s.rolling(w, min_periods=1)
            feat[f"{c}_rmean{w}"] = r.mean()
            feat[f"{c}_rstd{w}"] = s.rolling(w, min_periods=2).std()
            feat[f"{c}_rmin{w}"] = r.min()
            feat[f"{c}_rmax{w}"] = r.max()
            feat[f"{c}_dev{w}"] = s - feat[f"{c}_rmean{w}"]
        wl = max(cfg.windows)
        feat[f"{c}_z{wl}"] = (s - feat[f"{c}_rmean{wl}"]) / (feat[f"{c}_rstd{wl}"] + 1e-9)
    out = pd.concat([out, pd.DataFrame(feat)], axis=1)

    if rec["time"] is not None:
        t = rec["time"]
        dt = t.diff().dt.total_seconds() if rec["time_kind"] == "datetime" else t.diff()
        out["dt_s"] = dt
        out["t_rel"] = (t - t.iloc[0]).dt.total_seconds() if rec["time_kind"] == "datetime" else (t - t.iloc[0])
    return out


# --------------------------------------------------------------------------
# Recording-level split
# --------------------------------------------------------------------------
def choose_test_recordings(rec_labels: dict[str, set], cfg: WDSConfig, rep: CleaningReport) -> list[str]:
    names = sorted(rec_labels)
    if cfg.test_recordings:
        missing = [n for n in cfg.test_recordings if n not in names]
        if missing:
            raise ValueError(f"test_recordings not found: {missing}; available: {names}")
        return list(cfg.test_recordings)

    n_test = max(1, round(cfg.test_frac * len(names)))
    if len(names) < 2:
        raise ValueError("Need at least 2 recordings to split by recording.")
    events_exist = any(l != {"normal"} for l in rec_labels.values())
    rng = np.random.default_rng(cfg.seed)
    first = None
    for _ in range(500):
        test = sorted(rng.choice(names, size=n_test, replace=False).tolist())
        train = [n for n in names if n not in test]
        if first is None:
            first = test
        train_cls = set().union(*(rec_labels[n] for n in train))
        test_cls = set().union(*(rec_labels[n] for n in test))
        has_event = any(c != "normal" for c in test_cls)
        if test_cls <= train_cls and (has_event or not events_exist):
            return test
    rep.warn("Could not find a recording split where every test class also occurs in training; "
             "with so few recordings consider leave-one-recording-out (use the `recording` column with GroupKFold).")
    return first


# --------------------------------------------------------------------------
def run(cfg: WDSConfig) -> dict:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    rep = CleaningReport("WDSEventDB")
    recs = [prepare_recording(p, cfg, rep) for p in list_recordings(cfg.data_dir)]
    names = [r["name"] for r in recs]
    if len(set(names)) != len(names):
        raise ValueError("Recording file names must be unique (same stem found twice).")

    # channels: present in every recording, not too empty, not constant everywhere
    common = set.intersection(*(set(r["ch"].columns) for r in recs))
    dropped_not_common = sorted(set.union(*(set(r["ch"].columns) for r in recs)) - common)
    keep = []
    for c in sorted(common):
        worst_missing = max(r["ch"][c].isna().mean() for r in recs)
        allvals = pd.concat([r["ch"][c] for r in recs])
        if worst_missing > cfg.max_missing_frac or allvals.nunique(dropna=True) <= 1:
            continue
        keep.append(c)
    dropped_bad = sorted(common - set(keep))
    if not keep:
        raise ValueError("No usable sensor channels remain; run `inspect` and set drop_cols/label_cols/time_col.")
    rep.add("select_channels", kept=keep, dropped_not_in_every_recording=dropped_not_common,
            dropped_too_missing_or_constant=dropped_bad)

    frames, rec_labels = [], {}
    for r in recs:
        f = build_recording_features(r, keep, cfg, rep)
        f.insert(0, "sample_idx", np.arange(len(f)))
        f.insert(0, "recording", r["name"])
        lab = r["label"]
        if cfg.binary:
            lab = lab.where(lab == "normal", "event")
        f["label"] = lab.to_numpy()
        n_before = len(f)
        # rows that are still NaN in a base channel after the SHORT-gap fill are dropped, not invented
        f = f.dropna(subset=keep).copy()
        # first-sample artefacts only (diff, 2-sample std, dt): no past exists, so "no change" = 0
        derived = [c for c in f.columns if c not in ("recording", "label", "sample_idx") and c not in keep]
        f[derived] = f[derived].fillna(0.0)
        rep.add(f"features_{r['name']}", rows_before=n_before, rows_after=len(f),
                dropped_unfillable_rows=n_before - len(f), classes=sorted(set(f["label"])))
        rec_labels[r["name"]] = set(f["label"])
        frames.append(f)
    allf = pd.concat(frames, ignore_index=True)

    classes = sorted(set(allf["label"]), key=lambda c: (c != "normal", c))
    cmap = {c: i for i, c in enumerate(classes)}
    allf["label_id"] = allf["label"].map(cmap)

    # ---- split by recording ----------------------------------------------
    test_names = choose_test_recordings(rec_labels, cfg, rep)
    is_test = allf["recording"].isin(test_names)
    train, test = allf[~is_test].copy(), allf[is_test].copy()
    rep.add("split_by_recording", train_recordings=sorted(set(train["recording"])),
            test_recordings=test_names, train_rows=len(train), test_rows=len(test),
            train_class_counts=train["label"].value_counts().to_dict(),
            test_class_counts=test["label"].value_counts().to_dict())
    if len(set(allf["label"])) == 1:
        rep.warn("Only one class present overall - check label detection (see wds_schema.json).")

    meta = ["recording", "sample_idx", "label", "label_id"]
    feat_cols = [c for c in allf.columns if c not in meta]
    binary_cols = [c for c in feat_cols if c.endswith("_was_missing")]
    scale_cols = [c for c in feat_cols if c not in binary_cols]

    scaler = RobustScaler().fit(train[scale_cols].to_numpy(dtype=float))   # train recordings only
    def scaled(df):
        o = df.copy()
        o[scale_cols] = scaler.transform(df[scale_cols].to_numpy(dtype=float))
        return o
    scaled(train).to_csv(cfg.out_dir / "wds_train.csv", index=False)
    scaled(test).to_csv(cfg.out_dir / "wds_test.csv", index=False)
    allf.to_csv(cfg.out_dir / "wds_all_unscaled.csv", index=False)
    joblib.dump(scaler, cfg.out_dir / "wds_scaler.joblib")

    save_json({"channels": keep, "feature_columns": feat_cols, "scaled_columns": scale_cols,
               "label_map": cmap,
               "detected": [{"recording": r["name"], "label_columns": r["label_cols"],
                             "label_source": r["label_source"], "time_column_kind": r["time_kind"]} for r in recs],
               "excluded_from_features": "label/ground-truth columns, file names, recording ids, row indices",
               "split_unit": "whole recording"}, cfg.out_dir / "wds_schema.json")
    save_json({"class_weight": safe_class_weights(train["label"]), "note": "from training recordings only"},
              cfg.out_dir / "wds_class_weights.json")
    rep.save(cfg.out_dir / "wds_cleaning_report.json")
    return {"train": train, "test": test, "features": feat_cols, "label_map": cmap}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["inspect", "run"])
    ap.add_argument("--data-dir", required=True, type=Path, help="folder with the unzipped recordings")
    ap.add_argument("--out-dir", default=Path("out/wds"), type=Path)
    ap.add_argument("--time-col")
    ap.add_argument("--label-cols", nargs="*")
    ap.add_argument("--drop-cols", nargs="*", default=[])
    ap.add_argument("--binary", action="store_true", help="collapse all event types to 'event'")
    ap.add_argument("--test-recordings", nargs="*")
    a = ap.parse_args()
    if a.command == "inspect":
        inspect_dataset(a.data_dir)
        return
    run(WDSConfig(a.data_dir, a.out_dir, time_col=a.time_col, label_cols=a.label_cols,
                  drop_cols=tuple(a.drop_cols), binary=a.binary,
                  test_recordings=tuple(a.test_recordings) if a.test_recordings else None))


if __name__ == "__main__":
    main()
