"""Synthetic-data tests. Each fixture reproduces the documented quirks of the real
dataset so the pipelines are exercised on the problems they are meant to solve.

Run:  python tests/test_pipelines.py     (or: pytest -q)
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ampds2_pipeline as amp
import theft_pipeline as thf
import wdseventdb_pipeline as wds


# ======================================================================
# AMPds2
# ======================================================================
def make_ampds(dirpath: Path, days: int = 45, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    start = 1333263600  # 2012-04-01 00:00 America/Vancouver
    n = days * 1440
    ts = start + 60 * np.arange(n)
    hour = (np.arange(n) // 60) % 24
    P = 600 + 400 * np.sin(2 * np.pi * (hour - 6) / 24) + rng.normal(0, 40, n)
    P = np.clip(P, 80, None).round()
    P[5000:5120] = 0                       # dead-meter / swap period
    df = pd.DataFrame({"unix_ts": ts.astype(str), "V": 240.0, "I": P / 240, "f": 60.0, "DPF": 0.9,
                       "APF": 0.9, "P": P, "Pt": P.cumsum(), "Q": P * 0.3, "Qt": (P * 0.3).cumsum(),
                       "S": P * 1.1, "St": (P * 1.1).cumsum()})
    df.loc[20000:20010, "unix_ts"] = ["+" + t for t in df.loc[20000:20010, "unix_ts"]]   # algorithmically filled
    df = df.drop(index=range(30000, 30200))                                           # missing minutes
    df = pd.concat([df, df.iloc[[100, 101]]])                                          # duplicated rows
    df.loc[50, "unix_ts"] = str(int(ts[50]) + 7)                                       # +-10 s jitter
    df.to_csv(dirpath / "Electricity_WHE.csv", index=False)

    wx_idx = pd.date_range("2012-04-01", periods=days * 24, freq="h")   # LST
    temp = np.asarray(8 + 4 * np.sin(2 * np.pi * (np.asarray(wx_idx.hour) - 9) / 24)).astype(object)
    temp[10], temp[11], temp[300] = "M", "E", "**"
    wx = pd.DataFrame({"Date/Time": wx_idx.strftime("%Y-%m-%d %H:%M"), "Temp (C)": temp,
                       "Dew Point Temp (C)": 3.0, "Rel Hum (%)": 80, "Wind Spd (km/h)": 12,
                       "Stn Press (kPa)": 101.0, "Hmdx": "M", "Wind Chill": "M",
                       "Weather": "Cloudy", "Data Quality": "E"})
    wx.to_csv(dirpath / "Climate_HourlyWeather.csv", index=False)
    return {"P": P}


def test_ampds2():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d); (d / "in").mkdir()
        make_ampds(d / "in")
        res = amp.run(amp.AMPdsConfig(d / "in", d / "out"))
        train, val, test = res["train"], res["val"], res["test"]

        # chronological, non-overlapping
        assert train.index.max() < val.index.min() < val.index.max() < test.index.min()
        for part in (train, val, test):
            assert not part.isna().any().any()

        clean = pd.read_csv(d / "out" / "ampds2_hourly_clean_unscaled.csv", index_col=0, parse_dates=True)
        assert (clean["target_W"] > 0).all(), "dead-meter zeros must not survive as a target of 0 W"
        assert "temp_c" in clean and clean["temp_c"].between(-40, 45).all(), "M/E/** markers must become NaN, not numbers"
        assert "humidex" not in clean and "wind_chill" not in clean, "empty weather columns must be dropped"

        # leakage: lag-1 feature at t is the target at t-1 (nothing from t itself)
        full = clean.copy()
        consecutive = full.index.to_series().diff() == pd.Timedelta(hours=1)
        lag = full["WHE_P_mean_lag1"][consecutive]
        prev_target = full["target_W"].shift(1)[consecutive]
        assert np.allclose(lag, prev_target, atol=1e-6)
        # rolling mean at t uses t-1..t-3 only
        i = 400
        expect = full["target_W"].iloc[i - 3:i].mean()
        assert abs(full["roll_mean_3h"].iloc[i] - expect) < 1e-6 or not consecutive.iloc[i - 3:i + 1].all()

        # scaler fitted on train only: train columns centred, val columns are not forced to 0
        sc_cols = [c for c in res["features"] if c in ("temp_c", "roll_mean_24h")]
        scaled_train = pd.read_csv(d / "out" / "ampds2_train.csv", index_col=0)
        scaled_val = pd.read_csv(d / "out" / "ampds2_val.csv", index_col=0)
        for c in sc_cols:
            assert abs(scaled_train[c].mean()) < 1e-6
        assert any(abs(scaled_val[c].mean()) > 1e-6 for c in sc_cols)

        # horizon 6: no lag shorter than 6 h may exist
        res6 = amp.run(amp.AMPdsConfig(d / "in", d / "out6", horizon=6))
        lags = [int(c.split("lag")[-1]) for c in res6["features"] if "_lag" in c and c.split("lag")[-1].isdigit()]
        assert lags and min(lags) >= 6
    print("AMPds2 OK")


# ======================================================================
# SGCC theft
# ======================================================================
def make_sgcc(path: Path, n=420, days=400, seed=1):
    rng = np.random.default_rng(seed)
    base = rng.uniform(3, 15, n)[:, None]
    t = np.arange(days)
    x = base * (1 + 0.3 * np.sin(2 * np.pi * t / 365)) * rng.gamma(20, 1 / 20, (n, days))
    y = (rng.random(n) < 0.10).astype(int)
    for i in np.where(y == 1)[0]:
        k = rng.integers(50, 300)
        x[i, k:] *= rng.uniform(0.1, 0.4)
    x[rng.random(x.shape) < 0.04] = np.nan
    x[3, 5:15] = -4.0                                    # negative readings
    x[10:15] = 0.0                                       # dead meters
    x[20:23, 60:] = np.nan                               # mostly empty
    x[30, 200] = 5000                                    # spike
    dates = pd.date_range("2014-01-01", periods=days)
    cols = [d.strftime("%m/%d/%Y") for d in dates]
    df = pd.DataFrame(x, columns=cols).astype(object)
    df.iloc[40, 7] = "?"                                 # non-numeric
    perm = rng.permutation(days)                         # header NOT chronological
    df = df.iloc[:, perm]
    df.insert(0, "FLAG", y)
    df.insert(0, "CONS_NO", [f"C{i:05d}" for i in range(n)])
    df = pd.concat([df, df.iloc[[50]]])                  # duplicate customer id
    dup = df.iloc[60].copy(); dup["CONS_NO"] = "C_COPY"  # same series, other id
    df = pd.concat([df, dup.to_frame().T])
    df.to_csv(path, index=False)


def test_theft():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d); make_sgcc(d / "data set.csv")
        res = thf.run(thf.TheftConfig(d / "data set.csv", d / "out"))
        z = np.load(d / "out" / "theft_sequences.npz")

        dates = pd.to_datetime(z["dates"])
        assert dates.is_monotonic_increasing and len(dates) == 400, "columns must be re-ordered chronologically"
        for part in ("train", "val", "test"):
            X = z[f"X_{part}"]
            assert not np.isnan(X).any()
            assert X.min() >= 0 and X.max() <= 1 + 1e-6
        assert not np.isnan(res["values"].to_numpy()).any()
        assert (res["values"].to_numpy() >= 0).all(), "negatives must be gone"
        assert res["values"].to_numpy().max() < 4000, "spike must be winsorised"

        ids = res["values"].index
        assert not ids.duplicated().any() and "C_COPY" not in set(ids), "duplicate id / duplicate series removed"
        assert not any(i in set(ids) for i in [f"C{k:05d}" for k in range(10, 15)] if False)
        dead = [f"C{k:05d}" for k in range(10, 15)]
        assert not set(dead) & set(ids), "all-zero customers removed"
        assert not {f"C{k:05d}" for k in range(20, 23)} & set(ids), "mostly-empty customers removed"

        sp = pd.read_csv(d / "out" / "theft_splits.csv")
        assert sp["CONS_NO"].is_unique
        rates = sp.groupby("split")["FLAG"].mean()
        assert rates.max() - rates.min() < 0.06, f"split not stratified: {rates.to_dict()}"

        tr = pd.read_csv(d / "out" / "theft_features_train.csv", index_col=0)
        te = pd.read_csv(d / "out" / "theft_features_test.csv", index_col=0)
        assert not tr.isna().any().any() and not te.isna().any().any()
        assert abs(tr["mean"].median()) < 1e-5, "RobustScaler must be centred on TRAIN median"
        assert abs(te["mean"].median()) > 1e-5 or len(te) < 5

        import json
        cw = json.loads((d / "out" / "theft_class_weights.json").read_text())["class_weight"]
        assert cw["1"] > cw["0"]

        weekly = thf.to_weekly(z["X_train"], dates)
        assert weekly.shape[2] == 7 and weekly.shape[1] == (400 - z["first_monday_index"]) // 7
    print("Theft OK")


# ======================================================================
# WDSEventDB
# ======================================================================
def make_wds(dirpath: Path, with_label_col: bool, seed=2):
    rng = np.random.default_rng(seed)
    specs = {"normal_run1": "Normal", "normal_run2": "Normal", "leak_run1": "Leak", "leak_run2": "Leak",
             "attack_run1": "Cyberattack", "sensor_failure_run1": "Sensor Failure"}
    for name, kind in specs.items():
        n = 300
        t = pd.date_range("2025-03-01 10:00:00", periods=n, freq="s")
        p = 60 + rng.normal(0, 0.3, n)
        q = 12 + rng.normal(0, 0.2, n)
        ev = np.zeros(n, dtype=int)
        if kind == "Leak":
            p[150:] -= np.linspace(0, 8, n - 150); q[150:] += 2; ev[150:] = 1
        if kind == "Cyberattack":
            p[100:140] += 15; ev[100:140] = 1
        if kind == "Sensor Failure":
            p[200:230] = p[199]; ev[200:230] = 1                    # stuck at one value
        df = pd.DataFrame({"Unnamed: 0": np.arange(n), "Timestamp": t, "Pressure Sensor 1 (psi)": p,
                           "Flow Sensor (L/min)": q, "Pump Status": 1})
        if with_label_col:
            df["Event Label"] = np.where(ev == 1, kind, "Normal")
            df["Leak_Flow_Ground_Truth"] = ev * 2.0
        df.loc[40:42, "Flow Sensor (L/min)"] = np.nan                # short dropout
        df = pd.concat([df, df.iloc[[10]]])                          # duplicate timestamp
        df.to_excel(dirpath / f"{name}.xlsx", index=False)


def test_wds_with_labels():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d); (d / "in").mkdir(); make_wds(d / "in", with_label_col=True)
        wds.inspect_dataset(d / "in")
        res = wds.run(wds.WDSConfig(d / "in", d / "out", windows=(5, 20)))
        tr, te = res["train"], res["test"]

        assert set(tr["recording"]).isdisjoint(set(te["recording"])), "split must be by whole recording"
        assert set(res["label_map"]) == {"normal", "leak", "cyberattack", "sensor failure"}, \
            f"label classes polluted by the extra ground-truth column: {set(res['label_map'])}"
        assert set(te["label"]) - {"normal"}, "test set should contain at least one event class"
        assert set(te["label"]) <= set(tr["label"]), "every test class should also be trainable"

        feats = res["features"]
        banned = ("event_label", "leak_flow_ground_truth", "unnamed_0", "recording", "label")
        assert not [c for c in feats if c.startswith(banned)], "labels / index / ground truth leaked into features"
        assert "pump_status" not in res["features"] or True          # constant channel handled (dropped)
        assert "pump_status" not in " ".join(feats), "constant channel should be dropped"

        allf = pd.read_csv(d / "out" / "wds_all_unscaled.csv")
        sf = allf[allf["recording"] == "sensor_failure_run1"]
        assert sf["pressure_sensor_1_psi_flat_run"].max() >= 30, "stuck-sensor run length must be preserved"
        nr = allf[allf["recording"] == "normal_run1"]
        assert nr["pressure_sensor_1_psi_flat_run"].max() < 5
        assert nr["flow_sensor_l_min_was_missing"].sum() == 3, "filled dropouts must stay flagged"
        assert len(nr) == 300, "duplicate timestamp removed, nothing else dropped"
        assert allf["pressure_sensor_1_psi"].max() > 70, "attack spike must NOT be smoothed away"

        scaled = pd.read_csv(d / "out" / "wds_train.csv")
        assert abs(scaled["pressure_sensor_1_psi"].median()) < 1e-6, "scaler fitted on train recordings"
    print("WDSEventDB (label column) OK")


def test_wds_filename_labels_and_causality():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d); (d / "in").mkdir(); make_wds(d / "in", with_label_col=False)
        cfg = wds.WDSConfig(d / "in", d / "out")
        rep = wds.CleaningReport("t", verbose=False)
        res = wds.run(cfg)
        assert {"normal", "leakage", "cyberattack", "sensor_failure"} <= set(res["label_map"])
        warned = [s for s in rep.steps if s["step"] == "WARNING"]
        report = (d / "out" / "wds_cleaning_report.json").read_text()
        assert "FILE NAME" in report, "weak filename labels must be flagged to the user"

        # causality: editing the LAST sample must not change any earlier feature row
        rec = wds.prepare_recording(d / "in" / "leak_run1.xlsx", cfg, rep)
        chans = ["pressure_sensor_1_psi", "flow_sensor_l_min"]
        a = wds.build_recording_features(rec, chans, cfg, rep)
        rec["ch"].loc[rec["ch"].index[-1], "pressure_sensor_1_psi"] += 1000
        b = wds.build_recording_features(rec, chans, cfg, rep)
        pd.testing.assert_frame_equal(a.iloc[:-1].reset_index(drop=True), b.iloc[:-1].reset_index(drop=True))
    print("WDSEventDB (filename labels, causality) OK")


if __name__ == "__main__":
    test_ampds2()
    test_theft()
    test_wds_with_labels()
    test_wds_filename_labels_and_causality()
    print("\nAll pipeline tests passed.")
