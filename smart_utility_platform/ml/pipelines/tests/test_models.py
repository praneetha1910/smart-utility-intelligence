"""End-to-end tests: synthetic raw data -> cleaning pipeline -> model training.

Reuses the fixtures from test_pipelines.py, so the model layer is exercised on the
same documented quirks of the real datasets.

Run:  python tests/test_models.py     (or: pytest -q)
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import ampds2_pipeline as amp
import modeling as M
import theft_pipeline as thf
import train_forecast as tf
import train_theft as tt
import train_wds as tw
import wdseventdb_pipeline as wds
from test_pipelines import make_ampds, make_sgcc, make_wds


def backends() -> list[str]:
    return M.available_gbm_backends()


# ======================================================================
def test_forecast():
    for be in backends():
        with tempfile.TemporaryDirectory() as d:
            d = Path(d); (d / "in").mkdir()
            make_ampds(d / "in", days=60)
            amp.run(amp.AMPdsConfig(d / "in", d / "pipe"))
            res = tf.run(d / "pipe", d / "models", backend=be, n_trees=60, n_boot=50, perm=False)

            m = res["metrics"]
            assert {"naive_yesterday", "naive_last_week", "ridge", "random_forest", f"gbm_{be}"} <= set(m)
            for name, row in m.items():
                assert np.isfinite(row["test_mae"]) and row["test_mae"] > 0, name
            # the synthetic signal is a clean daily cycle: every learned model should beat "same hour last week"
            assert m["random_forest"]["test_mae"] < m["naive_last_week"]["test_mae"] * 1.5
            assert (d / "models" / "forecast_comparison.csv").exists()
            assert (d / "models" / "forecast_importance_random_forest.csv").exists()

            # no test information in the baselines: yesterday's value must equal the unscaled lag-24 column
            raw = pd.read_csv(d / "pipe" / "ampds2_hourly_clean_unscaled.csv", index_col=0, parse_dates=True)
            pred = pd.read_csv(d / "models" / "forecast_test_predictions.csv", index_col=0, parse_dates=True)
            assert np.allclose(pred["naive_yesterday"], raw.loc[pred.index, "WHE_P_mean_lag24"])
            assert (pred.drop(columns="actual_W") >= 0).all().all(), "predicted watts must be clipped at 0"
            # the reference baseline is picked on validation error
            ref = res["reference_baseline"]
            assert m[ref]["val_mae"] == min(m[b]["val_mae"] for b in ("naive_yesterday", "naive_last_week", "ridge"))
    print("Forecast OK (" + ", ".join(backends()) + ")")


def test_forecast_horizon_6():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d); (d / "in").mkdir()
        make_ampds(d / "in", days=60)
        amp.run(amp.AMPdsConfig(d / "in", d / "pipe", horizon=6))
        res = tf.run(d / "pipe", d / "models", backend="hist", n_trees=40, n_boot=20, perm=False)
        assert res["horizon_hours"] == 6
        assert res["models"]["naive_yesterday"]["source_column"] == "WHE_P_mean_lag24"
    print("Forecast horizon-6 OK")


# ======================================================================
def test_theft():
    for be in backends():
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            make_sgcc(d / "data set.csv", n=600)
            thf.run(thf.TheftConfig(d / "data set.csv", d / "pipe"))
            res = tt.run(d / "pipe", d / "models", backend=be, n_trees=80, isolation_forest=True,
                         n_boot=60, perm=True)
            m = res["metrics"]
            assert {"logistic", "random_forest", f"gbm_{be}", "isolation_forest"} <= set(m)
            for name in ("logistic", "random_forest", f"gbm_{be}"):
                assert m[name]["test_pr_auc"] > 0.5, f"{name} should find the planted sustained drops"
            assert m["isolation_forest"]["test_pr_auc"] == m["isolation_forest"]["test_pr_auc"]   # finite / not NaN
            gap = res["test_pr_auc_gap_vs_logistic"]["random_forest"]
            assert gap["ci95"][0] <= gap["ci95"][1] and gap["verdict"] in {"better", "worse", "not clearly different"}
            assert (d / "models" / "theft_permutation_random_forest.csv").exists()

            # the threshold comes from validation and is the same object used on test
            sc = pd.read_csv(d / "models" / "theft_test_scores.csv", index_col=0)
            assert len(sc) == int(res["metrics"]["logistic"]["test_tp"] + res["metrics"]["logistic"]["test_fp"]
                                  + res["metrics"]["logistic"]["test_fn"] + res["metrics"]["logistic"]["test_tn"])
    print("Theft OK (" + ", ".join(backends()) + ")")


def test_theft_flags_suspicious_feature():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        make_sgcc(d / "data set.csv", n=600)
        thf.run(thf.TheftConfig(d / "data set.csv", d / "pipe"))
        res = tt.run(d / "pipe", d / "models", backend="hist", n_trees=40, n_boot=20, perm=False,
                     drop_features=("missing_frac_before_imputation",))
        assert res["dropped_features"] == ["missing_frac_before_imputation"]
        imp = pd.read_csv(d / "models" / "theft_importance_random_forest.csv", index_col=0)
        assert "missing_frac_before_imputation" not in imp.index
    print("Theft drop-features OK")


# ======================================================================
def test_wds():
    for with_labels in (True, False):
        be = backends()[0]
        with tempfile.TemporaryDirectory() as d:
            d = Path(d); (d / "in").mkdir(); make_wds(d / "in", with_label_col=with_labels)
            wds.run(wds.WDSConfig(d / "in", d / "pipe"))
            res = tw.run(d / "pipe", d / "models", backend=be, n_trees=40, rounds=40)
            m = res["metrics"]
            assert {"logistic", "random_forest", f"gbm_{be}"} <= set(m)
            for name, row in m.items():
                assert 0.0 <= row["oof_macro_f1"] <= 1.0 and 0.0 <= row["test_macro_f1"] <= 1.0, name
            # test recordings never appear in training (split is by recording)
            tr = pd.read_csv(d / "pipe" / "wds_train.csv", usecols=["recording"])
            te = pd.read_csv(d / "models" / "wds_test_predictions.csv", usecols=["recording"])
            assert set(tr["recording"]).isdisjoint(set(te["recording"]))
            assert (d / "models" / "wds_per_recording.csv").exists()
            by_ch = pd.read_csv(d / "models" / "wds_importance_by_channel_random_forest.csv", index_col=0)
            assert abs(by_ch["importance"].sum() - 1.0) < 1e-6
            if with_labels:
                # one recording per event type: the script must say so instead of trusting OOF thresholds
                assert res["thin_event_types"], "thin event coverage must be detected and reported"
                assert m["random_forest"]["event_threshold"] == 0.5
    print("WDSEventDB OK")


def make_wds_many(dirpath: Path, reps: int = 3, seed: int = 5):
    """Several recordings per event type, so recording-level CV is actually meaningful."""
    for kind in ("Normal", "Leak", "Cyberattack"):
        for r in range(reps):
            rng = np.random.default_rng(seed + 10 * r + len(kind))
            n = 300
            p = 60 + rng.normal(0, 0.3, n); q = 12 + rng.normal(0, 0.2, n); ev = np.zeros(n, dtype=int)
            if kind == "Leak":
                p[150:] -= np.linspace(0, 8, n - 150); q[150:] += 2; ev[150:] = 1
            if kind == "Cyberattack":
                p[100:160] += 15; ev[100:160] = 1
            df = pd.DataFrame({"Timestamp": pd.date_range("2025-03-01 10:00:00", periods=n, freq="s"),
                               "Pressure (psi)": p, "Flow (L/min)": q,
                               "Event Label": np.where(ev == 1, kind, "Normal")})
            df.to_excel(dirpath / f"{kind.lower()}_run{r + 1}.xlsx", index=False)


def test_wds_enough_recordings_learns():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d); (d / "in").mkdir(); make_wds_many(d / "in", reps=4)
        wds.run(wds.WDSConfig(d / "in", d / "pipe", test_frac=0.25))
        res = tw.run(d / "pipe", d / "models", backend="hist", n_trees=60, rounds=60)
        assert not res["thin_event_types"], res["class_recording_coverage_train"]
        m = res["metrics"]
        assert m["random_forest"]["oof_macro_f1"] > 0.6, "out-of-fold (held-out recording) macro-F1 should be solid"
        assert m["random_forest"]["event_threshold"] != 0.5 or m["random_forest"]["event_threshold_source"] == "out_of_fold"
        assert m["random_forest"]["test_event_recall"] > 0.8
        assert m["random_forest"]["test_event_pr_auc"] > 0.8
    print("WDSEventDB learns with enough recordings OK")


def test_wds_stride_and_cv_choice():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d); (d / "in").mkdir(); make_wds(d / "in", with_label_col=True)
        wds.run(wds.WDSConfig(d / "in", d / "pipe"))
        res = tw.run(d / "pipe", d / "models", backend="hist", n_trees=30, rounds=30, train_stride=3, cv_max_folds=2)
        assert res["cv"]["folds"] == 2, "more recordings than cv_max_folds -> GroupKFold with that many folds"
        res2 = tw.run(d / "pipe", d / "models2", backend="hist", n_trees=30, rounds=30, cv_max_folds=10)
        assert res2["cv"]["folds"] == res2["cv"]["n_recordings"], "few recordings -> leave-one-recording-out"
    print("WDSEventDB stride / CV choice OK")


# ======================================================================
def test_modeling_units():
    # threshold maximises F1 on the data it is given
    y = np.array([0, 0, 0, 1, 1, 0, 1, 0, 0, 0])
    s = np.array([.1, .2, .15, .9, .8, .3, .7, .25, .05, .35])
    thr, f1 = M.best_f1_threshold(y, s)
    assert f1 == 1.0 and 0.35 < thr <= 0.7
    bm = M.binary_metrics(y, s, thr)
    assert bm["precision"] == 1.0 and bm["recall"] == 1.0 and bm["r_precision"] == 1.0

    # paired bootstrap: identical scores -> zero difference with a CI containing zero
    g = M.paired_bootstrap_diff(M._pr_auc, y, s, s, n_boot=50)
    assert g["delta"] == 0 and g["ci95"] == [0.0, 0.0]

    # a class missing from training must not break the wrapper or the prediction mapping
    X = pd.DataFrame(np.random.default_rng(0).normal(size=(60, 3)), columns=list("abc"))
    yy = np.array(["normal"] * 40 + ["leak"] * 20)
    for fam in ("rf", "gbm"):
        m = M.TreeModel(fam, "classification", {"max_depth": 4, "min_samples_leaf": 1, "depth": 3, "learning_rate": 0.1},
                        "hist", 20).fit(X, yy)
        assert set(m.predict(X)) <= {"normal", "leak"} and m.positive_score(X).shape == (60,)

    # permutation importance finds the one informative feature
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(300, 4)), columns=list("abcd"))
    yy = 3 * X["a"].to_numpy() + rng.normal(0, 0.1, 300)
    rf = M.TreeModel("rf", "regression", {"max_depth": 6, "min_samples_leaf": 2}, n_estimators=40).fit(X, yy)
    pi = M.permutation_importance(rf.predict, X, yy, M._mae, higher_is_better=False, n_repeats=3)
    assert pi.index[0] == "a" and pi.iloc[0] > 5 * max(pi.iloc[1], 1e-6)
    print("Modeling units OK")


if __name__ == "__main__":
    test_modeling_units()
    test_forecast()
    test_forecast_horizon_6()
    test_theft()
    test_theft_flags_suspicious_feature()
    test_wds()
    test_wds_enough_recordings_learns()
    test_wds_stride_and_cv_choice()
    print("\nAll model tests passed.")
