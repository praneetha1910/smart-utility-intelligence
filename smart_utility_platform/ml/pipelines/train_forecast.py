"""AMPds2 hourly demand forecasting: Random Forest vs gradient boosting vs simple baselines.

Consumes the output of ampds2_pipeline.py:
    ampds2_train.csv / ampds2_val.csv / ampds2_test.csv   (scaled features, raw-watt target)
    ampds2_hourly_clean_unscaled.csv                      (for the naive baselines, in watts)
    ampds2_features.json

Models
    baselines : same hour yesterday, same hour last week, Ridge regression
    trees     : Random Forest, gradient boosting (LightGBM / XGBoost / sklearn fallback)

Protocol (chronological, no leakage)
    1. fit on TRAIN, choose hyper-parameters (and boosting rounds) on VAL
    2. refit the winner on TRAIN+VAL with the chosen settings   (--no-refit to skip)
    3. score the untouched TEST block once
    4. paired block-bootstrap of the MAE gap to the best baseline

Run:
    python train_forecast.py --pipeline-dir out/ampds2 --out-dir out/ampds2_models
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

import modeling as M

TARGET = "target_W"
RIDGE_ALPHAS = [0.1, 1.0, 10.0, 100.0]


def load_split(pipeline_dir: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(pipeline_dir / f"ampds2_{name}.csv", index_col=0, parse_dates=True)


def seasonal_lag_column(horizon: int, period_hours: int, columns: list[str], base: str = "WHE_P_mean") -> str:
    """Smallest available lag that is a multiple of `period_hours` and >= horizon."""
    lags = sorted(int(c.rsplit("lag", 1)[1]) for c in columns
                  if c.startswith(f"{base}_lag") and c.rsplit("lag", 1)[1].isdigit())
    ok = [l for l in lags if l % period_hours == 0 and l >= horizon]
    if not ok:
        raise ValueError(f"no lag that is a multiple of {period_hours} h and >= horizon {horizon} (have {lags})")
    return f"{base}_lag{ok[0]}"


def run(pipeline_dir: Path, out_dir: Path, backend: str = "auto", n_trees: int = 300, refit: bool = True,
        n_jobs: int = -1, n_boot: int = 300, perm: bool = True, seed: int = 42) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = json.loads((pipeline_dir / "ampds2_features.json").read_text())
    feats, horizon = meta["features"], int(meta["horizon_hours"])
    train, val, test = (load_split(pipeline_dir, s) for s in ("train", "val", "test"))
    raw = pd.read_csv(pipeline_dir / "ampds2_hourly_clean_unscaled.csv", index_col=0, parse_dates=True)
    be = M.resolve_backend(backend)
    print(f"[forecast] horizon={horizon} h  rows train/val/test = {len(train)}/{len(val)}/{len(test)}  gbm={be}")

    Xtr, ytr = train[feats], train[TARGET].to_numpy()
    Xva, yva = val[feats], val[TARGET].to_numpy()
    Xte, yte = test[feats], test[TARGET].to_numpy()
    clip = lambda p: np.clip(np.asarray(p, dtype=float), 0.0, None)   # watts cannot be negative

    results: dict = {"horizon_hours": horizon, "gbm_backend": be, "models": {}}
    preds_val, preds_test, store = {}, {}, {}

    # ---- naive baselines (watts, from the unscaled table) ----------------
    for label, period in (("naive_yesterday", 24), ("naive_last_week", 168)):
        try:
            col = seasonal_lag_column(horizon, period, list(raw.columns))
        except ValueError as e:
            print(f"[forecast] skipping {label}: {e}")
            continue
        preds_val[label] = raw.loc[val.index, col].to_numpy()
        preds_test[label] = raw.loc[test.index, col].to_numpy()
        results["models"][label] = {"source_column": col, "seconds": 0.0}

    # ---- Ridge ------------------------------------------------------------
    with M.Timer() as t:
        best = min(RIDGE_ALPHAS, key=lambda a: M._mae(yva, clip(Ridge(alpha=a).fit(Xtr, ytr).predict(Xva))))
        X_fit, y_fit = (pd.concat([Xtr, Xva]), np.r_[ytr, yva]) if refit else (Xtr, ytr)
        ridge = Ridge(alpha=best).fit(X_fit, y_fit)
    preds_val["ridge"] = clip(Ridge(alpha=best).fit(Xtr, ytr).predict(Xva))
    preds_test["ridge"] = clip(ridge.predict(Xte))
    results["models"]["ridge"] = {"params": {"alpha": best}, "seconds": round(t.seconds, 2)}
    store["ridge"] = ridge

    # ---- Random Forest and gradient boosting ------------------------------
    for family, grid, label in (("rf", M.RF_GRID, "random_forest"), ("gbm", M.GBM_GRID, f"gbm_{be}")):
        with M.Timer() as t:
            best_mae, best_cfg, best_n = np.inf, None, None
            for params in M.expand_grid(grid):
                m = M.TreeModel(family, "regression", params, be, n_trees, seed, n_jobs)
                m.fit(Xtr, ytr, X_val=Xva, y_val=yva)
                mae = M._mae(yva, clip(m.predict(Xva)))
                if mae < best_mae:
                    best_mae, best_cfg, best_n, best_val_pred = mae, params, m.best_n_, clip(m.predict(Xva))
            final = M.TreeModel(family, "regression", best_cfg, be, n_trees, seed, n_jobs)
            if refit:
                final.fit(pd.concat([Xtr, Xva]), np.r_[ytr, yva],
                          n_trees=best_n if family == "gbm" else None)
            else:
                final.fit(Xtr, ytr, X_val=Xva, y_val=yva, n_trees=best_n if family == "gbm" else None)
        preds_val[label] = best_val_pred
        preds_test[label] = clip(final.predict(Xte))
        results["models"][label] = {"params": best_cfg, "n_rounds": best_n, "val_mae": best_mae,
                                    "seconds": round(t.seconds, 2), "refit_on_train_val": refit}
        store[label] = final
        print(f"[forecast] {label}: params={best_cfg} rounds={best_n} val_MAE={best_mae:.1f} W ({t.seconds:.1f}s)")

    # ---- metrics -----------------------------------------------------------
    rows = {}
    for name in preds_test:
        row = {**M.regression_metrics(yva, preds_val[name], "val_"), **M.regression_metrics(yte, preds_test[name], "test_")}
        row["seconds"] = results["models"][name]["seconds"]
        rows[name] = row
    table = pd.DataFrame(rows).T
    baselines = [n for n in ("naive_yesterday", "naive_last_week", "ridge") if n in table.index]
    # the reference baseline is chosen on VALIDATION error, not on test
    ref = min(baselines, key=lambda n: table.loc[n, "val_mae"])
    table["test_skill_vs_" + ref] = 1.0 - table["test_mae"] / table.loc[ref, "test_mae"]

    results["reference_baseline"] = ref
    results["test_mae_gap_vs_reference"] = {}
    for name in (n for n in ("random_forest", f"gbm_{be}") if n in preds_test):
        results["test_mae_gap_vs_reference"][name] = M.paired_bootstrap_diff(
            M._mae, yte, preds_test[name], preds_test[ref], n_boot=n_boot, block=168, seed=seed)
        g = results["test_mae_gap_vs_reference"][name]
        verdict = "better" if g["ci95"][1] < 0 else ("worse" if g["ci95"][0] > 0 else "not clearly different")
        g["verdict"] = verdict
    results["metrics"] = table.round(4).to_dict(orient="index")

    # ---- explanations ------------------------------------------------------
    for name in ("random_forest", f"gbm_{be}"):
        imp = store[name].importance()
        imp.to_csv(out_dir / f"forecast_importance_{name}.csv", header=["importance"])
        results["models"][name]["top_features"] = imp.head(10).round(4).to_dict()
        if perm:
            # explanation only (no decision depends on it). Scored on the test block because the
            # final model may have been refit on train+val, which would make val rows held-in.
            pi = M.permutation_importance(lambda X, m=store[name]: clip(m.predict(X)), Xte, yte,
                                          M._mae, higher_is_better=False, n_repeats=3, seed=seed)
            pi.to_csv(out_dir / f"forecast_permutation_{name}.csv", header=["mae_increase_W"])
        M.save_model(out_dir, name, store[name])
    M.save_model(out_dir, "ridge", store["ridge"])

    pd.DataFrame({"actual_W": yte, **{k: v for k, v in preds_test.items()}}, index=test.index) \
        .to_csv(out_dir / "forecast_test_predictions.csv")
    M.save_results(out_dir, "forecast", results, table)
    M.print_table("Forecast comparison (watts; lower is better)", table,
                  ["val_mae", "test_mae", "test_rmse", "test_wape", "seconds"] + [c for c in table if c.startswith("test_skill")])
    for k, g in results["test_mae_gap_vs_reference"].items():
        print(f"[forecast] {k} vs {ref}: dMAE={g['delta']:+.1f} W  95% CI [{g['ci95'][0]:+.1f}, {g['ci95'][1]:+.1f}]  -> {g['verdict']}")
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pipeline-dir", required=True, type=Path, help="output folder of ampds2_pipeline.py")
    ap.add_argument("--out-dir", default=Path("out/ampds2_models"), type=Path)
    ap.add_argument("--gbm", default="auto", choices=["auto", "lightgbm", "xgboost", "hist"])
    ap.add_argument("--n-trees", type=int, default=300, help="Random Forest size")
    ap.add_argument("--no-refit", action="store_true", help="keep the train-only model for the final test score")
    ap.add_argument("--no-perm", action="store_true", help="skip permutation importance")
    ap.add_argument("--n-jobs", type=int, default=-1)
    a = ap.parse_args()
    run(a.pipeline_dir, a.out_dir, a.gbm, a.n_trees, not a.no_refit, a.n_jobs, perm=not a.no_perm)


if __name__ == "__main__":
    main()
