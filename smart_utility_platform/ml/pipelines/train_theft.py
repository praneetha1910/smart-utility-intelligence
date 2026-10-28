"""SGCC electricity-theft detection: Random Forest vs gradient boosting vs logistic regression.

Consumes the output of theft_pipeline.py:
    theft_features_train.csv / _val.csv / _test.csv   (RobustScaler fitted on train, FLAG column)
    theft_class_weights.json                          (from the training split only)

Models
    baseline : logistic regression (class-weighted)
    trees    : Random Forest, gradient boosting (LightGBM / XGBoost / sklearn fallback)
    optional : Isolation Forest (--isolation-forest), unsupervised, no labels used

Protocol
    * hyper-parameters, boosting rounds and the decision threshold are chosen on VAL
    * the threshold is the F1-optimal one on VAL, then frozen for TEST
    * headline metric is PR-AUC (8.5 % positives makes accuracy and ROC-AUC flattering)
    * paired bootstrap of the PR-AUC gap to the logistic baseline on TEST customers
    * the final model is the train-only model (threshold and weights stay consistent)

Run:
    python train_theft.py --pipeline-dir out/theft --out-dir out/theft_models
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression

import modeling as M

LABEL = "FLAG"
LR_GRID = {"C": [0.1, 1.0, 10.0]}
SUSPECT_FEATURE = "missing_frac_before_imputation"


def load(pipeline_dir: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(pipeline_dir / f"theft_features_{name}.csv", index_col=0)


def run(pipeline_dir: Path, out_dir: Path, backend: str = "auto", n_trees: int = 300, n_jobs: int = -1,
        isolation_forest: bool = False, drop_features: tuple = (), n_boot: int = 300,
        perm: bool = True, seed: int = 42) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    tr, va, te = (load(pipeline_dir, s) for s in ("train", "val", "test"))
    weights = json.loads((pipeline_dir / "theft_class_weights.json").read_text())["class_weight"]
    feats = [c for c in tr.columns if c != LABEL and c not in set(drop_features)]
    be = M.resolve_backend(backend)
    ytr, yva, yte = (d[LABEL].to_numpy().astype(int) for d in (tr, va, te))
    Xtr, Xva, Xte = tr[feats], va[feats], te[feats]
    wtr, wva = M.class_weight_vector(ytr, weights), M.class_weight_vector(yva, weights)
    print(f"[theft] customers train/val/test = {len(tr)}/{len(va)}/{len(te)}  theft rate "
          f"{ytr.mean():.3f}/{yva.mean():.3f}/{yte.mean():.3f}  features={len(feats)}  gbm={be}")
    if drop_features:
        print(f"[theft] dropped features: {list(drop_features)}")

    results: dict = {"gbm_backend": be, "dropped_features": list(drop_features), "models": {}}
    scores_val, scores_test, store = {}, {}, {}

    # ---- logistic regression baseline -------------------------------------
    with M.Timer() as t:
        best_ap, best_c = -1.0, None
        for p in M.expand_grid(LR_GRID):
            lr = LogisticRegression(C=p["C"], max_iter=2000).fit(Xtr, ytr, sample_weight=wtr)
            ap_ = M._pr_auc(yva, lr.predict_proba(Xva)[:, 1])
            if ap_ > best_ap:
                best_ap, best_c, best_lr = ap_, p, lr
    scores_val["logistic"] = best_lr.predict_proba(Xva)[:, 1]
    scores_test["logistic"] = best_lr.predict_proba(Xte)[:, 1]
    results["models"]["logistic"] = {"params": best_c, "val_pr_auc": best_ap, "seconds": round(t.seconds, 2)}
    store["logistic"] = best_lr

    # ---- trees ---------------------------------------------------------------
    for family, grid, label in (("rf", M.RF_GRID, "random_forest"), ("gbm", M.GBM_GRID, f"gbm_{be}")):
        with M.Timer() as t:
            best_ap, best_model, best_cfg = -1.0, None, None
            for params in M.expand_grid(grid):
                m = M.TreeModel(family, "classification", params, be, n_trees, seed, n_jobs)
                m.fit(Xtr, ytr, sample_weight=wtr, X_val=Xva, y_val=yva, w_val=wva)
                ap_ = M._pr_auc(yva, m.predict_proba(Xva)[:, 1])
                if ap_ > best_ap:
                    best_ap, best_model, best_cfg = ap_, m, params
        scores_val[label] = best_model.predict_proba(Xva)[:, 1]
        scores_test[label] = best_model.predict_proba(Xte)[:, 1]
        results["models"][label] = {"params": best_cfg, "n_rounds": best_model.best_n_,
                                    "val_pr_auc": best_ap, "seconds": round(t.seconds, 2)}
        store[label] = best_model
        print(f"[theft] {label}: params={best_cfg} rounds={best_model.best_n_} val_PR-AUC={best_ap:.4f} ({t.seconds:.1f}s)")

    # ---- optional unsupervised detector --------------------------------------
    if isolation_forest:
        with M.Timer() as t:
            iso = IsolationForest(n_estimators=300, contamination="auto", random_state=seed, n_jobs=n_jobs).fit(Xtr)
        scores_val["isolation_forest"] = -iso.score_samples(Xva)
        scores_test["isolation_forest"] = -iso.score_samples(Xte)
        results["models"]["isolation_forest"] = {"note": "unsupervised; labels never used", "seconds": round(t.seconds, 2)}
        store["isolation_forest"] = iso

    # ---- metrics: threshold from VAL, applied once to TEST -------------------
    rows = {}
    for name in scores_test:
        thr, val_f1 = M.best_f1_threshold(yva, scores_val[name])
        row = {"val_f1": val_f1, **{f"test_{k}": v for k, v in M.binary_metrics(yte, scores_test[name], thr).items()}}
        row["seconds"] = results["models"][name]["seconds"]
        rows[name] = row
    table = pd.DataFrame(rows).T
    results["metrics"] = table.round(4).to_dict(orient="index")

    # ---- is the extra complexity worth it? ------------------------------------
    results["test_pr_auc_gap_vs_logistic"] = {}
    for name in (n for n in scores_test if n not in {"logistic", "isolation_forest"}):
        g = M.paired_bootstrap_diff(M._pr_auc, yte, scores_test[name], scores_test["logistic"],
                                    n_boot=n_boot, seed=seed)
        g["verdict"] = "better" if g["ci95"][0] > 0 else ("worse" if g["ci95"][1] < 0 else "not clearly different")
        results["test_pr_auc_gap_vs_logistic"][name] = g

    # ---- explanations ----------------------------------------------------------
    for name in ("random_forest", f"gbm_{be}"):
        imp = store[name].importance()
        imp.to_csv(out_dir / f"theft_importance_{name}.csv", header=["importance"])
        results["models"][name]["top_features"] = imp.head(10).round(4).to_dict()
        if imp.index[0] == SUSPECT_FEATURE:
            print(f"[theft] WARNING: {SUSPECT_FEATURE!r} is the top feature of {name}. It may reflect how the data "
                  f"was collected, not behaviour. Re-run with --drop-features {SUSPECT_FEATURE}.")
            results.setdefault("warnings", []).append(f"{SUSPECT_FEATURE} tops importance for {name}")
        if perm:   # scored on val: the theft models are train-only, so val rows are genuinely held-out
            pi = M.permutation_importance(lambda X, mm=store[name]: mm.predict_proba(X)[:, 1], Xva, yva,
                                          M._pr_auc, higher_is_better=True, n_repeats=5, seed=seed)
            pi.to_csv(out_dir / f"theft_permutation_{name}.csv", header=["pr_auc_drop"])
        M.save_model(out_dir, name, store[name])
    M.save_model(out_dir, "logistic", store["logistic"])

    pd.DataFrame({LABEL: yte, **{f"score_{k}": v for k, v in scores_test.items()}}, index=te.index) \
        .to_csv(out_dir / "theft_test_scores.csv")
    M.save_results(out_dir, "theft", results, table)
    M.print_table("Theft detection (threshold chosen on val, applied to test)", table,
                  ["test_pr_auc", "test_roc_auc", "test_precision", "test_recall", "test_f1",
                   "test_r_precision", "test_recall_top10pct", "seconds"])
    for k, g in results["test_pr_auc_gap_vs_logistic"].items():
        print(f"[theft] {k} vs logistic: dPR-AUC={g['delta']:+.4f}  95% CI [{g['ci95'][0]:+.4f}, {g['ci95'][1]:+.4f}]  -> {g['verdict']}")
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pipeline-dir", required=True, type=Path, help="output folder of theft_pipeline.py")
    ap.add_argument("--out-dir", default=Path("out/theft_models"), type=Path)
    ap.add_argument("--gbm", default="auto", choices=["auto", "lightgbm", "xgboost", "hist"])
    ap.add_argument("--n-trees", type=int, default=300)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--isolation-forest", action="store_true")
    ap.add_argument("--drop-features", nargs="*", default=[], help=f"e.g. {SUSPECT_FEATURE}")
    ap.add_argument("--no-perm", action="store_true")
    a = ap.parse_args()
    run(a.pipeline_dir, a.out_dir, a.gbm, a.n_trees, a.n_jobs, a.isolation_forest,
        tuple(a.drop_features), perm=not a.no_perm)


if __name__ == "__main__":
    main()
