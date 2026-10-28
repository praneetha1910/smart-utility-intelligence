"""WDSEventDB leakage / cyber-attack / sensor-failure diagnosis: RF vs gradient boosting vs logistic regression.

Consumes the output of wdseventdb_pipeline.py:
    wds_train.csv / wds_test.csv   (RobustScaler fitted on training recordings)
    wds_schema.json                (feature columns, channels, label map, how labels were detected)
    wds_class_weights.json

Why this script validates differently from the other two
    Neighbouring samples of one recording are near-duplicates, so a random row split
    (or a random K-fold) would score memorisation. Hyper-parameters are therefore chosen
    by LEAVE-ONE-RECORDING-OUT cross-validation over the training recordings (GroupKFold
    when there are more than --cv-max-folds recordings). Gradient boosting uses a fixed
    number of rounds here: the held-out recording must not also decide when to stop.
    Class weights are recomputed inside each fold from that fold's training labels.

Reported
    * macro-F1 and per-event-type precision / recall / F1 (out-of-fold on train recordings AND on the test recordings)
    * event-vs-normal PR-AUC and F1, with the threshold taken from out-of-fold scores
    * a per-recording table (does each recording get flagged, and how much of it)
    * importance aggregated per sensor channel
    No confidence interval is given: with a handful of recordings an interval over rows would
    be falsely tight. Read the per-recording table instead.

If labels came from FILE NAMES (see wds_schema.json) they are whole-recording labels, so
sample-level scores below mean "which kind of recording is this", not "when did the event start".

Run:
    python train_wds.py --pipeline-dir out/wds --out-dir out/wds_models
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

import modeling as M
from common import safe_class_weights

NORMAL = "normal"
META = ["recording", "sample_idx", "label", "label_id"]
LR_GRID = {"C": [0.1, 1.0]}


class LRModel:
    """Logistic regression with the same tiny API surface the CV loop needs."""

    def __init__(self, params, seed=42):
        self.m = LogisticRegression(C=params["C"], max_iter=500, random_state=seed)

    def fit(self, X, y, sample_weight=None, **_):
        self.m.fit(X, y, sample_weight=sample_weight)
        self.classes_ = self.m.classes_
        return self

    def predict(self, X):
        return self.m.predict(X)

    def positive_score(self, X, negative=NORMAL):
        proba, cl = self.m.predict_proba(X), list(self.m.classes_)
        return 1.0 - proba[:, cl.index(negative)] if negative in cl else proba[:, -1]


def make_model(family: str, params: dict, be: str, n_trees: int, rounds: int, seed: int, n_jobs: int):
    if family == "lr":
        return LRModel(params, seed), None
    if family == "rf":
        return M.TreeModel("rf", "classification", params, be, n_trees, seed, n_jobs), None
    return M.TreeModel("gbm", "classification", params, be, n_trees, seed, n_jobs), rounds


def fit_one(model, n_rounds, X, y):
    w = np.array([safe_class_weights(pd.Series(y))[str(v)] for v in y])
    if n_rounds is None:
        model.fit(X, y, sample_weight=w)
    else:
        model.fit(X, y, sample_weight=w, n_trees=n_rounds)
    return model


def cross_validate(family, params, X, y, groups, folds, be, n_trees, rounds, seed, n_jobs, labels):
    oof_pred = np.empty(len(y), dtype=object)
    oof_evt = np.zeros(len(y))
    for tr_i, va_i in folds:
        m, nr = make_model(family, params, be, n_trees, rounds, seed, n_jobs)
        fit_one(m, nr, X.iloc[tr_i], y[tr_i])
        oof_pred[va_i] = m.predict(X.iloc[va_i])
        oof_evt[va_i] = m.positive_score(X.iloc[va_i], negative=NORMAL)
    rep = M.multiclass_report(y, oof_pred, labels)
    return rep["macro_f1"], oof_pred, oof_evt


def channel_of(feature: str, channels: list[str]) -> str:
    hits = [c for c in channels if feature == c or feature.startswith(c + "_")]
    return max(hits, key=len) if hits else "(time / other)"


def run(pipeline_dir: Path, out_dir: Path, backend: str = "auto", n_trees: int = 200, rounds: int = 200,
        train_stride: int = 1, cv_max_folds: int = 6, n_jobs: int = -1, seed: int = 42) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    schema = json.loads((pipeline_dir / "wds_schema.json").read_text())
    feats, channels = schema["feature_columns"], schema["channels"]
    labels = [c for c, _ in sorted(schema["label_map"].items(), key=lambda kv: kv[1])]
    train = pd.read_csv(pipeline_dir / "wds_train.csv")
    test = pd.read_csv(pipeline_dir / "wds_test.csv")
    be = M.resolve_backend(backend)

    sources = {d["label_source"] for d in schema.get("detected", [])}
    if sources <= {"filename", "none", None}:
        print("[wds] WARNING: every label comes from FILE NAMES (or is missing). These are whole-recording labels: "
              "scores describe recording type, not event timing.")

    tr = train.iloc[::train_stride].reset_index(drop=True) if train_stride > 1 else train
    Xtr, ytr, gtr = tr[feats], tr["label"].astype(str).to_numpy(), tr["recording"].to_numpy()
    Xte, yte = test[feats], test["label"].astype(str).to_numpy()
    n_groups = len(set(gtr))
    if n_groups < 2:
        raise ValueError("Need at least 2 training recordings for recording-level cross-validation.")
    splitter = LeaveOneGroupOut() if n_groups <= cv_max_folds else GroupKFold(n_splits=cv_max_folds)
    folds = list(splitter.split(Xtr, ytr, gtr))
    print(f"[wds] train rows={len(tr)} ({n_groups} recordings)  test rows={len(test)} "
          f"({test['recording'].nunique()} recordings)  features={len(feats)}  gbm={be}  "
          f"cv={'leave-one-recording-out' if n_groups <= cv_max_folds else f'GroupKFold({cv_max_folds})'}")

    # An event type that exists in only one training recording disappears from training whenever that
    # recording is held out, so its out-of-fold score is meaningless and out-of-fold thresholds are unreliable.
    coverage = {c: int(tr.loc[tr["label"].astype(str) == c, "recording"].nunique()) for c in labels}
    thin = [c for c, k in coverage.items() if c != NORMAL and k < 2]
    if thin:
        print(f"[wds] WARNING: event type(s) {thin} occur in fewer than 2 training recordings, so recording-level "
              "cross-validation cannot score them and the out-of-fold event threshold is unreliable "
              "(falling back to 0.5 on P(event)). Treat the test table as the only honest number, and get more recordings "
              "per event type if you can.")

    grids = {"lr": M.expand_grid(LR_GRID), "rf": M.expand_grid(M.RF_GRID), "gbm": M.expand_grid(M.GBM_GRID)}
    names = {"lr": "logistic", "rf": "random_forest", "gbm": f"gbm_{be}"}
    results: dict = {"gbm_backend": be, "cv": {"n_recordings": n_groups, "folds": len(folds)},
                     "class_recording_coverage_train": coverage, "thin_event_types": thin, "models": {}}
    final, test_pred, test_evt, oof_keep = {}, {}, {}, {}

    for fam in ("lr", "rf", "gbm"):
        with M.Timer() as t:
            best = (-1.0, None, None, None)
            for params in grids[fam]:
                f1, oof_p, oof_e = cross_validate(fam, params, Xtr, ytr, gtr, folds, be, n_trees, rounds, seed, n_jobs, labels)
                if f1 > best[0] or best[1] is None:
                    best = (f1, params, oof_p, oof_e)
            cv_f1, params, oof_p, oof_e = best
            model, nr = make_model(fam, params, be, n_trees, rounds, seed, n_jobs)
            fit_one(model, nr, Xtr, ytr)
        nm = names[fam]
        final[nm], oof_keep[nm] = model, (oof_p, oof_e)
        test_pred[nm] = model.predict(Xte)
        test_evt[nm] = model.positive_score(Xte, negative=NORMAL)
        results["models"][nm] = {"params": params, "cv_macro_f1": cv_f1, "seconds": round(t.seconds, 2)}
        print(f"[wds] {nm}: params={params} cv_macro_F1={cv_f1:.4f} ({t.seconds:.1f}s)")

    # ---- test + out-of-fold reporting -------------------------------------------
    y_evt_oof = (ytr != NORMAL).astype(int)
    y_evt_te = (yte != NORMAL).astype(int)
    rows, per_class = {}, {}
    for nm in final:
        oof_p, oof_e = oof_keep[nm]
        rep_oof = M.multiclass_report(ytr, oof_p, labels)
        rep_te = M.multiclass_report(yte, test_pred[nm], labels)
        row = {"oof_macro_f1": rep_oof["macro_f1"], "test_macro_f1": rep_te["macro_f1"],
               "test_accuracy": float((yte == test_pred[nm]).mean())}
        if 0 < y_evt_oof.sum() < len(y_evt_oof):
            if thin:
                thr, row["event_threshold_source"] = 0.5, "default_0.5 (thin event coverage)"
            else:
                thr, _ = M.best_f1_threshold(y_evt_oof, oof_e)    # threshold from out-of-fold scores only
                row["event_threshold_source"] = "out_of_fold"
            row["event_threshold"] = thr
            if 0 < y_evt_te.sum() < len(y_evt_te):
                em = M.binary_metrics(y_evt_te, test_evt[nm], thr)
                row.update({f"test_event_{k}": em[k] for k in ("pr_auc", "roc_auc", "precision", "recall", "f1")})
        row["seconds"] = results["models"][nm]["seconds"]
        rows[nm] = row
        per_class[nm] = {"oof": rep_oof["per_class"], "test": rep_te["per_class"], "test_confusion": rep_te["confusion"]}
    table = pd.DataFrame(rows).T
    results["metrics"] = table.round(4).to_dict(orient="index")
    results["per_event_type"] = per_class

    # ---- per-recording view -----------------------------------------------------------
    rec_rows = []
    for nm in final:
        thr = rows[nm].get("event_threshold", 0.5)
        d = pd.DataFrame({"recording": test["recording"], "true": yte, "pred": test_pred[nm], "evt": test_evt[nm]})
        for r, g in d.groupby("recording"):
            rec_rows.append({"model": nm, "recording": r, "true_label": "+".join(sorted(set(g["true"]))),
                             "rows": len(g), "frac_true_event": float((g["true"] != NORMAL).mean()),
                             "frac_flagged_event": float((g["evt"] >= thr).mean()),
                             "row_accuracy": float((g["true"] == g["pred"]).mean())})
    per_rec = pd.DataFrame(rec_rows)
    per_rec.to_csv(out_dir / "wds_per_recording.csv", index=False)

    # ---- explanations (per channel) -----------------------------------------------------
    for nm in (n for n in final if n != "logistic"):
        imp = final[nm].importance()
        imp.to_csv(out_dir / f"wds_importance_{nm}.csv", header=["importance"])
        by_ch = imp.groupby([channel_of(f, channels) for f in imp.index]).sum().sort_values(ascending=False)
        by_ch.to_csv(out_dir / f"wds_importance_by_channel_{nm}.csv", header=["importance"])
        results["models"][nm]["top_channels"] = by_ch.head(8).round(4).to_dict()
        M.save_model(out_dir, nm, final[nm])
    M.save_model(out_dir, "logistic", final["logistic"])

    pd.DataFrame({"recording": test["recording"], "sample_idx": test["sample_idx"], "true": yte,
                  **{f"pred_{k}": v for k, v in test_pred.items()},
                  **{f"event_score_{k}": v for k, v in test_evt.items()}}).to_csv(
        out_dir / "wds_test_predictions.csv", index=False)
    M.save_results(out_dir, "wds", results, table)
    M.print_table("Event diagnosis (out-of-fold on train recordings, then held-out test recordings)", table,
                  ["oof_macro_f1", "test_macro_f1", "test_accuracy", "test_event_pr_auc", "test_event_recall",
                   "test_event_precision", "test_event_f1", "seconds"])
    print("\nPer-recording (test):\n" + per_rec.round(3).to_string(index=False))
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pipeline-dir", required=True, type=Path, help="output folder of wdseventdb_pipeline.py run")
    ap.add_argument("--out-dir", default=Path("out/wds_models"), type=Path)
    ap.add_argument("--gbm", default="auto", choices=["auto", "lightgbm", "xgboost", "hist"])
    ap.add_argument("--n-trees", type=int, default=200, help="Random Forest size")
    ap.add_argument("--gbm-rounds", type=int, default=200)
    ap.add_argument("--train-stride", type=int, default=1,
                    help="use every k-th training row (rolling features make neighbours near-duplicates); cuts compute")
    ap.add_argument("--cv-max-folds", type=int, default=6)
    ap.add_argument("--n-jobs", type=int, default=-1)
    a = ap.parse_args()
    run(a.pipeline_dir, a.out_dir, a.gbm, a.n_trees, a.gbm_rounds, a.train_stride, a.cv_max_folds, a.n_jobs)


if __name__ == "__main__":
    main()
