"""Shared modelling layer for the three training scripts.

Sits on top of the cleaning pipelines and only ever consumes their outputs
(train / val / test files, feature lists, class weights). Nothing here refits a
scaler or touches the raw data.

Design rules (same spirit as common.py):
  1. Hyper-parameters, early stopping and decision thresholds are chosen on
     VALIDATION data only. The test split is scored once, at the end.
  2. Class weights come from the pipeline's training-only weights file.
  3. Every comparison includes a simple baseline, and the gap to that baseline
     is reported with a paired bootstrap interval, so "the trees are better"
     is a measured claim and not an impression.
  4. Search grids are deliberately tiny (4 combinations per model): tuning must
     not become the main compute cost.
"""
from __future__ import annotations

import itertools
import json
import time
import warnings
from pathlib import Path
from typing import Callable, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)

# LightGBM >= 4.7 deprecates `eval_set` in favour of eval_X / eval_y, which older releases lack.
# `eval_set` works on every version, so keep it and silence only that one message.
warnings.filterwarnings("ignore", message=r".*eval_set.*deprecated.*")

# --------------------------------------------------------------------------
# Search grids (tiny on purpose)
# --------------------------------------------------------------------------
RF_GRID = {"max_depth": [12, None], "min_samples_leaf": [1, 5]}
GBM_GRID = {"depth": [4, 6], "learning_rate": [0.05, 0.10]}
HIST_ITERS = 300   # the dependency-free fallback has no early stopping


def expand_grid(grid: dict) -> list[dict]:
    keys = list(grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]


def available_gbm_backends() -> list[str]:
    out = []
    for name in ("lightgbm", "xgboost"):
        try:
            __import__(name)
            out.append(name)
        except ImportError:
            pass
    return out + ["hist"]


def resolve_backend(requested: str) -> str:
    avail = available_gbm_backends()
    if requested == "auto":
        return avail[0]
    if requested not in avail:
        raise ValueError(f"GBM backend {requested!r} not available; installed: {avail}")
    return requested


# --------------------------------------------------------------------------
# One wrapper for RF and the three GBM back-ends
# --------------------------------------------------------------------------
class TreeModel:
    """Uniform fit / predict API over RandomForest and LightGBM / XGBoost / HistGB.

    * Labels are re-encoded to 0..K-1 internally (XGBoost needs it, and a
      cross-validation fold may be missing a class); `classes_` maps back.
    * `best_n_` is the number of boosting rounds picked by early stopping
      (validation data), reused when refitting on train+val.
    """

    def __init__(self, family: str, task: str, params: dict, backend: str = "auto",
                 n_estimators: int = 300, seed: int = 42, n_jobs: int = -1):
        assert family in {"rf", "gbm"} and task in {"regression", "classification"}
        self.family, self.task, self.params = family, task, dict(params)
        self.backend = resolve_backend(backend) if family == "gbm" else "sklearn"
        self.n_estimators, self.seed, self.n_jobs = n_estimators, seed, n_jobs
        self.model = None
        self.classes_: np.ndarray | None = None
        self.best_n_: int | None = None
        self.feature_names_: list[str] = []

    # ---- construction ----------------------------------------------------
    def _build(self, n_classes: int | None, n_trees: int | None = None):
        p, seed, nj = self.params, self.seed, self.n_jobs
        clf = self.task == "classification"
        if self.family == "rf":
            cls = RandomForestClassifier if clf else RandomForestRegressor
            mf = "sqrt" if clf else 0.5
            return cls(n_estimators=n_trees or self.n_estimators, max_depth=p.get("max_depth"),
                       min_samples_leaf=p.get("min_samples_leaf", 1), max_features=mf,
                       n_jobs=nj, random_state=seed)
        depth, lr = p["depth"], p["learning_rate"]
        n = n_trees or 2000
        if self.backend == "lightgbm":
            import lightgbm as lgb
            cls = lgb.LGBMClassifier if clf else lgb.LGBMRegressor
            return cls(n_estimators=n, learning_rate=lr, max_depth=depth, num_leaves=2 ** depth - 1,
                       min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                       importance_type="gain", n_jobs=nj, random_state=seed, verbose=-1)
        if self.backend == "xgboost":
            import xgboost as xgb
            cls = xgb.XGBClassifier if clf else xgb.XGBRegressor
            kw = dict(n_estimators=n, learning_rate=lr, max_depth=depth, min_child_weight=1,
                      subsample=0.8, colsample_bytree=0.8, tree_method="hist", n_jobs=nj,
                      random_state=seed, early_stopping_rounds=50 if n_trees is None else None)
            if not clf:
                kw["eval_metric"] = "mae"
            return cls(**kw)
        cls = HistGradientBoostingClassifier if clf else HistGradientBoostingRegressor
        return cls(max_iter=n_trees or HIST_ITERS, learning_rate=lr, max_depth=depth,
                   max_leaf_nodes=2 ** depth, early_stopping=False, random_state=seed)

    # ---- fit -------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y, sample_weight=None, X_val: pd.DataFrame | None = None,
            y_val=None, w_val=None, n_trees: int | None = None) -> "TreeModel":
        self.feature_names_ = list(X.columns)
        y = np.asarray(y)
        if self.task == "classification":
            self.classes_, y_enc = np.unique(y, return_inverse=True)
            yv_enc = None
            if y_val is not None:
                lookup = {c: i for i, c in enumerate(self.classes_)}
                yv = np.asarray(y_val)
                keep = np.array([v in lookup for v in yv])   # a class unseen in training can't be scored
                X_val, yv_enc = X_val[keep], np.array([lookup[v] for v in yv[keep]])
                w_val = None if w_val is None else np.asarray(w_val)[keep]
        else:
            y_enc, yv_enc = y.astype(float), (None if y_val is None else np.asarray(y_val, dtype=float))

        n_cls = len(self.classes_) if self.task == "classification" else None
        self.model = self._build(n_cls, n_trees)
        kw: dict = {}
        if sample_weight is not None:
            kw["sample_weight"] = np.asarray(sample_weight, dtype=float)

        use_es = (self.family == "gbm" and self.backend in {"lightgbm", "xgboost"}
                  and n_trees is None and X_val is not None and len(X_val))
        if self.family == "gbm" and self.backend == "xgboost" and not use_es:
            self.model.set_params(early_stopping_rounds=None)
        if use_es and self.backend == "lightgbm":
            import lightgbm as lgb
            kw["eval_set"] = [(X_val, yv_enc)]
            if w_val is not None:
                kw["eval_sample_weight"] = [np.asarray(w_val, dtype=float)]
            if self.task == "regression":
                kw["eval_metric"] = "l1"
            kw["callbacks"] = [lgb.early_stopping(50, verbose=False)]
        elif use_es and self.backend == "xgboost":
            kw["eval_set"] = [(X_val, yv_enc)]
            if w_val is not None:
                kw["sample_weight_eval_set"] = [np.asarray(w_val, dtype=float)]
            kw["verbose"] = False
        self.model.fit(X, y_enc, **kw)

        if self.family == "rf":
            self.best_n_ = self.model.n_estimators
        elif use_es:
            self.best_n_ = int(getattr(self.model, "best_iteration_", None)
                               or getattr(self.model, "best_iteration", None) or 0) + 1
        else:
            self.best_n_ = n_trees or HIST_ITERS
        return self

    # ---- predict ---------------------------------------------------------
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.task == "regression":
            return self.model.predict(X)
        return self.classes_[np.asarray(self.model.predict(X)).astype(int)]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)

    def positive_score(self, X: pd.DataFrame, negative="normal") -> np.ndarray:
        """P(event) = 1 - P(normal) for multi-class, P(class 1) for binary."""
        proba = self.predict_proba(X)
        classes = list(self.classes_)
        if negative in classes:
            return 1.0 - proba[:, classes.index(negative)]
        return proba[:, -1]

    def importance(self) -> pd.Series:
        imp = np.asarray(self.model.feature_importances_, dtype=float) \
            if hasattr(self.model, "feature_importances_") else np.zeros(len(self.feature_names_))
        s = pd.Series(imp, index=self.feature_names_)
        return (s / s.sum() if s.sum() > 0 else s).sort_values(ascending=False)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def regression_metrics(y, pred, name_prefix: str = "") -> dict:
    y, pred = np.asarray(y, dtype=float), np.asarray(pred, dtype=float)
    err = pred - y
    return {f"{name_prefix}mae": float(np.mean(np.abs(err))),
            f"{name_prefix}rmse": float(np.sqrt(np.mean(err ** 2))),
            f"{name_prefix}wape": float(np.abs(err).sum() / np.abs(y).sum())}


def best_f1_threshold(y, score) -> tuple[float, float]:
    """Threshold maximising F1 on the data it is given (call on VALIDATION only)."""
    p, r, t = precision_recall_curve(y, score)
    f1 = np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-12), 0.0)[:-1]
    if len(f1) == 0:
        return 0.5, 0.0
    i = int(np.argmax(f1))
    return float(t[i]), float(f1[i])


def binary_metrics(y, score, threshold: float) -> dict:
    y, score = np.asarray(y).astype(int), np.asarray(score, dtype=float)
    pred = (score >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    n_pos = int(y.sum())
    order = np.argsort(-score, kind="stable")
    top_pos = int(y[order[:n_pos]].sum()) if n_pos else 0
    top10 = max(1, int(round(0.10 * len(y))))
    return {
        "pr_auc": float(average_precision_score(y, score)) if 0 < n_pos < len(y) else float("nan"),
        "roc_auc": float(roc_auc_score(y, score)) if 0 < n_pos < len(y) else float("nan"),
        "threshold": float(threshold), "precision": prec, "recall": rec,
        "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "r_precision": top_pos / n_pos if n_pos else float("nan"),   # precision in the top-n_pos ranked
        "recall_top10pct": float(y[order[:top10]].sum() / n_pos) if n_pos else float("nan"),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def multiclass_report(y_true, y_pred, labels: list) -> dict:
    """Per-class precision / recall / F1 + macro-F1 + confusion matrix (rows = truth)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    per = {}
    for c in labels:
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        per[str(c)] = {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0,
                       "support": int((y_true == c).sum())}
    present = [str(c) for c in labels if per[str(c)]["support"] > 0]
    macro = float(np.mean([per[c]["f1"] for c in present])) if present else float("nan")
    cm = pd.crosstab(pd.Series(y_true, name="true"), pd.Series(y_pred, name="pred")) \
        .reindex(index=labels, columns=labels, fill_value=0)
    return {"macro_f1": macro, "per_class": per, "confusion": cm.to_dict()}


# --------------------------------------------------------------------------
# Paired bootstrap: is model A really different from baseline B?
# --------------------------------------------------------------------------
def paired_bootstrap_diff(stat: Callable, y, score_a, score_b, n_boot: int = 300,
                          block: int = 1, seed: int = 0) -> dict:
    """stat(model A) - stat(baseline B) with a 95 % percentile interval.

    `block` > 1 resamples contiguous blocks (use ~168 for hourly series so that
    autocorrelation is respected). The same resample is used for A and B, so
    the interval reflects the *difference*, not each score separately.
    """
    y, a, b = np.asarray(y), np.asarray(score_a), np.asarray(score_b)
    n = len(y)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    diffs = []
    for _ in range(n_boot):
        if block > 1:
            starts = rng.integers(0, max(n - block + 1, 1), n_blocks)
            idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
        else:
            idx = rng.integers(0, n, n)
        yy = y[idx]
        if stat.__name__ in {"_pr_auc"} and (yy.sum() == 0 or yy.sum() == len(yy)):
            continue
        diffs.append(stat(yy, a[idx]) - stat(yy, b[idx]))
    d = np.asarray(diffs)
    return {"delta": float(stat(y, a) - stat(y, b)),
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "n_boot": int(len(d))}


def _pr_auc(y, s):
    return float(average_precision_score(y, s))


def _mae(y, p):
    return float(np.mean(np.abs(np.asarray(p, dtype=float) - np.asarray(y, dtype=float))))


# --------------------------------------------------------------------------
# Explanations
# --------------------------------------------------------------------------
def permutation_importance(predict: Callable[[pd.DataFrame], np.ndarray], X: pd.DataFrame, y,
                           metric: Callable, higher_is_better: bool, n_repeats: int = 5,
                           seed: int = 0) -> pd.Series:
    """Drop in score when one feature is shuffled. Pass rows the model has NOT been fitted on."""
    rng = np.random.default_rng(seed)
    base = metric(y, predict(X))
    out = {}
    for c in X.columns:
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[c] = rng.permutation(Xp[c].to_numpy())
            s = metric(y, predict(Xp))
            drops.append((base - s) if higher_is_better else (s - base))
        out[c] = float(np.mean(drops))
    return pd.Series(out).sort_values(ascending=False)


# --------------------------------------------------------------------------
# Small utilities
# --------------------------------------------------------------------------
class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter(); return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.t0


def class_weight_vector(y, weights: dict) -> np.ndarray:
    """Per-row weights from a {class: weight} dict (keys compared as strings)."""
    return np.array([weights[str(v)] for v in np.asarray(y)], dtype=float)


def save_results(out_dir: Path, stem: str, results: dict, table: pd.DataFrame) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{stem}_results.json").write_text(json.dumps(results, indent=2, default=str))
    table.to_csv(out_dir / f"{stem}_comparison.csv")


def save_model(out_dir: Path, name: str, model) -> None:
    (out_dir / "models").mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "models" / f"{name}.joblib")


def print_table(title: str, table: pd.DataFrame, cols: Iterable[str]) -> None:
    cols = [c for c in cols if c in table.columns]
    print(f"\n{title}\n" + table[cols].round(4).to_string())
