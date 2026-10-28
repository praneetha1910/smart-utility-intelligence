# Cleaning pipelines for the three project datasets

One pipeline per data source in your architecture diagram. Each one loads the files,
cleans them, builds features and writes ML-ready, leakage-safe train/test files plus a
JSON audit trail of what it changed.

| Module | Dataset | Feeds | Split unit |
|---|---|---|---|
| `ampds2_pipeline.py` | AMPds2 (house electricity, 1-min) | Forecasting | chronological |
| `theft_pipeline.py` | SGCC Electricity Theft Detection | Theft indicators | stratified by customer |
| `wdseventdb_pipeline.py` | WDSEventDB (water testbed events) | Leakage indicators | whole recording |

```
pip install -r requirements.txt
python tests/test_pipelines.py            # cleaning-pipeline tests on synthetic data, ~1 min
python tests/test_models.py               # pipeline -> model end-to-end tests
```

## Run

```
# AMPds2: needs Electricity_WHE.csv and Climate_HourlyWeather.csv in the folder
python ampds2_pipeline.py --data-dir AMPds2/ --out-dir out/ampds2 --horizon 1

# SGCC theft
python theft_pipeline.py --csv "data set.csv" --out-dir out/theft

# WDSEventDB: unzip WDSEventDB.zip first, then LOOK before running
python wdseventdb_pipeline.py inspect --data-dir WDSEventDB/
python wdseventdb_pipeline.py run     --data-dir WDSEventDB/ --out-dir out/wds
```

Every run writes `*_cleaning_report.json` (counts for each step). Read it once per dataset.

## What each pipeline fixes

**AMPds2**: `+`-prefixed timestamps (algorithmically filled rows, kept as `imputed_frac`),
+-10 s timestamp jitter, duplicate/missing minutes, all-zero runs on the mains meter
(the meter swap, treated as outage not as 0 W demand), weather `M/E/NA/**` markers,
weather in Local *Standard* Time (converted properly, no DST error), empty Humidex/Wind-Chill
columns. Features: lags, rolling stats, HDD/CDD, heat-pump cutoff (<= 2 C), BC holidays,
family-away periods, cyclic calendar. Target: hourly mean watts, horizon configurable.

**SGCC theft**: non-chronological date headers, non-numeric and negative readings, empty and
all-zero customers, duplicate IDs, duplicate series (would sit in both train and test),
spikes (per-customer winsorising), gaps (interpolation, then own median). Outputs per-customer
min-max sequences (+ `to_weekly()` for 2-D CNNs) and ~28 engineered features, including
sustained-drop indicators (`min_30d_mean_over_mean`, `last_90d_over_mean`).

**WDSEventDB**: types, duplicate timestamps, ordering, NaN/inf, row-index columns, constant
channels; causal rolling features, differences, z-scores, flat-line run lengths, missing flags.
It deliberately does **not** despike or smooth: spikes, stuck values and sudden changes are the
events you are trying to detect.

## Leakage rules baked in

- Scalers, class weights and any dataset-wide statistic are fitted on **training rows only**.
- AMPds2 features for hour *t* use electricity data up to *t - horizon*; only weather and calendar
  for *t* (known in advance) are used.
- SGCC cleaning is per customer, so it cannot leak between rows; the split precedes scaling.
- WDSEventDB: labels, ground-truth columns, file names and indices never enter the features;
  rolling windows never cross a recording boundary (tested: editing the last sample changes no
  earlier feature); the split is by recording. With few recordings use
  `GroupKFold(groups=df["recording"])`, not a random row split.

## Things to verify on the real files (I could not download them)

1. **WDSEventDB schema.** The Zenodo record does not document columns and the workspace could not
   download the zip, so detection is automatic. Run `inspect`, then check `wds_schema.json`. If
   labels come only from file names the pipeline says so loudly: that is a whole-recording label,
   not event timing. If it stops with "several label-like columns", set `WDSConfig.label_cols`.
2. **AMPds2 meter-swap zeros** are detected by run length (>= 10 min of 0 W), not by hard-coded
   dates. Check `dead_meter_minutes_to_nan` in the report looks plausible.
3. **SGCC all-zero customers** are dropped by default (the report prints how many were theft).
   Use `--keep-all-zero` to keep them. `missing_frac_before_imputation` can be a data-collection
   artefact rather than behaviour: drop it if it tops your feature importance.
4. **Imbalance** (~8.5 % theft) is handled with class weights, not resampling. If you oversample
   (e.g. SMOTE), do it on the training split only, after these pipelines.


# Modelling layer (classical ML only)

Three training scripts consume the pipeline outputs. No deep learning, CPU only.
LightGBM is used if installed, then XGBoost, then scikit-learn's `HistGradientBoosting` as a
dependency-free fallback (`--gbm lightgbm|xgboost|hist`).

| Script | Reads | Baselines | Trees | Validation |
|---|---|---|---|---|
| `train_forecast.py` | `out/ampds2` | same hour yesterday, same hour last week, Ridge | Random Forest, GBM | chronological val; winner refit on train+val |
| `train_theft.py` | `out/theft` | logistic regression | Random Forest, GBM (+ Isolation Forest with `--isolation-forest`) | stratified val; F1 threshold fixed on val |
| `train_wds.py` | `out/wds` | logistic regression | Random Forest, GBM | leave-one-recording-out over training recordings |

```
python train_forecast.py --pipeline-dir out/ampds2 --out-dir out/ampds2_models
python train_theft.py    --pipeline-dir out/theft  --out-dir out/theft_models
python train_wds.py      --pipeline-dir out/wds    --out-dir out/wds_models
```

Each run writes `*_comparison.csv`, `*_results.json` (parameters, metrics, bootstrap verdicts),
feature importances, test predictions and the fitted models (`models/*.joblib`).

## What is reported

- **Forecasting**: MAE, RMSE, WAPE in watts, skill versus the best baseline (picked on validation
  error), and a paired block-bootstrap (168 h blocks) interval on the MAE gap.
- **Theft**: PR-AUC (headline), ROC-AUC, precision / recall / F1 at the validation-chosen threshold,
  R-precision, recall in the top 10 %, and a paired bootstrap interval on the PR-AUC gap to logistic regression.
- **WDSEventDB**: out-of-fold and test macro-F1, per-event-type precision / recall / F1, event-vs-normal
  PR-AUC and F1, a per-recording table (`wds_per_recording.csv`), and importance summed per sensor channel.

The bootstrap verdict (`better` / `worse` / `not clearly different`) is the evidence for or against
the extra complexity of the tree models. If it says "not clearly different" for a task, the baseline wins on cost.

## Leakage and honesty rules (continuing the ones above)

- Hyper-parameters, boosting rounds and thresholds are chosen on validation data only; the test split is scored once.
- WDS class weights are recomputed inside each cross-validation fold; GBM rounds are fixed there so the held-out
  recording cannot also decide when to stop.
- WDS: if an event type occurs in fewer than 2 training recordings, recording-level CV cannot score it. The script
  prints a warning, records `thin_event_types` in the results, and uses 0.5 on P(event) instead of an out-of-fold threshold.
- Theft: the script warns if `missing_frac_before_imputation` tops the importance ranking; re-run with
  `--drop-features missing_frac_before_imputation`.
- Weights come from the pipelines' training-only weights; nothing here refits a scaler.

## Compute (measured)

Theft at the real SGCC size (about 42k customers x 28 features, 29k training rows), one CPU core, 300-tree forest,
4-combination grids: Random Forest ~160 s, LightGBM ~3 s, logistic regression and Isolation Forest under 1 s.
The forest dominates; use `--n-trees 100` to cut it by about two thirds. On `train_wds.py`, `--train-stride k` uses every
k-th training row (rolling features make neighbours near-duplicates) and scales the cost down accordingly.
Not measured: AMPds2 and WDSEventDB at real size, because the real files were not available.
