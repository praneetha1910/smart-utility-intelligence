# Machine Learning and Data

This area owns dataset inspection, loading, validation, preprocessing, model training, evaluation, and saved ML artifacts.

Current modules:

- `data/loaders.py`: discover and load supported tabular files.
- `data/audit.py`: summarize schemas, missing values, types, duplicates, and candidate columns.
- `data/preprocessing.py`: reusable, conservative transformations.
- `data/dataset_profiles.py`: helper functions for candidate label, timestamp, and ID columns.
- `scripts/inspect_datasets.py`: generate dataset audit reports.
- `tests/`: unit tests for preprocessing and data helpers.

Planned model packages:

- `forecasting/` for AMPds2 experiments.
- `theft_detection/` for suspicious electricity usage experiments.
- `water_leakage/` for WDSEventDB event detection.

Do not select final features or models until the actual downloaded files have been audited. Keep `data/raw/` unchanged.
