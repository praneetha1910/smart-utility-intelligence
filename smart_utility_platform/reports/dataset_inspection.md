# Dataset reconnaissance — Steps 1–3

## Scope and evidence

This is a **source-level inspection** based on the datasets' publisher/repository descriptions. The raw archives could not be downloaded in the current build environment, so actual column-by-column inspection is pending. Run `python scripts/inspect_datasets.py` after placing the files under `data/raw/` to generate the local audit report.

## 1. AMPds2 — electricity forecasting

- **Source:** [Harvard Dataverse AMPds2](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/FIE0S4); [dataset description/download information](https://makonin.com/ampds/).
- **Publisher-reported content:** One residential household monitored over 2012–2014; electricity, water, natural gas, weather and billing information. Meter readings are described as one-minute intervals, with 1,051,200 readings per meter over two years.
- **Fit:** Useful for prototyping short-term electricity-load forecasting and studying household usage patterns.
- **Limitations:** It is one household, not a college campus. The exact file chosen (for example, the combined `Electricity_P.csv` versus individual meter files) affects columns and the target. Confirm whether the selected value is power (W) or energy (Wh/kWh) before calculating daily consumption.
- **Next checks:** Identify timestamp field and units; identify the whole-house or chosen meter signal; check missing timestamps, gaps, duplicates, daylight-saving/timezone handling, and whether readings are power or energy. Use chronological train/validation/test periods.

## 2. Electricity Theft Detection

- **Source:** [Official GitHub repository](https://github.com/henryRDlab/ElectricityTheftDetection); research paper: [Zheng et al. (2018)](https://doi.org/10.1109/TII.2017.2785963).
- **Repository-reported content:** Consumption for 42,372 customers over 1,035 days, from 2014-01-01 to 2016-10-31. The repository asks users to obtain `data.zip`, `data.z01`, and `data.z02` and extract them together.
- **Fit:** Potentially appropriate for a supervised model if the extracted data includes a correctly understood normal/theft label; otherwise, use an anomaly-detection approach and avoid claiming ground truth.
- **Limitations:** Dataset rows are described in research as customers with daily consumption readings, but the exact source file name, header names and label coding must be checked locally. Missing/incorrect readings and class imbalance need handling.
- **Next checks:** Check which column encodes the class and what 0/1 means; count each class; quantify missing values; identify customer IDs and daily reading columns; never impute or scale using the test set.

## 3. WDSEventDB — water-distribution event detection

- **Source:** [Official GitHub repository](https://github.com/Moazeni-s-Concise-Lab/WDSEventDB); [Zenodo dataset record](https://zenodo.org/records/17547955).
- **Publisher-reported content:** Sensor measurements from a water-distribution testbed; archive is organized into `clean`, `cyberattack`, `Leakage`, and `Sensor failure` scenario folders. The published paper description reports 16 measurements each second, including pressure, flow, pump-state and valve-related signals, with normal/anomalous event labels.
- **Fit:** Suitable for event/anomaly detection and potentially event classification, including leakage, if the scenario labels can be mapped reliably.
- **Limitations:** Do not classify all anomalous rows as leaks: cyberattacks and sensor failures are separate event types. The supplied measurements may not map directly to a simple `inflow_litres`/`outflow_litres` schema.
- **Next checks:** List files per scenario; compare column schemas across folders; find timestamp/sample order and event labels; check units, sampling consistency and class balance; use time-aware splits to avoid near-duplicate temporal samples leaking across train/test.

## Combined decision

Keep the three datasets separate and build independent data adapters. Their row meanings and sampling frequencies differ. The dashboard can combine model results later, but raw observations should not be concatenated into one table just to make a unified dataset.

## Step 2 — Scaffold created

The project layout separates raw data, preprocessing, tests, reports, future model code, the future Streamlit dashboard and the optional API. The generated audit script is the bridge from source-level knowledge to a verified, local schema report.

## Step 3 — Preprocessing implemented

Reusable helpers are conservative by default:

- load supported tabular files;
- audit schema, candidate timestamps/labels/IDs, missing values and sampled duplicates;
- trim header whitespace without mutating the source;
- validate required columns;
- parse timestamps only when the column and numeric units are known;
- preserve missing values unless an explicit strategy is selected;
- optionally remove exact duplicate rows and report the count;
- convert sampled power in watts to estimated daily kWh only when the sampling interval and units have been confirmed;
- split customer-row theft data into numeric features, labels and identifiers after the label is confirmed.

## References

1. Makonin et al., “Electricity, water, and natural gas consumption of a residential house in Canada from 2012 to 2014,” *Scientific Data* (2016), [article](https://www.nature.com/articles/sdata201637).
2. Zheng et al., “Wide and Deep Convolutional Neural Networks for Electricity-Theft Detection to Secure Smart Grids,” *IEEE Transactions on Industrial Informatics* (2018), [DOI](https://doi.org/10.1109/TII.2017.2785963).
3. Raza et al., “WDSEventDB: A real-time benchmark dataset for event diagnosis for water distribution networks,” [Zenodo record](https://zenodo.org/records/17547955).
