# Food Safety Risk Scoring for Inspection Prioritization

Business Analytics capstone (topic 4). **Question:** if inspection capacity is limited, which establishments should be inspected first to find as many high-risk (failed) inspections as possible?

Data: City of Chicago Food Inspections (`data/raw/food-inspections.csv`, 2010-01 to 2019-12).

## Quick start

Requires [uv](https://docs.astral.sh/uv/). It installs Python 3.12 and all dependencies into `.venv`.

```bash
uv sync                                   # create .venv from pyproject.toml / uv.lock
uv run python -m src.data.download_data   # get the raw CSV into data/raw/
uv run python -m src.run_pipeline         # full pipeline, ~1-2 min
uv run streamlit run src/dashboard/app.py # inspector dashboard
```

The raw data (about 220 MB) and all generated outputs (`data/`, `models/`, `reports/`) are not in the repository. The pipeline re-creates the outputs. The reported results come from an extract ending 2019-12-04. A newer download contains more recent years, so update the split dates in `src/config.py` before re-running. You can also place an existing extract at `data/raw/food-inspections.csv` by hand.

## Pipeline

| Step | Module | Output |
|---|---|---|
| 1. Clean and de-duplicate | `src/data/clean_data.py` | `data/processed/inspection_events.parquet`, `reports/evaluation/data_cleaning_log.md` |
| 2. Point-in-time history features | `src/features/build_features.py` | `data/processed/model_table.parquet` |
| 3. Train and calibrate models | `src/models/train_models.py` | `models/*.joblib`, `reports/evaluation/model_selection.csv` |
| 4. Evaluate on the test year | `src/models/evaluate.py` | `reports/evaluation/test_metrics.csv`, `topk_capture.csv`, `evaluation_summary.md`, figures |
| 5. Leakage audit | `src/models/leakage_audit.py` | `reports/evaluation/leakage_audit.md` |
| 6. Fairness and targeting policies | `src/fairness/fairness_policies.py` | `reports/evaluation/fairness_*.csv`, `policy_comparison.csv` |
| 7. Prioritized establishment list | `src/models/score_establishments.py` | `data/processed/priority_list.csv` / `.xlsx` |
| 8. Policy memo | `src/report/build_memo.py` | `reports/policy_memo/policy_memo.docx` (source: `policy_memo.md`) |

Settings (paths, split dates, capacity levels) are in `src/config.py`.

## Design

* **Unit of prediction:** one completed inspection (Pass, Pass w/ Conditions, Fail). **Target:** `Fail`. The overall fail rate is 22%, a moderate imbalance, so PR-AUC and top-k capture are the primary metrics.
* **Temporal split:** train 2015–2017, validation 2018 (model selection and calibration), test 2019. History features look back to 2010.
* **Features** use only events from strictly earlier days: prior inspections and fails, violations recorded at earlier visits, time since the last inspection, inspections in the last 12 months, facility type, city risk category, inspection type, approximate region, and the fail rate of nearby establishments in the past 12 months (spatial feature).
* **Models:** logistic regression and random forest (required baseline), plus cost-sensitive XGBoost (stretch). Each is calibrated with a sigmoid map on the validation year. The random forest was selected on validation PR-AUC.
* **Duplicates:** 149 duplicate IDs were dropped. 1,702 same-license, same-day records were merged into single events, keeping the worst result of the day. Re-inspections are kept as separate events, and inspection type is a feature.

## Results (test year 2019, 13,761 inspections, 22.9% failed)

| | Top 5% | Top 10% | Top 20% |
|---|---:|---:|---:|
| Failed inspections found (random forest) | 354 | 620 | 1,130 |
| Expected by random selection | 157 | 315 | 630 |
| Recall@k | 11.2% | 19.7% | 35.9% |
| Precision@k | 51.5% | 45.1% | 41.1% |

* Random forest: PR-AUC 0.391 (random: 0.229), ROC-AUC 0.70, Brier 0.162.
* Calibration matters for the cost-sensitive model: its Brier score falls from 0.234 to 0.161 after calibration.
* Coverage gap: under a pure citywide ranking, the Central region gets 2% of visits and only 4% of its fails are reached. A regional quota raises the minimum regional recall to 18% with the same total (619 vs 620 fails).

## Requirement checklist

| Requirement | Where |
|---|---|
| History features (prior failures, time since last, frequency, facility type, geography) | `src/features/build_features.py` |
| High-risk target and class imbalance | `reports/evaluation/data_cleaning_log.md`, `reports/figures/class_balance.png` |
| Calibrated risk score (Brier, reliability plot) | `train_models.py`, `reports/figures/reliability_plot.png` |
| Budgets of 5/10/20%, expected high-risk inspections found | `reports/evaluation/topk_capture.csv`, `capture_curve.png`, `fails_captured_by_budget.png` |
| Fairness/coverage by geography and facility | `reports/evaluation/fairness_region.csv`, `fairness_facility_group.csv`, figures |
| Logistic regression + tree-based model | `src/models/train_models.py` |
| Temporal split + strict leakage audit | `src/models/leakage_audit.py`, `reports/evaluation/leakage_audit.md` |
| Stretch: calibrated gradient boosting, cost-sensitive learning | `xgb_cost_sens` in `train_models.py` |
| Stretch: spatial features from prior nearby outcomes | `nearby_fail_rate_365d`, `nearby_inspections_365d` |
| Stretch: alternative threshold policies (recall / capacity / subgroup coverage) | `reports/evaluation/policy_comparison.csv` |
| Deliverable: pipeline + prioritized establishment list | `src/run_pipeline.py`, `data/processed/priority_list.xlsx` |
| Deliverable: dashboard (risk distribution, map, capacity slider / threshold) | `src/dashboard/app.py` |
| Deliverable: policy memo | `reports/policy_memo/policy_memo.docx` |
| Guardrail: no "avoided incidents" claims | memo §5, dashboard "Method & limitations" tab |

## Limitations

The data ends in December 2019, discrimination is moderate, and the highest-risk decile is over-predicted by about 15–20%. Regions are approximate latitude/longitude bands. See memo §5.
