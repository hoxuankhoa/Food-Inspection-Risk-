"""Shared paths and settings for the food-inspection risk pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] #Path(__file__).resolve trỏ đến scr/config.py, parents[0] là thư mục src/ còn parents[1] là thư mục food-inspection-risk-main/
#ROOT sẽ trỏ đến thư mục gốc của project
RAW_CSV = ROOT / "data" / "raw" / "food-inspections.csv"
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
EVALUATION = REPORTS / "evaluation"
MEMO_DIR = REPORTS / "policy_memo"

EVENTS_PARQUET = PROCESSED / "inspection_events.parquet"
FEATURES_PARQUET = PROCESSED / "model_table.parquet"
TEST_PREDICTIONS = PROCESSED / "test_predictions.parquet"
PRIORITY_LIST_CSV = PROCESSED / "priority_list.csv"
PRIORITY_LIST_XLSX = PROCESSED / "priority_list.xlsx"

# Temporal split by inspection date (inclusive start, exclusive end).
# History features look back over the full record (2010+); only the rows
# being predicted are restricted to these windows.
# Update these dates after downloading a newer extract of the data.
TRAIN_START = "2015-01-01"
VALID_START = "2018-01-01"   # validation year: model selection + calibration
TEST_START = "2019-01-01"    # held-out test year: all reported metrics
TEST_END = "2020-01-01"

# Inspection budgets evaluated, as a share of inspections in the period.
CAPACITY_LEVELS = [0.05, 0.10, 0.20]
# Budget used for the headline numbers, fairness tables and policy memo.
HEADLINE_CAPACITY = 0.10

RANDOM_STATE = 42

for _d in (RAW_CSV.parent, PROCESSED, MODELS, FIGURES, EVALUATION, MEMO_DIR):
    _d.mkdir(parents=True, exist_ok=True)
