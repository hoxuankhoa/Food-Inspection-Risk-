"""Step 2 - point-in-time establishment history features.

For every prediction row (license, date) the features are computed only from
inspection events of strictly earlier days. Nothing from the inspection being
predicted (its Result, its Violations) is used.

Output: data/processed/model_table.parquet  (one row per completed inspection
        in the modelling window, with features, target and split label)
"""
import numpy as np
import pandas as pd

from src import config

NUMERIC_FEATURES = [
    "n_prior_inspections", "n_prior_fails", "prior_fail_rate",
    "n_inspections_365d", "n_fails_365d", "n_prior_noncompleted",
    "mean_prior_violations", "mean_prior_serious",
    "last_failed", "last_conditional", "prior_conditional_rate",
    "last_n_violations", "last_n_serious",
    "days_since_last_inspection", "days_since_last_fail", "days_since_first_seen",
    "first_inspection",
    "nearby_inspections_365d", "nearby_fail_rate_365d",
    "month",
]
CATEGORICAL_FEATURES = ["type_group", "facility_group", "risk_level", "region"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Columns that must never be used as predictors (outcome of the inspection
# being predicted or derived from it).
FORBIDDEN = {"result", "fail", "violations", "n_violations", "n_serious_violations",
             "completed", "inspection_id", "n_records_same_day"}

GRID = 0.01          # degrees, ~1.1 km x 0.8 km cells
WINDOW_DAYS = 365


def _history_counts(events: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    """Per-establishment counts and last-event attributes, strictly before target date."""
    ev = events[["license", "date", "completed", "fail", "result",
                 "n_violations", "n_serious_violations"]].copy()
    ev["_kind"] = 1  # event rows sort after target rows on the same day
    ev["c_insp"] = ev["completed"].astype(int)
    ev["c_fail"] = ev["fail"]
    ev["c_cond"] = (ev["result"] == "Pass w/ Conditions").astype(int)
    ev["c_noncompleted"] = ((~ev["completed"]) & (ev["result"] != "Out of Business")).astype(int)
    ev["c_viol"] = ev["n_violations"] * ev["c_insp"]
    ev["c_serious"] = ev["n_serious_violations"] * ev["c_insp"]
    comp = ev["completed"]
    ev["last_date"] = ev["date"].where(comp)
    ev["last_failed"] = ev["fail"].where(comp).astype(float)
    ev["last_conditional"] = ev["c_cond"].where(comp).astype(float)
    ev["last_n_violations"] = ev["n_violations"].where(comp).astype(float)
    ev["last_n_serious"] = ev["n_serious_violations"].where(comp).astype(float)
    ev["last_fail_date"] = ev["date"].where(ev["fail"] == 1)
    ev["first_date"] = ev["date"]

    now = targets[["target_id", "license", "date"]].assign(_kind=0, _lag=0)
    # Pseudo-targets one window earlier: subtracting their cumulative counts
    # gives counts over [date - 365d, date).
    lag = now.assign(date=now["date"] - pd.Timedelta(days=WINDOW_DAYS), _lag=1)

    count_cols = ["c_insp", "c_fail", "c_cond", "c_noncompleted", "c_viol", "c_serious"]
    last_cols = ["last_date", "last_failed", "last_conditional", "last_n_violations", "last_n_serious",
                 "last_fail_date", "first_date"]
    allrows = pd.concat([ev, now, lag], ignore_index=True, sort=False)
    allrows[count_cols] = allrows[count_cols].fillna(0)
    allrows = allrows.sort_values(["license", "date", "_kind"], kind="stable")
    g = allrows.groupby("license", sort=False)
    allrows[count_cols] = g[count_cols].cumsum()
    allrows["first_date"] = g["first_date"].cummin()
    allrows[last_cols] = allrows.groupby("license", sort=False)[last_cols].ffill()

    cur = allrows[(allrows["_kind"] == 0) & (allrows["_lag"] == 0)].set_index("target_id")
    old = allrows[(allrows["_kind"] == 0) & (allrows["_lag"] == 1)].set_index("target_id")
    old = old.reindex(cur.index)

    out = pd.DataFrame(index=cur.index)
    out["n_prior_inspections"] = cur["c_insp"]
    out["n_prior_fails"] = cur["c_fail"]
    # Fixed Laplace-style smoothing (not estimated from data): prior of 1 fail in 4.
    out["prior_fail_rate"] = (cur["c_fail"] + 1) / (cur["c_insp"] + 4)
    out["n_inspections_365d"] = cur["c_insp"] - old["c_insp"]
    out["n_fails_365d"] = cur["c_fail"] - old["c_fail"]
    out["n_prior_noncompleted"] = cur["c_noncompleted"]
    denom = cur["c_insp"].replace(0, np.nan)
    out["mean_prior_violations"] = cur["c_viol"] / denom
    out["mean_prior_serious"] = cur["c_serious"] / denom
    out["last_failed"] = cur["last_failed"]
    out["last_conditional"] = cur["last_conditional"]
    out["prior_conditional_rate"] = (cur["c_cond"] + 1) / (cur["c_insp"] + 4)
    out["last_n_violations"] = cur["last_n_violations"]
    out["last_n_serious"] = cur["last_n_serious"]
    out["days_since_last_inspection"] = (cur["date"] - cur["last_date"]).dt.days
    out["days_since_last_fail"] = (cur["date"] - cur["last_fail_date"]).dt.days
    out["days_since_first_seen"] = (cur["date"] - cur["first_date"]).dt.days
    out["first_inspection"] = (cur["c_insp"] == 0).astype(int)
    return out


def _nearby_counts(events: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    """Completed inspections / fails in the 3x3 grid neighbourhood in the prior 365 days."""
    ev = events[events["completed"] & events["lat"].notna()]
    origin = events["date"].min()
    lat0, lon0 = ev["lat"].min(), ev["lon"].min()
    ix = ((ev["lat"] - lat0) // GRID).astype(int).to_numpy()
    iy = ((ev["lon"] - lon0) // GRID).astype(int).to_numpy()
    day = (ev["date"] - origin).dt.days.to_numpy()
    nx, ny, nd = ix.max() + 1, iy.max() + 1, day.max() + 1

    def neighbourhood_cumsum(weights):
        grid = np.zeros((nx + 2, ny + 2, nd), dtype=np.int32)
        np.add.at(grid, (ix + 1, iy + 1, day), weights)
        nb = sum(grid[1 + dx:nx + 1 + dx, 1 + dy:ny + 1 + dy]
                 for dx in (-1, 0, 1) for dy in (-1, 0, 1))
        cum = np.zeros((nx, ny, nd + 1), dtype=np.int32)
        np.cumsum(nb, axis=2, out=cum[:, :, 1:])
        return cum  # cum[..., k] = total over days < k

    cum_insp = neighbourhood_cumsum(np.ones_like(day))
    cum_fail = neighbourhood_cumsum(ev["fail"].to_numpy())

    t = targets.set_index("target_id")
    tx = np.floor((t["lat"] - lat0) / GRID)
    ty = np.floor((t["lon"] - lon0) / GRID)
    ok = (t["lat"].notna() & (tx >= 0) & (tx < nx) & (ty >= 0) & (ty < ny)).to_numpy()
    td = np.clip((t["date"] - origin).dt.days.to_numpy(), 0, nd)
    tlo = np.clip(td - WINDOW_DAYS, 0, nd)
    txi = np.where(ok, tx, 0).astype(int)
    tyi = np.where(ok, ty, 0).astype(int)

    n_insp = (cum_insp[txi, tyi, td] - cum_insp[txi, tyi, tlo]).astype(float)
    n_fail = (cum_fail[txi, tyi, td] - cum_fail[txi, tyi, tlo]).astype(float)
    n_insp[~ok] = np.nan
    n_fail[~ok] = np.nan
    out = pd.DataFrame(index=t.index)
    out["nearby_inspections_365d"] = n_insp
    out["nearby_fail_rate_365d"] = (n_fail + 1) / (n_insp + 4)
    return out


def build_features(events: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    """targets needs: target_id, license, date, lat, lon, type_group,
    facility_group, risk_level, region. Returns targets + feature columns."""
    hist = _history_counts(events, targets)
    near = _nearby_counts(events, targets)
    out = targets.set_index("target_id").join(hist).join(near)
    out["month"] = out["date"].dt.month
    return out.reset_index()


def assign_split(dates: pd.Series) -> pd.Series:
    return pd.Series(np.select(
        [dates < config.TRAIN_START, dates < config.VALID_START,
         dates < config.TEST_START, dates < config.TEST_END],
        ["history_only", "train", "valid", "test"], default="out_of_window"),
        index=dates.index)


def main() -> None:
    events = pd.read_parquet(config.EVENTS_PARQUET)
    targets = events[events["completed"]].copy()
    targets["split"] = assign_split(targets["date"])
    targets = targets[targets["split"].isin(["train", "valid", "test"])]
    targets = targets.rename(columns={"inspection_id": "target_id"})

    table = build_features(events, targets)
    assert not FORBIDDEN & set(FEATURES), "forbidden column in feature list"
    table.to_parquet(config.FEATURES_PARQUET, index=False)

    print(f"Model table: {len(table):,} rows -> {config.FEATURES_PARQUET}")
    print(table.groupby("split")["fail"].agg(["size", "mean"]))
    print(table[NUMERIC_FEATURES].describe().T.round(2))


if __name__ == "__main__":
    main()
