"""Step 6 - fairness / coverage analysis and alternative targeting policies.

Uses the selected calibrated model on the test year.

* Coverage by geography (approximate city region) and facility category under
  the global top-k rule: selection rate, recall within group, precision within
  group, calibration within group.
* Alternative policies at the same budget:
    A. Global top-k by risk (default)
    B. Regional quota: budget split by each region's share of inspections,
       top-k within each region
    C. Coverage floor: global top-k, then swap the lowest-ranked picks for the
       best unselected inspections of any region whose within-region recall is
       below a floor
* Probability-threshold policies (budget is an output, not an input).

Outputs: reports/evaluation/fairness_*.csv, policy_comparison.csv, figures.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config, viz_style as vs
from src.models.evaluate import ranking_score

GROUPS = {"region": "Region (approx. from coordinates)", "facility_group": "Facility category"}


def global_topk(df: pd.DataFrame, k: int) -> np.ndarray:
    sel = np.zeros(len(df), dtype=bool)
    sel[np.argsort(-df["score"].to_numpy(), kind="stable")[:k]] = True
    return sel


def regional_quota(df: pd.DataFrame, k: int, group: str = "region") -> np.ndarray:
    sel = np.zeros(len(df), dtype=bool)
    shares = df[group].value_counts(normalize=True)
    quotas = (shares * k).round().astype(int)
    pos = np.arange(len(df))
    for g, q in quotas.items():
        idx = pos[(df[group] == g).to_numpy()]
        top = idx[np.argsort(-df["score"].to_numpy()[idx], kind="stable")[:q]]
        sel[top] = True
    return sel


def coverage_floor(df: pd.DataFrame, k: int, floor: float, group: str = "region") -> np.ndarray:
    """Global top-k, then guarantee each group's *expected* recall >= floor.

    Expected recall uses the calibrated probabilities, so the rule could be
    applied before the visits happen (it never looks at actual outcomes)."""
    sel = global_topk(df, k)
    score = df["score"].to_numpy()
    prob = df["prob"].to_numpy()
    grp = df[group].to_numpy()
    for g in pd.unique(grp):
        in_g = grp == g
        exp_total = prob[in_g].sum()
        while prob[in_g & sel].sum() / exp_total < floor:
            add_cands = np.where(in_g & ~sel)[0]
            drop_cands = np.where(~in_g & sel)[0]
            if len(add_cands) == 0 or len(drop_cands) == 0:
                break
            sel[add_cands[np.argmax(score[add_cands])]] = True
            sel[drop_cands[np.argmin(score[drop_cands])]] = False
    return sel


def group_table(df: pd.DataFrame, sel: np.ndarray, group: str) -> pd.DataFrame:
    d = df.assign(selected=sel, captured=sel & (df["fail"] == 1))
    g = d.groupby(group).agg(
        inspections=("fail", "size"), fails=("fail", "sum"), fail_rate=("fail", "mean"),
        mean_predicted=("prob", "mean"), selected=("selected", "sum"),
        captured=("captured", "sum"))
    g["share_of_inspections"] = g["inspections"] / g["inspections"].sum()
    g["share_of_selected"] = g["selected"] / g["selected"].sum()
    g["selection_rate"] = g["selected"] / g["inspections"]
    g["recall_in_group"] = g["captured"] / g["fails"]
    g["precision_in_group"] = g["captured"] / g["selected"].replace(0, np.nan)
    g["calibration_gap"] = g["mean_predicted"] - g["fail_rate"]
    return g.sort_values("inspections", ascending=False)


def policy_summary(df: pd.DataFrame, sel: np.ndarray, name: str) -> dict:
    y = df["fail"].to_numpy()
    reg = group_table(df, sel, "region")
    reg = reg[reg.index != "Unknown"]
    fac = group_table(df, sel, "facility_group")
    fac = fac[fac["inspections"] >= 200]
    return {
        "policy": name, "inspections": int(sel.sum()), "fails_captured": int(y[sel].sum()),
        "recall": y[sel].sum() / y.sum(), "precision": y[sel].mean(),
        "expected_fails_from_scores": df["prob"].to_numpy()[sel].sum(),
        "min_region_recall": reg["recall_in_group"].min(),
        "max_region_recall": reg["recall_in_group"].max(),
        "min_facility_recall (groups >= 200)": fac["recall_in_group"].min(),
    }


def main() -> None:
    vs.apply()
    selected = (config.MODELS / "selected_model.txt").read_text().strip()
    preds = pd.read_parquet(config.TEST_PREDICTIONS)
    df = preds.assign(score=ranking_score(preds, selected), prob=preds[f"{selected}_cal"])
    df = df.reset_index(drop=True)
    n = len(df)

    # ---- coverage tables at each budget ---------------------------------------
    for group in GROUPS:
        frames = []
        for p in config.CAPACITY_LEVELS:
            t = group_table(df, global_topk(df, int(round(p * n))), group)
            frames.append(t.assign(capacity=p).reset_index())
        pd.concat(frames).to_csv(config.EVALUATION / f"fairness_{group}.csv", index=False)

    # ---- policy comparison --------------------------------------------------
    rows = []
    for p in config.CAPACITY_LEVELS:
        k = int(round(p * n))
        floor = p * 1.5  # e.g. at 10 % capacity every region gets >= 15 % expected recall
        rows += [
            {"capacity": p, **policy_summary(df, global_topk(df, k), "A. Global top-k by risk")},
            {"capacity": p, **policy_summary(df, regional_quota(df, k), "B. Regional quota (by volume)")},
            {"capacity": p, **policy_summary(df, coverage_floor(df, k, floor),
                                             f"C. Global top-k + regional floor ({floor:.0%} exp. recall)")},
        ]
    for thr in (0.30, 0.40, 0.50):
        sel = df["prob"].to_numpy() >= thr
        rows.append({"capacity": sel.mean(),
                     **policy_summary(df, sel, f"D. Threshold P(Fail) >= {thr:.2f}")})
    policies = pd.DataFrame(rows)
    policies.to_csv(config.EVALUATION / "policy_comparison.csv", index=False)

    # ---- figures at the headline budget ---------------------------------------
    k = int(round(config.HEADLINE_CAPACITY * n))
    reg_a = group_table(df, global_topk(df, k), "region")
    reg_b = group_table(df, regional_quota(df, k), "region")
    reg_a = reg_a[reg_a.index != "Unknown"]
    reg_b = reg_b.loc[reg_a.index]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ys = np.arange(len(reg_a))
    ax = axes[0]
    ax.barh(ys + 0.2, reg_a["fail_rate"] * 100, 0.38, color=vs.REFERENCE, label="Actual fail rate")
    ax.barh(ys - 0.2, reg_a["selection_rate"] * 100, 0.38, color=vs.MODEL_COLORS[selected],
            label="Share selected (global top-10%)")
    ax.set_yticks(ys, reg_a.index); ax.invert_yaxis()
    ax.set_xlabel("% of the region's test-year inspections")
    ax.set_title("Who gets selected vs. who fails")
    ax.grid(axis="y", visible=False); ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    ax = axes[1]
    ax.barh(ys - 0.2, reg_a["recall_in_group"] * 100, 0.38, color=vs.MODEL_COLORS[selected],
            label="A. Global top-k")
    ax.barh(ys + 0.2, reg_b["recall_in_group"] * 100, 0.38, color=vs.POLICY_B,
            label="B. Regional quota")
    for y_, v in zip(ys, reg_a["recall_in_group"] * 100):
        ax.text(v + 0.5, y_ - 0.2, f"{v:.0f}%", va="center", fontsize=8, color=vs.TEXT)
    for y_, v in zip(ys, reg_b["recall_in_group"] * 100):
        ax.text(v + 0.5, y_ + 0.2, f"{v:.0f}%", va="center", fontsize=8, color=vs.TEXT)
    ax.set_yticks(ys, reg_a.index); ax.invert_yaxis()
    ax.set_xlabel("% of the region's failed inspections captured")
    ax.set_title("Coverage (recall) by region at 10% capacity")
    ax.set_xlim(0, max(reg_a["recall_in_group"].max(), reg_b["recall_in_group"].max()) * 115)
    ax.grid(axis="y", visible=False); ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    fig.tight_layout()
    vs.save(fig, config.FIGURES / "fairness_region.png")

    fac = group_table(df, global_topk(df, k), "facility_group")
    fac = fac[fac["inspections"] >= 200]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ys = np.arange(len(fac))
    ax.barh(ys + 0.2, fac["fail_rate"] * 100, 0.38, color=vs.REFERENCE, label="Actual fail rate")
    ax.barh(ys - 0.2, fac["recall_in_group"] * 100, 0.38, color=vs.MODEL_COLORS[selected],
            label="Recall within group (top-10%)")
    ax.set_yticks(ys, fac.index); ax.invert_yaxis()
    ax.set_xlabel("%")
    ax.set_title("Coverage by facility category at 10% capacity")
    ax.grid(axis="y", visible=False); ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    vs.save(fig, config.FIGURES / "fairness_facility.png")

    pd.set_option("display.width", 220)
    cols = ["inspections", "fail_rate", "mean_predicted", "share_of_inspections",
            "share_of_selected", "selection_rate", "recall_in_group", "precision_in_group"]
    print(reg_a[cols].round(3).to_string())
    print(fac[cols].round(3).to_string())
    print(policies.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
