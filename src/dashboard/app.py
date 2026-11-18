"""Inspector dashboard.  Run:  uv run streamlit run src/dashboard/app.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import config, viz_style as vs  # noqa: E402
from src.fairness.fairness_policies import group_table, regional_quota  # noqa: E402
from src.models.evaluate import capture_curve, ranking_score  # noqa: E402

st.set_page_config(page_title="Food Safety Inspection Prioritization", layout="wide")

SELECTED_COLOR = vs.MODEL_COLORS["logreg"]   # blue = "selected for inspection"
OTHER_COLOR = "#b5b4ae"


@st.cache_data
def load():
    pl = pd.read_csv(config.PRIORITY_LIST_CSV)
    preds = pd.read_parquet(config.TEST_PREDICTIONS)
    model = (config.MODELS / "selected_model.txt").read_text().strip()
    preds = preds.assign(score=ranking_score(preds, model), prob=preds[f"{model}_cal"])
    metrics = pd.read_csv(config.EVALUATION / "test_metrics.csv")
    policies = pd.read_csv(config.EVALUATION / "policy_comparison.csv")
    return pl, preds, model, metrics, policies


def style(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10),
                      plot_bgcolor=vs.SURFACE, paper_bgcolor=vs.SURFACE,
                      font=dict(color=vs.TEXT_2), title_font=dict(color=vs.TEXT, size=15),
                      legend=dict(orientation="h", y=-0.2), hoverlabel=dict(bgcolor="white"))
    fig.update_xaxes(gridcolor=vs.GRID, zeroline=False)
    fig.update_yaxes(gridcolor=vs.GRID, zeroline=False)
    return fig


pl, preds, model_name, metrics, policies = load()
model_label = vs.MODEL_LABELS[model_name]
scored_as_of = pl["scored_as_of"].iloc[0]

# ---------------------------------------------------------------- sidebar ----
st.sidebar.header("Scope")
regions = sorted(pl["region"].unique())
facilities = sorted(pl["facility_group"].unique())
sel_regions = st.sidebar.multiselect("Region", regions, default=regions)
sel_fac = st.sidebar.multiselect("Facility category", facilities, default=facilities)

st.sidebar.header("Targeting rule")
mode = st.sidebar.radio("Select establishments by", ["Inspection capacity", "Risk threshold"])
if mode == "Inspection capacity":
    capacity_pct = st.sidebar.slider("Capacity (% of establishments in scope)", 1, 40, 10)
    policy = st.sidebar.radio("Allocation", ["Global top-k by risk", "Regional quota (by volume)"],
                              help="Regional quota splits the budget across regions in "
                                   "proportion to their number of establishments, then "
                                   "takes the highest risk inside each region.")
else:
    threshold = st.sidebar.slider("Inspect if predicted P(Fail) ≥", 0.10, 0.75, 0.40, 0.01)
    policy = "Threshold"

scope = pl[pl["region"].isin(sel_regions) & pl["facility_group"].isin(sel_fac)].copy()
scope = scope.sort_values("risk_score", ascending=False).reset_index(drop=True)
if scope.empty:
    st.warning("No establishments in the selected scope.")
    st.stop()

if mode == "Inspection capacity":
    k = max(1, int(round(capacity_pct / 100 * len(scope))))
    if policy == "Global top-k by risk":
        selected = np.arange(len(scope)) < k
    else:
        selected = regional_quota(scope.assign(score=scope["risk_score"]), k)
else:
    selected = (scope["risk_score"] >= threshold).to_numpy()
    k = int(selected.sum())
    capacity_pct = 100 * k / len(scope)
scope["selected"] = selected
chosen = scope[scope["selected"]]

# ----------------------------------------------------------------- header ----
st.title("Food Safety Risk Scoring: Inspection Prioritization")
st.caption(f"Chicago food establishments active in the last 2 years, scored as of {scored_as_of} "
           f"for their next routine inspection. Model: {model_label}, calibrated on 2018, "
           "back-tested on 2019. Risk score = estimated probability that the inspection "
           "results in **Fail**.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Establishments in scope", f"{len(scope):,}")
c2.metric("Selected for inspection", f"{len(chosen):,}", f"{capacity_pct:.1f}% of scope",
          delta_color="off")
c3.metric("Expected fails among selected", f"{chosen['risk_score'].sum():,.0f}",
          f"vs {len(chosen) * scope['risk_score'].mean():,.0f} if chosen at random",
          delta_color="off",
          help="Sum of calibrated probabilities. In the 2019 back-test the model's own "
               "expectation for the top 10% overstated the actual count by ~18%; see the "
               "Capacity planning tab for back-tested numbers.")
c4.metric("Mean risk: selected vs all", f"{chosen['risk_score'].mean():.2f}",
          f"all: {scope['risk_score'].mean():.2f}", delta_color="off")

tab_list, tab_cap, tab_fair, tab_model, tab_about = st.tabs(
    ["Priority list & map", "Capacity planning (2019 back-test)", "Fairness & coverage",
     "Model quality", "Method & limitations"])

# ---------------------------------------------------------- priority list ----
with tab_list:
    left, right = st.columns([1, 1.25])
    with left:
        fig = px.histogram(scope, x="risk_score", color=np.where(scope["selected"], "Selected",
                                                                  "Not selected"),
                           nbins=50, color_discrete_map={"Selected": SELECTED_COLOR,
                                                         "Not selected": OTHER_COLOR},
                           labels={"risk_score": "Predicted P(Fail)", "color": ""},
                           title="Risk score distribution")
        fig.update_traces(marker_line_color=vs.SURFACE, marker_line_width=1)
        if mode == "Risk threshold":
            fig.add_vline(threshold, line_dash="dash", line_color=vs.TEXT_2,
                          annotation_text=f"threshold {threshold:.2f}")
        fig.update_layout(bargap=0.05, yaxis_title="Establishments", barmode="stack")
        st.plotly_chart(style(fig), width="stretch")

        by_region = scope.groupby("region").agg(establishments=("license", "size"),
                                                selected=("selected", "sum"),
                                                mean_risk=("risk_score", "mean"))
        by_region["selected_%"] = 100 * by_region["selected"] / by_region["establishments"]
        st.dataframe(by_region.sort_values("establishments", ascending=False)
                     .style.format({"mean_risk": "{:.3f}", "selected_%": "{:.1f}"}),
                     width="stretch")
    with right:
        geo = scope.dropna(subset=["lat", "lon"]).copy()
        geo["status"] = np.where(geo["selected"], "Selected", "Not selected")
        geo = geo.sort_values("selected")  # draw selected on top
        fig = px.scatter_map(
            geo, lat="lat", lon="lon", color="status",
            color_discrete_map={"Selected": SELECTED_COLOR, "Not selected": OTHER_COLOR},
            hover_name="dba_name",
            hover_data={"address": True, "facility_group": True, "risk_score": ":.2f",
                        "priority_rank": True, "why_flagged": True, "lat": False,
                        "lon": False, "status": False},
            zoom=9.3, center={"lat": 41.84, "lon": -87.68}, map_style="carto-positron",
            title="Where the selected establishments are")
        fig.update_traces(marker=dict(size=6))
        st.plotly_chart(style(fig, 560), width="stretch")

    st.subheader(f"Inspection list ({len(chosen):,} establishments)")
    show = chosen[["priority_rank", "dba_name", "address", "zip", "region", "facility_type",
                   "risk_level", "risk_score", "why_flagged", "n_prior_inspections",
                   "n_prior_fails", "days_since_last_inspection"]]
    st.dataframe(show.style.format({"risk_score": "{:.3f}",
                                    "days_since_last_inspection": "{:.0f}"}),
                 width="stretch", hide_index=True, height=420)
    st.download_button("Download list (CSV)", show.to_csv(index=False).encode("utf-8"),
                       file_name="inspection_list.csv", mime="text/csv")

# ------------------------------------------------------- capacity planning ----
with tab_cap:
    st.markdown(
        "How the same rule would have performed on **2019 inspections the model never saw** "
        "(13,761 completed inspections, 22.9% failed). Use this to set the budget; the "
        "numbers below are actual outcomes, not model expectations.")
    y = preds["fail"].to_numpy()
    share, cum = capture_curve(y, preds["score"].to_numpy())
    p = min(capacity_pct, 100) / 100
    k_bt = int(round(p * len(preds)))
    order = np.argsort(-preds["score"].to_numpy(), kind="stable")
    captured = int(y[order[:k_bt]].sum())
    random_exp = k_bt * y.mean()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Inspections at this capacity", f"{k_bt:,}", f"{p:.0%} of 2019 volume",
              delta_color="off")
    m2.metric("Failed inspections found", f"{captured:,}",
              f"+{captured - random_exp:,.0f} vs random", delta_color="normal")
    m3.metric("Recall (share of all fails found)", f"{captured / y.sum():.1%}")
    m4.metric("Hit rate (precision)", f"{captured / max(k_bt, 1):.1%}",
              f"random: {y.mean():.1%}", delta_color="off")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=share * 100, y=cum * 100, name=model_label,
                             line=dict(color=vs.MODEL_COLORS[model_name], width=2),
                             hovertemplate="capacity %{x:.1f}%<br>fails captured %{y:.1f}%"
                                           "<extra></extra>"))
    fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], name="Random selection",
                             line=dict(color=vs.REFERENCE, dash="dot", width=1.5),
                             hoverinfo="skip"))
    fig.add_vline(p * 100, line_color=vs.TEXT_2, line_dash="dash",
                  annotation_text=f"{p:.0%} → {captured / y.sum():.0%} of fails")
    fig.update_layout(title="Share of failed inspections captured vs. capacity (2019)",
                      xaxis_title="Capacity (% of inspections, highest risk first)",
                      yaxis_title="% of failed inspections captured", hovermode="x unified")
    fig.update_xaxes(range=[0, 60]); fig.update_yaxes(range=[0, 100])
    st.plotly_chart(style(fig, 420), width="stretch")

    topk = pd.read_csv(config.EVALUATION / "topk_capture.csv")
    t = topk[topk["capacity"].isin(config.CAPACITY_LEVELS)]
    t = t.assign(model=t["model"].map(vs.MODEL_LABELS))
    st.markdown("**Fixed budgets, all models (2019 back-test)**")
    st.dataframe(t[["subset", "model", "capacity", "inspections_k", "fails_captured",
                    "random_expected_fails", "recall_at_k", "precision_at_k", "lift_vs_random"]]
                 .style.format({"capacity": "{:.0%}", "recall_at_k": "{:.1%}",
                                "precision_at_k": "{:.1%}", "lift_vs_random": "{:.2f}x",
                                "random_expected_fails": "{:.0f}"}),
                 width="stretch", hide_index=True)

# ---------------------------------------------------------------- fairness ----
with tab_fair:
    st.markdown(
        f"Coverage on the 2019 back-test at **{p:.0%} capacity**. *Recall in group* = share of "
        "that group's failed inspections that the rule would have reached. A rule that "
        "maximises total fails found can still leave a group almost unvisited.")
    bt = preds.reset_index(drop=True)
    sel_a = np.zeros(len(bt), dtype=bool)
    sel_a[order[:k_bt]] = True
    sel_b = regional_quota(bt, k_bt)
    cols = ["inspections", "fail_rate", "mean_predicted", "share_of_inspections",
            "share_of_selected", "recall_in_group", "precision_in_group"]
    fmt = {c: "{:.1%}" for c in cols if c != "inspections"}
    ga = group_table(bt, sel_a, "region")
    gb = group_table(bt, sel_b, "region")
    comp = pd.DataFrame({"A. Global top-k": ga["recall_in_group"],
                         "B. Regional quota": gb["recall_in_group"]}).drop(index="Unknown",
                                                                          errors="ignore")
    comp = comp.reset_index().melt(id_vars="region", var_name="policy", value_name="recall")
    fig = px.bar(comp, y="region", x="recall", color="policy", barmode="group", orientation="h",
                 color_discrete_map={"A. Global top-k": vs.MODEL_COLORS[model_name],
                                     "B. Regional quota": vs.POLICY_B},
                 labels={"recall": "Recall in region", "region": ""},
                 title="Share of each region's failed inspections reached")
    fig.update_xaxes(tickformat=".0%")
    fig.update_traces(hovertemplate="%{y}: %{x:.1%}<extra>%{fullData.name}</extra>")
    st.plotly_chart(style(fig, 400), width="stretch")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**By region, global top-k**")
        st.dataframe(ga[cols].style.format(fmt), width="stretch")
    with c2:
        st.markdown("**By facility category, global top-k**")
        st.dataframe(group_table(bt, sel_a, "facility_group")[cols].style.format(fmt),
                     width="stretch")
    st.markdown("**Policy comparison (2019 back-test)**")
    st.dataframe(policies.style.format({c: "{:.1%}" for c in policies.columns
                                        if "recall" in c or c in ("precision", "capacity")}
                                       | {"expected_fails_from_scores": "{:.0f}"}),
                 width="stretch", hide_index=True)

# ------------------------------------------------------------ model quality ----
with tab_model:
    show_m = metrics.assign(model=metrics["model"].map(vs.MODEL_LABELS))
    st.dataframe(show_m.style.format({c: "{:.4f}" for c in show_m.columns
                                      if show_m[c].dtype.kind == "f"}),
                 width="stretch", hide_index=True)
    c1, c2 = st.columns(2)
    c1.image(str(config.FIGURES / "reliability_plot.png"))
    c2.image(str(config.FIGURES / "pr_curves.png"))
    c1.image(str(config.FIGURES / "permutation_importance.png"))
    c2.image(str(config.FIGURES / "risk_distribution.png"))

# ------------------------------------------------------------------- about ----
with tab_about:
    st.markdown((config.MEMO_DIR / "dashboard_notes.md").read_text(encoding="utf-8"))
