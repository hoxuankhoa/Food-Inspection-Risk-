"""Shared chart styling (validated categorical palette, recessive axes)."""
import matplotlib as mpl
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
REFERENCE = "#8a8984"   # neutral grey for random / heuristic reference lines

# Colour follows the entity: each model keeps its slot in every chart.
MODEL_COLORS = {
    "logreg": "#2a78d6",          # slot 1 blue
    "random_forest": "#eb6834",   # slot 2 orange
    "xgb_cost_sens": "#1baf7a",   # slot 3 aqua
}
MODEL_LABELS = {
    "logreg": "Logistic regression",
    "random_forest": "Random forest",
    "xgb_cost_sens": "XGBoost (cost-sensitive)",
    "heuristic": "Rule: prior fail rate",
    "random": "Random selection",
}
SEQUENTIAL = "#2a78d6"
POLICY_B = "#6250d6"     # slot 7 violet: alternative policy, distinct from model hues


def apply() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "savefig.dpi": 150, "savefig.bbox": "tight",
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "axes.titlecolor": TEXT,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "xtick.color": TEXT_2, "ytick.color": TEXT_2, "xtick.labelsize": 9,
        "ytick.labelsize": 9, "legend.frameon": False, "legend.fontsize": 9,
        "legend.labelcolor": TEXT, "lines.linewidth": 2, "font.size": 10,
    })


def save(fig: plt.Figure, path) -> None:
    fig.savefig(path)
    plt.close(fig)
