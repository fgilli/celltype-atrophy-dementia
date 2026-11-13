"""
Plots the mean within-group Spearman rho from heterogeneity_summary.csv
(atrophy_heterogeneity.py) for all groups of the three cohorts.

Output: outputs/heterogeneity/fig7_heterogeneity_summary.pdf / .png, fig7_plotted_values.csv
Usage:  python heterogeneity_summary_plot.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[1]

CONFIG = {
    "summary_csv": REPO / "outputs" / "heterogeneity" / "heterogeneity_summary.csv",
    "out_dir": REPO / "outputs" / "heterogeneity",
    "rho_column": "mean_spearman_rho",
}

X_ORDER = ["CN", "AD", "FTD", "LBD", "VAD", "AD (typical)", "PCA", "bvFTD", "SD", "nfvPPA"]
COLORS = {"TUM": "#1f5fa8", "NIFD": "#c2410c", "NACC": "#5b5b5b"}
MARKERS = {"TUM": "o", "NIFD": "s", "NACC": "D"}
OFFSET = {"TUM": -0.24, "NIFD": 0.0, "NACC": 0.24}


def main():
    cfg = CONFIG
    h = pd.read_csv(cfg["summary_csv"])

    rows = []
    for _, r in h.iterrows():
        g = r["group"]
        if r["grouping_var"] == "subtype":
            if g == "CN":
                continue
            if g == "AD":
                g = "AD (typical)"
        if g not in X_ORDER:
            continue
        rows.append({"dataset": r["dataset"], "label": g, "n": int(r["n_subjects"]), "rho": r[cfg["rho_column"]]})
    d = pd.DataFrame(rows)

    plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans",
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(6.3, 2.9))
    for ds in ["TUM", "NIFD", "NACC"]:
        s = d[d["dataset"] == ds]
        xs = [X_ORDER.index(g) + OFFSET[ds] for g in s["label"]]
        for x, (_, r) in zip(xs, s.iterrows()):
            ax.vlines(x, 0, r["rho"], color=COLORS[ds], lw=1.2, alpha=0.6)
            ax.text(x, max(r["rho"], 0) + 0.015, f"{r['n']}", ha="center", va="bottom", fontsize=5, color="#444")
        ax.scatter(xs, s["rho"], color=COLORS[ds], marker=MARKERS[ds], s=30, zorder=3,
                   label=ds, edgecolor="white", linewidth=0.6)
    ax.axvline(4.5, color="#bbb", lw=0.8, ls="--")
    ax.text(2, 0.55, "Diagnostic families", ha="center", fontsize=8, color="#444")
    ax.text(7, 0.55, "Clinical subtypes", ha="center", fontsize=8, color="#444")
    ax.set_xticks(range(len(X_ORDER)))
    ax.set_xticklabels(X_ORDER, fontsize=8)
    ax.set_ylabel(r"Mean within-group Spearman $\rho$")
    ax.set_ylim(min(-0.06, d["rho"].min() - 0.02), 0.58)
    ax.axhline(0, color="#888", lw=0.6)
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    ax.yaxis.grid(True, color="#eee")
    ax.set_axisbelow(True)
    plt.tight_layout()

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "fig7_heterogeneity_summary.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=300)
    d.to_csv(out_dir / "fig7_plotted_values.csv", index=False)
    print(d.to_string(index=False))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
