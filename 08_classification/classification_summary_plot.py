"""
Summary figure of the classification: fold-wise balanced accuracy per task,
cohort and feature set, with stars for the Holm-corrected comparisons against
regional atrophy. Also prints and saves the numbers for the results table.

Needs classification_pipeline.py and compare_feature_sets_ttest.py to be run
first. Cohorts without results are skipped.

Output: outputs/classification/fig8_classification_summary.pdf / .png, fig8_table6_values.csv
Usage:  python classification_summary_plot.py
"""
import json
from math import sqrt
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "outputs" / "classification"

COHORTS = {"TUM": "tum", "NIFD": "nifd", "NACC": "nacc", "NIFD_3C": "nifd_3class"}
PANELS = [
    ("TUM", "01_healthy_vs_sick", "TUM\nhealthy vs. sick"),
    ("NIFD", "01_healthy_vs_sick", "NIFD\nhealthy vs. sick"),
    ("NACC", "01_healthy_vs_sick", "NACC\nhealthy vs. sick"),
    ("TUM", "02_ftd_subtype_classification", "TUM\nbvFTD vs. SD"),
    ("NIFD", "02_ftd_subtype_classification", "NIFD\nbvFTD vs. SD"),
    ("NIFD_3C", "02_ftd_subtype_classification", "NIFD\nbvFTD vs. SD\nvs. nfvPPA"),
]
FEATURE_SETS = [("atrophy_regional", "Reg.", "#8a8a8a"),
                ("celltype", "Cell", "#2a7f62"),
                ("combined", "Comb.", "#7b4fa0")]


def nested_cv(J, task, fs):
    return J[task]["per_feature_set"][fs]["per_hemi_mode"][0]["nested_cv"]


def folds(J, task, fs):
    return np.array(nested_cv(J, task, fs)["per_fold_balanced_accuracy"])


def n_classes(J, task):
    return len(nested_cv(J, task, "atrophy_regional")["classes"])


def nadeau_bengio(J, task, fs):
    """Corrected repeated k-fold t-test of fs vs. atrophy_regional."""
    a, b = nested_cv(J, task, "atrophy_regional"), nested_cv(J, task, fs)
    d = np.array(b["per_fold_balanced_accuracy"]) - np.array(a["per_fold_balanced_accuracy"])
    n = len(d)
    se = sqrt((1 / n + b["mean_n_test"] / b["mean_n_train"]) * d.var(ddof=1))
    t = d.mean() / se
    tc = stats.t.ppf(0.975, n - 1)
    return d.mean(), d.mean() - tc * se, d.mean() + tc * se, 2 * stats.t.sf(abs(t), n - 1)


def holm(p):
    p = np.asarray(p, float)
    adj = np.empty_like(p)
    running = 0.0
    for i, k in enumerate(np.argsort(p)):
        running = max(running, min(1.0, (len(p) - i) * p[k]))
        adj[k] = running
    return adj


def main():
    J, tests = {}, {}
    for cohort, ds in COHORTS.items():
        summary = ROOT / ds / "run_summary_all_tasks.json"
        ttest = ROOT / "significance" / f"{ds}_reported_comparisons_ttest" / "feature_set_ttest_results.csv"
        if summary.exists() and ttest.exists():
            J[cohort] = json.loads(summary.read_text())
            tests[cohort] = pd.read_csv(ttest)
        else:
            print(f"Note: results for {cohort} not found - skipped")
    panels = [p for p in PANELS if p[0] in J and p[1] in J[p[0]]]

    rows = []
    for cohort, task, _ in panels:
        for fs in ["celltype", "combined"]:
            t = tests[cohort]
            x = t[(t["task"] == task) & (t["feature_b"] == fs) & (t["hemi_mode"] == "average")].iloc[0]
            d, lo, hi, p = nadeau_bengio(J[cohort], task, fs)
            rows.append({"cohort": cohort, "task": task, "feature_set": fs, "n_classes": n_classes(J[cohort], task),
                         "mean_diff": x["mean_difference"], "ci_low": x["ci95_low"], "ci_high": x["ci95_high"],
                         "p": x["p_two_sided"], "p_holm_file": x["p_holm"], "check_mean_diff": d, "check_p": p})
    res = pd.DataFrame(rows)
    res["p_holm_reported"] = np.nan
    for cohort in res["cohort"].unique():
        m = res["cohort"] == cohort
        res.loc[m, "p_holm_reported"] = holm(res.loc[m, "p"])
    assert np.allclose(res["mean_diff"], res["check_mean_diff"], atol=1e-9), "mean difference mismatch"
    assert np.allclose(res["p"], res["check_p"], rtol=1e-6), "p-value mismatch"
    assert np.allclose(res["p_holm_file"], res["p_holm_reported"], rtol=1e-6), "Holm-adjusted p mismatch"

    print("Nested CV balanced accuracy (mean +- SD over the outer folds)")
    for cohort, task, title in panels:
        vals = "  ".join(f"{fs}: {folds(J[cohort], task, fs).mean():.3f} +- {folds(J[cohort], task, fs).std(ddof=1):.3f}"
                         for fs, _, _ in FEATURE_SETS)
        print(f"  {title.replace(chr(10), ' '):36s} {vals}")
    print("\nComparisons with regional atrophy")
    with pd.option_context("display.width", 220, "display.float_format", "{:.4g}".format):
        print(res[["cohort", "task", "n_classes", "feature_set", "mean_diff", "ci_low", "ci_high", "p",
                   "p_holm_reported"]].to_string(index=False))

    plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans",
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, len(panels), figsize=(6.3 * len(panels) / 6, 3.0), sharey=True, squeeze=False)
    rng = np.random.default_rng(0)
    for ax, (cohort, task, title) in zip(axes[0], panels):
        for i, (fs, _, c) in enumerate(FEATURE_SETS):
            y = folds(J[cohort], task, fs)
            ax.scatter(i + rng.uniform(-0.18, 0.18, len(y)), y, s=5, color=c, alpha=0.35, linewidth=0)
            ax.plot([i - 0.28, i + 0.28], [y.mean()] * 2, color=c, lw=2.2, solid_capstyle="round")
        for j, fs in enumerate(["celltype", "combined"]):
            pa = res[(res.cohort == cohort) & (res.task == task) & (res.feature_set == fs)]["p_holm_reported"].iloc[0]
            if pa < 0.05:
                yy = 1.02 + 0.05 * j
                ax.plot([0, j + 1], [yy, yy], color="#333", lw=0.7)
                stars = "***" if pa < 0.001 else ("**" if pa < 0.01 else "*")
                ax.text((j + 1) / 2, yy + 0.003, stars, ha="center", va="bottom", fontsize=7)
        ax.set_title(title, fontsize=7.5)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels([f[1] for f in FEATURE_SETS], fontsize=6.5, rotation=90)
        ax.axhline(1 / n_classes(J[cohort], task), color="#bbb", lw=0.7, ls=":")
        ax.set_xlim(-0.6, 2.6)
    axes[0, 0].set_ylabel("Balanced accuracy (outer folds)")
    axes[0, 0].set_ylim(0.28, 1.13)
    axes[0, 0].set_yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.tight_layout(w_pad=0.4)

    ROOT.mkdir(parents=True, exist_ok=True)
    out = ROOT / "fig8_classification_summary.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=300)
    res.to_csv(ROOT / "fig8_table6_values.csv", index=False)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
