"""
Statistical comparison of feature sets: corrected repeated k-fold t-test.

Within one run of classification_pipeline.py, all feature sets of a task
are evaluated on identical outer folds (verified here via
oof_predictions.csv). For every reported comparison, the paired fold
differences

    d_rk = balanced_accuracy_B(r, k) - balanced_accuracy_A(r, k)

over the 5 x 10 outer folds are tested with the corrected repeated k-fold
t-test of Nadeau & Bengio (2003), as in the scikit-learn example
"Statistical comparison of models using grid search":

    corrected var = var(d, ddof=1) * (1 / (k * r) + n_test / n_train)
    t = mean(d) / sqrt(corrected var),  df = k * r - 1

Feature set A is atrophy_regional (reference), B is celltype and combined.
Holm (and Benjamini-Hochberg) adjustment is applied within each run over
exactly the comparisons that pass the filters (tasks, hemi_modes), i.e.
over the comparisons reported in the thesis (TUM / NIFD: 2 tasks x 2
comparisons, NACC: 1 x 2).

Outputs (CONFIG["out_dir"])
    feature_set_ttest_results.csv
    feature_set_paired_fold_differences_<feature_b>.csv
    feature_set_ttest_config.json

Usage
    python compare_feature_sets_ttest.py --profile tum    (tum | nifd | nifd_3class | nacc)
"""

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as t_dist

REPO = Path(__file__).resolve().parents[1]
CLASSIFICATION_ROOT = REPO / "outputs" / "classification"

CONFIG = {
    "feature_a": "atrophy_regional",
    "feature_b": ["celltype", "combined"],
    "score_col": "held_out_balanced_accuracy",
    "tasks": ["01_healthy_vs_sick", "02_ftd_subtype_classification"],
    "hemi_modes": ["average"],
    # "two-sided", "greater" (B better than A) or "less"
    "alternative": "two-sided",
    "alpha": 0.05,
    "known_feature_sets": ["atrophy_regional", "celltype", "combined"],
    "min_recommended_pairs": 20,
}

# Tasks reported per profile (defines the Holm family)
REPORTED_TASKS = {
    "tum": ["01_healthy_vs_sick", "02_ftd_subtype_classification"],
    "nifd": ["01_healthy_vs_sick", "02_ftd_subtype_classification"],
    "nifd_3class": ["02_ftd_subtype_classification"],
    "nacc": ["01_healthy_vs_sick"],
}


# ============================================================
# Statistics
# ============================================================
def compute_corrected_ttest(differences, n_train, n_test):
    """t, df, one- and two-sided p-values and corrected 95% CI of the mean difference."""
    differences = np.asarray(differences, dtype=float)
    df = len(differences) - 1
    mean = float(np.mean(differences))
    std = float(np.sqrt(np.var(differences, ddof=1) * (1 / len(differences) + n_test / n_train)))

    if std == 0.0:
        if mean == 0.0:
            return {"t_stat": 0.0, "df": df, "p_greater": 1.0, "p_less": 1.0, "p_two_sided": 1.0,
                    "ci95_low": 0.0, "ci95_high": 0.0, "corrected_std": 0.0}
        sign = np.sign(mean)
        return {"t_stat": float(sign * np.inf), "df": df, "p_greater": 0.0 if sign > 0 else 1.0,
                "p_less": 0.0 if sign < 0 else 1.0, "p_two_sided": 0.0,
                "ci95_low": mean, "ci95_high": mean, "corrected_std": 0.0}

    t_stat = mean / std
    t_crit = t_dist.ppf(0.975, df)
    return {
        "t_stat": float(t_stat),
        "df": df,
        "p_greater": float(t_dist.sf(t_stat, df)),
        "p_less": float(t_dist.cdf(t_stat, df)),
        "p_two_sided": float(2 * t_dist.sf(abs(t_stat), df)),
        "ci95_low": mean - t_crit * std,
        "ci95_high": mean + t_crit * std,
        "corrected_std": std,
    }


def holm_adjust(pvals):
    p = np.asarray(pvals, dtype=float)
    adj = np.empty(len(p))
    running_max = 0.0
    for rank, idx in enumerate(np.argsort(p)):
        running_max = max(running_max, (len(p) - rank) * p[idx])
        adj[idx] = min(1.0, running_max)
    return adj


def bh_adjust(pvals):
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    adj = np.empty(m)
    running_min = 1.0
    for i, idx in enumerate(np.argsort(p)[::-1]):
        running_min = min(running_min, p[idx] * m / (m - i))
        adj[idx] = running_min
    return adj


# ============================================================
# Pairing of the runs
# ============================================================
def parse_rel_path(rel_dir, cfg):
    """'<task>_<feature_set>/<hemi_mode>/nested_cv' -> (task, feature_set, hemi_mode)."""
    parts = rel_dir.replace("\\", "/").split("/")
    task_dir, hemi_mode = parts[0], parts[1]
    for fs in cfg["known_feature_sets"]:
        if task_dir.endswith("_" + fs):
            return task_dir[: -len(fs) - 1], fs, hemi_mode
    return task_dir, None, hemi_mode


def check_identical_folds(dir_a, dir_b):
    """Both runs must hold out the same subjects in every (repeat, fold)."""
    frames = []
    for d in (dir_a, dir_b):
        path = os.path.join(d, "oof_predictions.csv")
        if not os.path.exists(path):
            return False, "oof_predictions.csv missing"
        o = pd.read_csv(path)
        o["subject_id"] = o["subject_id"].astype(str)
        frames.append(o[["subject_id", "repeat_id", "fold_id"]].sort_values(["repeat_id", "subject_id"])
                      .reset_index(drop=True))
    if len(frames[0]) != len(frames[1]):
        return False, "different number of out-of-fold rows"
    if not frames[0].equals(frames[1]):
        return False, "fold assignments differ"
    return True, "identical folds"


def compare_one(path_a, path_b, cfg):
    key, col = ["repeat_id", "fold_id"], cfg["score_col"]
    a, b = pd.read_csv(path_a), pd.read_csv(path_b)
    merged = a[key + ["n_train", "n_test", col]].merge(
        b[key + ["n_train", "n_test", col]], on=key, suffixes=("_a", "_b"), how="inner", validate="one_to_one")
    if len(merged) != len(a) or len(merged) != len(b):
        raise ValueError("fold keys differ between the two feature sets")
    if not (merged["n_test_a"] == merged["n_test_b"]).all():
        raise ValueError("n_test differs between the two feature sets")
    merged = merged[merged[[f"{col}_a", f"{col}_b"]].notna().all(axis=1)].copy()
    merged["difference"] = merged[f"{col}_b"] - merged[f"{col}_a"]
    n_train, n_test = float(merged["n_train_a"].mean()), float(merged["n_test_a"].mean())
    return merged, n_train, n_test, compute_corrected_ttest(merged["difference"], n_train, n_test)


def find_pairs(run_dir, cfg, fb):
    """Yield (rel_dir_a, rel_dir_b, task, hemi_mode) for every comparison that passes the filters."""
    for path_a in sorted(glob.glob(os.path.join(glob.escape(run_dir), "**", "nested_cv_folds.csv"), recursive=True)):
        rel_a = os.path.relpath(os.path.dirname(path_a), run_dir)
        task, feature_set, hemi_mode = parse_rel_path(rel_a, cfg)
        if feature_set != cfg["feature_a"]:
            continue
        if cfg["hemi_modes"] and hemi_mode not in cfg["hemi_modes"]:
            continue
        if cfg["tasks"] and task not in cfg["tasks"]:
            continue
        parts = rel_a.replace("\\", "/").split("/")
        parts[0] = f"{task}_{fb}"
        rel_b = os.path.join(*parts)
        if not os.path.exists(os.path.join(run_dir, rel_b, "nested_cv_folds.csv")):
            print(f"[warn] {rel_a}: no '{fb}' counterpart - skipped")
            continue
        yield rel_a, rel_b, task, hemi_mode


def main():
    parser = argparse.ArgumentParser(description="Corrected repeated k-fold t-test between feature sets.")
    parser.add_argument("--profile", required=True, choices=sorted(REPORTED_TASKS))
    args = parser.parse_args()

    cfg = dict(CONFIG, tasks=REPORTED_TASKS[args.profile])
    run_dir = str(CLASSIFICATION_ROOT / args.profile)
    out_dir = CLASSIFICATION_ROOT / "significance" / f"{args.profile}_reported_comparisons_ttest"
    out_dir.mkdir(parents=True, exist_ok=True)
    fa, col = cfg["feature_a"], cfg["score_col"]

    rows, long_rows = [], {fb: [] for fb in cfg["feature_b"]}
    for fb in cfg["feature_b"]:
        for rel_a, rel_b, task, hemi_mode in find_pairs(run_dir, cfg, fb):
            dir_a, dir_b = os.path.join(run_dir, rel_a), os.path.join(run_dir, rel_b)
            ok, msg = check_identical_folds(dir_a, dir_b)
            if not ok:
                print(f"[warn] {rel_a}: {msg} - not a valid paired comparison, skipped")
                continue
            merged, n_train, n_test, stats = compare_one(
                os.path.join(dir_a, "nested_cv_folds.csv"), os.path.join(dir_b, "nested_cv_folds.csv"), cfg)
            if len(merged) < cfg["min_recommended_pairs"]:
                print(f"[warn] {rel_a}: only {len(merged)} paired fold scores")
            rows.append({
                "run": args.profile, "task": task, "hemi_mode": hemi_mode,
                "feature_a": fa, "feature_b": fb,
                "n_pairs": len(merged), "n_repeats": int(merged["repeat_id"].nunique()),
                "mean_n_train": n_train, "mean_n_test": n_test,
                "mean_a": float(merged[f"{col}_a"].mean()), "mean_b": float(merged[f"{col}_b"].mean()),
                "mean_difference": float(merged["difference"].mean()),
                "share_folds_b_better": float((merged["difference"] > 0).mean()),
                **stats,
            })
            merged = merged.rename(columns={f"{col}_a": f"{col}_{fa}", f"{col}_b": f"{col}_{fb}",
                                            "difference": f"difference_{fb}_minus_{fa}"})
            merged.insert(0, "hemi_mode", hemi_mode)
            merged.insert(0, "task", task)
            long_rows[fb].append(merged)

    if not rows:
        raise RuntimeError(f"No valid paired comparisons found in {run_dir}")

    res = pd.DataFrame(rows)
    p_col = {"greater": "p_greater", "less": "p_less", "two-sided": "p_two_sided"}[cfg["alternative"]]
    res["p_primary"] = res[p_col]
    res["p_holm"] = holm_adjust(res["p_primary"])
    res["p_bh"] = bh_adjust(res["p_primary"])
    res["significant_holm"] = res["p_holm"] < cfg["alpha"]
    res["holm_family_size"] = len(res)
    res = res.sort_values(["task", "feature_b", "hemi_mode"]).reset_index(drop=True)

    res.to_csv(out_dir / "feature_set_ttest_results.csv", index=False)
    for fb, frames in long_rows.items():
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(
                out_dir / f"feature_set_paired_fold_differences_{fb}.csv", index=False)
    with open(out_dir / "feature_set_ttest_config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    show = ["task", "feature_b", "n_pairs", "mean_a", "mean_b", "mean_difference",
            "ci95_low", "ci95_high", "t_stat", "df", "p_primary", "p_holm", "significant_holm"]
    with pd.option_context("display.width", 250, "display.max_columns", None, "display.float_format", "{:.4f}".format):
        print(f"\nCorrected repeated k-fold t-test, {cfg['feature_b']} - {fa}, H1: {cfg['alternative']}, "
              f"Holm over {len(res)} comparisons")
        print(res[show].to_string(index=False))
    print(f"\n[done] Results written to {out_dir}")


if __name__ == "__main__":
    main()
