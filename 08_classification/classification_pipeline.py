"""
Decision-tree classification with nested cross-validation.

Tasks:         healthy vs. sick (CN vs. all patients)
               FTD subtypes (bvFTD vs. SD)
Feature sets:  atrophy_regional   atrophy mass of the 8 DK region groups
               celltype           the 8 normalized cell-type scores
               combined           cell-type scores + total atrophy mass
               (lh and rh are averaged)

A single tree with class_weight="balanced" is tuned over max_depth {1, 2, 3}
and min_samples_leaf {3, 5, 7, 10} in an inner 5-fold CV. The outer loop is a
5-fold CV repeated 10 times (seed 42), so all feature sets are evaluated on
the same 50 folds. A final tree is fitted on all subjects, only for
interpretation (rules, plot, feature importances).

Profiles: tum and nifd leave out nfvPPA, nacc has no subtype task and
nifd_3class is the bvFTD vs. SD vs. nfvPPA task (only its subtype task is
used).

Needs the output of 06_cell_type_scores/compute_cell_type_scores.py.

Output: outputs/classification/{profile}/{task}_{feature_set}/average/
            nested_cv/   fold scores, out-of-fold predictions, confusion matrix
            tree rules, tree plot, feature importances, leaf report (final tree)

Usage:  python classification_pipeline.py --profile tum   (tum | nifd | nifd_3class | nacc)
"""

import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
OUTPUTS = REPO / "outputs"
CLASSIFICATION_OUT_ROOT = OUTPUTS / "classification"

# ============================================================
# CONFIG
# ============================================================
SHARED_CONFIG = {
    "hemis": ["lh", "rh"],
    "fwhm": 5,
    "feature_sets": {
        "celltype": {
            "array_template": "{ds}_{hemi}_scores_norm.fwhm{fwhm}.npy",
            "feature_def_path": "{data_dir}/feature_names.npy",
            "n_features": 8,
        },
        "atrophy_regional": {
            "array_template": "{ds}_{hemi}_atrophy_regional.fwhm{fwhm}.npy",
            "feature_def_path": "{data_dir}/atrophy_regional_names.npy",
            "n_features": 8,
        },
        "combined": {
            "array_template": "{ds}_{hemi}_scores_norm_all.fwhm{fwhm}.npy",
            "feature_def_path": "{data_dir}/feature_names_all.npy",
            "n_features": 9,
        },
    },
    "all_feature_sets": ["atrophy_regional", "celltype", "combined"],
    "hemi_modes": ["average"],
    "random_state": 42,
    "cv_folds": 5,
    "n_outer_folds": 5,
    "n_outer_repeats": 10,
    "max_depth_grid": [1, 2, 3],
    "min_samples_leaf_grid": [3, 5, 7, 10],
    "class_weight": "balanced",
}

PROFILES = {
    "tum": {
        "meta_path": DATA / "metadata" / "meta_TUM.csv",
        "data_dir": OUTPUTS / "tum" / "scores",
        "ds": "tum",
        "subject_id_col": "PTID",
        "dx_family_col": "DX_family",
        "dx_subtype_col": "Subtype",
        "excluded_subtypes": ["nfvPPA"],
        "ftd_expected_subtypes": ["bvFTD", "SD"],
        "out_dir": CLASSIFICATION_OUT_ROOT / "tum",
    },
    "nifd": {
        "meta_path": DATA / "metadata" / "meta_NIFD.csv",
        "data_dir": OUTPUTS / "nifd" / "scores",
        "ds": "nifd",
        "subject_id_col": "PTID",
        "dx_family_col": "DX_family",
        "dx_subtype_col": "Subtype",
        "excluded_subtypes": ["nfvPPA", "L_SD"],
        "ftd_expected_subtypes": ["bvFTD", "SD"],
        "out_dir": CLASSIFICATION_OUT_ROOT / "nifd",
    },
    "nifd_3class": {
        "meta_path": DATA / "metadata" / "meta_NIFD.csv",
        "data_dir": OUTPUTS / "nifd" / "scores",
        "ds": "nifd",
        "subject_id_col": "PTID",
        "dx_family_col": "DX_family",
        "dx_subtype_col": "Subtype",
        "excluded_subtypes": ["L_SD"],
        "ftd_expected_subtypes": ["bvFTD", "SD", "nfvPPA"],
        "out_dir": CLASSIFICATION_OUT_ROOT / "nifd_3class",
    },
    "nacc": {
        "meta_path": DATA / "metadata" / "meta_NACC.csv",
        "data_dir": OUTPUTS / "nacc" / "scores",
        "ds": "nacc",
        "subject_id_col": "PTID",
        "dx_family_col": "DX_family",
        "dx_subtype_col": "DX_family",
        "excluded_subtypes": [],
        "ftd_expected_subtypes": None,
        "out_dir": CLASSIFICATION_OUT_ROOT / "nacc",
    },
}

REQUIRED_PROFILE_KEYS = [
    "meta_path", "data_dir", "ds", "subject_id_col", "dx_family_col",
    "dx_subtype_col", "excluded_subtypes", "ftd_expected_subtypes", "out_dir",
]


def build_config(profile_name):
    """SHARED_CONFIG updated with PROFILES[profile_name]."""
    if profile_name not in PROFILES:
        raise ValueError(f"Unknown profile {profile_name!r}; choose from {sorted(PROFILES)}")
    profile = PROFILES[profile_name]
    missing = [k for k in REQUIRED_PROFILE_KEYS if k not in profile]
    if missing:
        raise ValueError(f"Profile {profile_name!r} is missing keys: {missing}")
    cfg = copy.deepcopy(SHARED_CONFIG)
    cfg.update(copy.deepcopy(profile))
    cfg["meta_path"], cfg["data_dir"], cfg["out_dir"] = (str(cfg[k]) for k in ("meta_path", "data_dir", "out_dir"))
    cfg["profile"] = profile_name
    return cfg


# ============================================================
# Loading
# ============================================================
def load_meta(cfg):
    """
    Metadata aligned with the score arrays, with excluded subtypes removed.

    If compute_cell_type_scores.py dropped subjects ({ds}_kept_rows.npy),
    only the kept metadata rows are used. The row index of the returned
    table is the row index into the score arrays (no reset_index after the
    subtype exclusion, build_features relies on it).
    """
    meta = pd.read_csv(cfg["meta_path"])
    kept_path = os.path.join(cfg["data_dir"], f"{cfg['ds']}_kept_rows.npy")
    if os.path.exists(kept_path):
        meta = meta.iloc[np.load(kept_path)].reset_index(drop=True)

    required = [cfg["subject_id_col"], cfg["dx_family_col"], cfg["dx_subtype_col"]]
    missing = [c for c in required if c not in meta.columns]
    if missing:
        raise ValueError(f"metadata is missing expected columns: {missing}")

    excluded = cfg.get("excluded_subtypes") or []
    if excluded:
        mask = meta[cfg["dx_subtype_col"]].isin(excluded)
        if mask.any():
            print(f"[info] Excluding {int(mask.sum())} subject(s) with subtype in {excluded} from all tasks")
            meta = meta.loc[~mask].copy()
    return meta


def load_feature_names(feature_set_cfg, cfg):
    """Feature names of a feature set (generic names if the definition file is missing)."""
    n = feature_set_cfg["n_features"]
    fallback = [f"feature_{i}" for i in range(n)]
    path = feature_set_cfg["feature_def_path"].format(data_dir=cfg["data_dir"], ds=cfg["ds"], fwhm=cfg["fwhm"])
    if not os.path.exists(path):
        print(f"[warn] feature names not found ({path}); using generic names")
        return fallback
    names = [str(x) for x in np.asarray(np.load(path, allow_pickle=True)).reshape(-1)]
    if len(names) != n:
        print(f"[warn] {path} has {len(names)} names, expected {n}; using generic names")
        return fallback
    return names


def load_hemi_array(cfg, feature_set_cfg, hemi):
    path = os.path.join(cfg["data_dir"],
                        feature_set_cfg["array_template"].format(hemi=hemi, fwhm=cfg["fwhm"], ds=cfg["ds"]))
    if not os.path.exists(path):
        raise FileNotFoundError(f"Expected score array not found: {path}")
    arr = np.load(path)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.shape[1] != feature_set_cfg["n_features"]:
        raise ValueError(f"{path} has {arr.shape[1]} columns, expected {feature_set_cfg['n_features']}")
    return arr


# ============================================================
# Tasks
# ============================================================
def prepare_healthy_vs_sick(meta, cfg):
    """CN vs. every other diagnostic family."""
    task_meta = meta.copy()
    task_meta["label"] = np.where(task_meta[cfg["dx_family_col"]] == "CN", "CN", "sick")
    print(f"[info] Task 'healthy_vs_sick': {len(task_meta)} subjects")
    print(task_meta["label"].value_counts().to_string())
    return task_meta


def prepare_ftd_subtype_classification(meta, cfg):
    """FTD patients only, labeled by subtype (subtypes not in ftd_expected_subtypes are dropped)."""
    ftd_meta = meta.loc[meta[cfg["dx_family_col"]] == "FTD"].copy()
    unexpected = sorted(set(ftd_meta[cfg["dx_subtype_col"]].unique()) - set(cfg["ftd_expected_subtypes"]))
    if unexpected:
        print(f"[warn] Dropping FTD subtypes not in ftd_expected_subtypes: {unexpected}")
        ftd_meta = ftd_meta[~ftd_meta[cfg["dx_subtype_col"]].isin(unexpected)].copy()
    ftd_meta["label"] = ftd_meta[cfg["dx_subtype_col"]]
    print(f"[info] Task 'ftd_subtype_classification': {len(ftd_meta)} subjects")
    print(ftd_meta["label"].value_counts().to_string())
    return ftd_meta


def build_features(cfg, task_meta, feature_names, feature_set_cfg, hemi_mode):
    """(X, y, feature names, subject ids) for one task, feature set and hemisphere mode."""
    idx = task_meta.index.to_numpy()
    hemi_arrays = {}
    for hemi in cfg["hemis"]:
        arr = load_hemi_array(cfg, feature_set_cfg, hemi)
        if arr.shape[0] <= idx.max():
            raise ValueError(f"{hemi} array has {arr.shape[0]} rows, but the metadata index goes up to {idx.max()}")
        hemi_arrays[hemi] = arr[idx]

    if hemi_mode == "concat":
        X = np.concatenate([hemi_arrays[h] for h in cfg["hemis"]], axis=1)
        out_names = [f"{h}_{name}" for h in cfg["hemis"] for name in feature_names]
    elif hemi_mode == "average":
        X = np.mean(np.stack([hemi_arrays[h] for h in cfg["hemis"]], axis=0), axis=0)
        out_names = list(feature_names)
    else:
        raise ValueError(f"Unknown hemi_mode: {hemi_mode}")
    return X, task_meta["label"].to_numpy(), out_names, task_meta[cfg["subject_id_col"]].to_numpy()


# ============================================================
# Fitting and nested cross-validation
# ============================================================
def fit_tree(X_train, y_train, cfg):
    """Tune max_depth / min_samples_leaf with GridSearchCV; returns (tree, params, inner CV score)."""
    base = DecisionTreeClassifier(class_weight=cfg["class_weight"], random_state=cfg["random_state"])
    param_grid = {"max_depth": cfg["max_depth_grid"], "min_samples_leaf": cfg["min_samples_leaf_grid"]}
    cv = StratifiedKFold(n_splits=cfg["cv_folds"], shuffle=True, random_state=cfg["random_state"])
    grid = GridSearchCV(base, param_grid, cv=cv, scoring="balanced_accuracy", n_jobs=-1)
    grid.fit(X_train, y_train)
    print(f"[info] Best params: {grid.best_params_} (CV balanced accuracy: {grid.best_score_:.3f})")
    return grid.best_estimator_, grid.best_params_, float(grid.best_score_)


def iter_outer_splits(X, y, cfg):
    """Yield (repeat_id, fold_id, train_idx, test_idx) of the repeated stratified outer CV."""
    n_splits = cfg["n_outer_folds"]
    outer_cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=int(cfg.get("n_outer_repeats", 1)),
                                       random_state=cfg["random_state"])
    for split_id, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        yield split_id // n_splits, split_id % n_splits, train_idx, test_idx


def finalize_nested_cv(fold_rows, oof_rows, out_dir, label_desc):
    """Write the nested-CV bundle and return its summary."""
    os.makedirs(out_dir, exist_ok=True)
    folds_df = pd.DataFrame(fold_rows)
    folds_df.to_csv(os.path.join(out_dir, "nested_cv_folds.csv"), index=False)

    oof_df = pd.DataFrame(oof_rows).sort_values(["repeat_id", "subject_id"]).reset_index(drop=True)
    oof_df.to_csv(os.path.join(out_dir, "oof_predictions.csv"), index=False)

    per_repeat_pooled = {
        int(r): float(balanced_accuracy_score(g["true_label"], g["predicted_label"]))
        for r, g in oof_df.groupby("repeat_id")
    }

    oof_rep0 = oof_df[oof_df["repeat_id"] == oof_df["repeat_id"].min()]
    y_true, y_pred = oof_rep0["true_label"].to_numpy(), oof_rep0["predicted_label"].to_numpy()
    labels_sorted = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels_sorted)
    pd.DataFrame(cm, index=labels_sorted, columns=labels_sorted).to_csv(os.path.join(out_dir, "confusion_matrix.csv"))
    with open(os.path.join(out_dir, "classification_report.json"), "w") as f:
        json.dump(classification_report(y_true, y_pred, output_dict=True, zero_division=0), f, indent=2)

    scores = folds_df["held_out_balanced_accuracy"].dropna().to_numpy()
    mean_bal_acc = float(np.mean(scores)) if len(scores) else None
    std_bal_acc = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
    pooled = float(np.mean(list(per_repeat_pooled.values())))
    n_repeats = int(folds_df["repeat_id"].nunique())
    summary = {
        "label_desc": label_desc,
        "n_outer_folds": int(len(folds_df)),
        "n_outer_folds_per_repeat": int(folds_df["fold_id"].nunique()),
        "n_outer_repeats": n_repeats,
        "n_samples_total": int(oof_rep0["subject_id"].nunique()),
        "mean_n_train": float(folds_df["n_train"].mean()),
        "mean_n_test": float(folds_df["n_test"].mean()),
        "per_fold_balanced_accuracy": [float(s) for s in scores],
        "mean_balanced_accuracy": mean_bal_acc,
        "std_balanced_accuracy": std_bal_acc,
        "min_balanced_accuracy": float(np.min(scores)) if len(scores) else None,
        "max_balanced_accuracy": float(np.max(scores)) if len(scores) else None,
        "pooled_oof_balanced_accuracy": pooled,
        "pooled_oof_balanced_accuracy_per_repeat": per_repeat_pooled,
        "confusion_matrix_and_report_from_repeat": int(oof_df["repeat_id"].min()),
        "classes": labels_sorted,
    }
    with open(os.path.join(out_dir, "nested_cv_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[nested-cv] {label_desc}: balanced accuracy over {summary['n_outer_folds']} outer folds "
          f"({n_repeats} repeats) = {mean_bal_acc:.3f} +/- {std_bal_acc:.3f} (pooled OOF {pooled:.3f})")
    return summary


def run_nested_cv_single_tree(X, y, subject_ids, cfg, out_dir, label_desc="target"):
    """Nested CV: hyperparameters are tuned inside every outer training fold."""
    X, y, subject_ids = np.asarray(X), np.asarray(y), np.asarray(subject_ids)
    fold_rows, oof_rows = [], []
    for repeat_id, fold_id, train_idx, test_idx in iter_outer_splits(X, y, cfg):
        tree, best_params, inner_score = fit_tree(X[train_idx], y[train_idx], cfg)
        y_pred = tree.predict(X[test_idx])
        y_te = y[test_idx]
        fold_bal_acc = balanced_accuracy_score(y_te, y_pred)
        print(f"[nested-cv] repeat {repeat_id + 1}, fold {fold_id + 1}/{cfg['n_outer_folds']} ({label_desc}): "
              f"n_train={len(train_idx)}, n_test={len(test_idx)}, balanced accuracy={fold_bal_acc:.3f}")
        fold_rows.append({
            "repeat_id": repeat_id,
            "fold_id": fold_id,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "best_params": json.dumps(best_params),
            "inner_cv_balanced_accuracy": float(inner_score),
            "held_out_balanced_accuracy": float(fold_bal_acc),
        })
        for sid, yt, yp in zip(subject_ids[test_idx], y_te, y_pred):
            oof_rows.append({"subject_id": sid, "repeat_id": repeat_id, "fold_id": fold_id,
                             "true_label": yt, "predicted_label": yp, "correct": yt == yp})
    return finalize_nested_cv(fold_rows, oof_rows, out_dir, label_desc)


# ============================================================
# Final tree: node statistics, rules and plot
# ============================================================
def compute_node_stats(tree, X, y):
    """Raw (unweighted) class counts and number of subjects per node."""
    node_indicator = tree.decision_path(X).toarray()
    classes = list(tree.classes_)
    value = np.zeros((tree.tree_.node_count, 1, len(classes)))
    n_node_samples = np.zeros(tree.tree_.node_count, dtype=np.int64)
    for node_id in range(tree.tree_.node_count):
        mask = node_indicator[:, node_id].astype(bool)
        n_node_samples[node_id] = int(mask.sum())
        for ci, c in enumerate(classes):
            value[node_id, 0, ci] = int(np.sum(y[mask] == c))
    return value, n_node_samples


def compute_frozen_labels(tree):
    """Predicted (class-weighted) label of every node, identical to tree.predict()."""
    classes = list(tree.classes_)
    return {node: classes[int(np.argmax(tree.tree_.value[node, 0, :]))] for node in range(tree.tree_.node_count)}


def compute_label_confidence(value, n_node_samples, frozen_labels, classes):
    """{node: (fraction of subjects matching the node label, n)}; fraction None if n = 0."""
    class_to_idx = {c: i for i, c in enumerate(classes)}
    result = {}
    for node in range(value.shape[0]):
        n = int(n_node_samples[node])
        if n == 0:
            result[node] = (None, 0)
        else:
            result[node] = (float(value[node, 0, class_to_idx[frozen_labels[node]]]) / n, n)
    return result


def export_text_with_node_stats(tree, feature_names, value, n_node_samples, frozen_labels, confidence, header):
    tree_ = tree.tree_
    classes = list(tree.classes_)
    lines = [header, ""]

    def recurse(node, depth):
        indent = "|   " * depth
        counts = {c: int(value[node, 0, i]) for i, c in enumerate(classes)}
        n = int(n_node_samples[node])
        conf, _ = confidence[node]
        conf_str = f"{conf:.0%}" if conf is not None else "n/a"
        if tree_.feature[node] != -2:
            fname, thresh = feature_names[tree_.feature[node]], tree_.threshold[node]
            lines.append(f"{indent}[node {node}] n={n} {counts} (label={frozen_labels[node]}, agreement={conf_str})")
            lines.append(f"{indent}|--- {fname} <= {thresh:.4f}")
            recurse(tree_.children_left[node], depth + 1)
            lines.append(f"{indent}|--- {fname} >  {thresh:.4f}")
            recurse(tree_.children_right[node], depth + 1)
        else:
            lines.append(f"{indent}[leaf {node}] n={n} {counts} -> predict {frozen_labels[node]} (agreement={conf_str})")

    recurse(0, 0)
    return "\n".join(lines)


def tree_layout(tree_, leaf_spacing=1.0):
    """(x, y) position of every node: leaves left to right, parents centered above their children."""
    positions = {}
    next_leaf_x = [0.0]

    def recurse(node, depth):
        if tree_.feature[node] == -2:
            x = next_leaf_x[0]
            next_leaf_x[0] += leaf_spacing
        else:
            x = (recurse(tree_.children_left[node], depth + 1) + recurse(tree_.children_right[node], depth + 1)) / 2.0
        positions[node] = (x, -depth)
        return x

    recurse(0, 0)
    return positions


def render_tree(tree, feature_names, value, n_node_samples, frozen_labels, confidence, fig_path, title=""):
    """Matplotlib rendering of the tree with raw subject counts (reads only the tree structure arrays)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyBboxPatch

    tree_ = tree.tree_
    classes = list(tree.classes_)
    class_colors = dict(zip(classes, plt.cm.Set2.colors[:len(classes)]))
    positions = tree_layout(tree_)
    n_leaves = sum(1 for node in range(tree_.node_count) if tree_.feature[node] == -2)
    max_depth = max(-y for _, y in positions.values())

    box_w, box_h = 0.9, 0.62
    fig, ax = plt.subplots(figsize=(max(10, n_leaves * 1.9), max(6, (max_depth + 1) * 2.4)))

    for node in range(tree_.node_count):
        if tree_.feature[node] == -2:
            continue
        x0, y0 = positions[node]
        for child, label in ((tree_.children_left[node], "True"), (tree_.children_right[node], "False")):
            x1, y1 = positions[child]
            ax.add_line(Line2D([x0, x1], [y0 - box_h / 2, y1 + box_h / 2], color="gray", linewidth=1, zorder=1))
            ax.text((x0 + x1) / 2, (y0 - box_h / 2 + y1 + box_h / 2) / 2, label, fontsize=8, color="dimgray",
                    ha="center", va="center", backgroundcolor="white", zorder=2)

    for node in range(tree_.node_count):
        x, y = positions[node]
        n = int(n_node_samples[node])
        label = frozen_labels[node]
        conf, _ = confidence[node]
        lines = []
        if tree_.feature[node] != -2:
            lines += [f"{feature_names[tree_.feature[node]]}", f"≤ {tree_.threshold[node]:.4f}"]
        lines.append(f"n = {n}")
        lines.append(", ".join(f"{c}:{int(v)}" for c, v in zip(classes, value[node, 0, :])))
        if n > 0:
            lines.append(f"→ {label} ({conf:.0%})" if conf is not None else f"→ {label}")
        else:
            lines.append("(no subjects here)")
        box = FancyBboxPatch((x - box_w / 2, y - box_h / 2), box_w, box_h,
                             boxstyle="round,pad=0.02,rounding_size=0.05", linewidth=1,
                             edgecolor="black" if n > 0 else "lightgray",
                             facecolor=class_colors.get(label, "lightgray") if n > 0 else "whitesmoke", zorder=3)
        ax.add_patch(box)
        ax.text(x, y, "\n".join(lines), fontsize=7.5, ha="center", va="center", zorder=4)

    ax.set_xlim(-1, max(x for x, _ in positions.values()) + 1)
    ax.set_ylim(-max_depth - 1, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, fontsize=15)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[info] Tree plot written to {fig_path}")


def save_tree_artifacts(tree, feature_names, X, y, out_dir, cfg, task_name, feature_set_name):
    """Feature importances, rules, plot and leaf report of the final (all-subject) tree."""
    os.makedirs(out_dir, exist_ok=True)
    importances = tree.feature_importances_
    order = np.argsort(importances)[::-1]
    fi = pd.DataFrame({"feature": [feature_names[i] for i in order], "importance": importances[order]})
    fi.to_csv(os.path.join(out_dir, "feature_importances.csv"), index=False)
    print(fi.to_string(index=False))

    classes = list(tree.classes_)
    frozen_labels = compute_frozen_labels(tree)
    value, n_node_samples = compute_node_stats(tree, X, y)
    confidence = compute_label_confidence(value, n_node_samples, frozen_labels, classes)

    prefix = f"{cfg['profile']}_{feature_set_name}"
    with open(os.path.join(out_dir, f"{prefix}_tree_rules_{task_name}.txt"), "w") as f:
        f.write(export_text_with_node_stats(
            tree, feature_names, value, n_node_samples, frozen_labels, confidence,
            header="Final tree fitted on all subjects; raw subject counts per node. 'label' is the "
                   "class-weighted prediction (tree.predict); 'agreement' is the share of subjects "
                   "at the node that have this label."))

    # With class_weight="balanced" a leaf can predict a class that is not its
    # raw-count majority; label_disagreement marks these leaves.
    classes_arr = np.array(classes)
    leaf_rows = []
    for node in range(tree.tree_.node_count):
        if tree.tree_.feature[node] != -2:
            continue
        conf, n = confidence[node]
        raw_majority = classes_arr[int(np.argmax(value[node, 0, :]))] if n > 0 else None
        leaf_rows.append({
            "node_id": node,
            "frozen_predicted_label": frozen_labels[node],
            "raw_majority_label": raw_majority,
            "label_disagreement": raw_majority is not None and raw_majority != frozen_labels[node],
            "n": n,
            "confidence": conf,
        })
    leaf_df = pd.DataFrame(leaf_rows).sort_values("node_id")
    leaf_df.to_csv(os.path.join(out_dir, "leaf_confidence_report.csv"), index=False)
    print(leaf_df.to_string(index=False))

    try:
        render_tree(tree, feature_names, value, n_node_samples, frozen_labels, confidence,
                    fig_path=os.path.join(out_dir, f"{prefix}_tree_plot_{task_name}.png"))
    except Exception as e:
        print(f"[warn] Tree plot skipped: {e}")


def save_leaf_assignments(tree, X, y, subject_ids, out_dir):
    df = pd.DataFrame({
        "subject_id": subject_ids,
        "leaf_id": tree.apply(X),
        "true_label": y,
        "predicted_label": tree.predict(X),
    })
    df["correct"] = df["true_label"] == df["predicted_label"]
    df.sort_values(["leaf_id", "subject_id"]).to_csv(os.path.join(out_dir, "leaf_assignments_full.csv"), index=False)


# ============================================================
# Runs
# ============================================================
def run_single_config(cfg, task_meta, feature_set_name, hemi_mode, task_out_dir, task_name):
    """Nested CV and final tree for one task, feature set and hemisphere mode."""
    feature_set_cfg = cfg["feature_sets"][feature_set_name]
    feature_names = load_feature_names(feature_set_cfg, cfg)
    print(f"\n{'-' * 60}\n[info] {task_name}: feature_set={feature_set_name}, hemi_mode={hemi_mode}\n{'-' * 60}")
    out_dir = os.path.join(task_out_dir, hemi_mode)
    os.makedirs(out_dir, exist_ok=True)

    X, y, out_names, subject_ids = build_features(cfg, task_meta, feature_names, feature_set_cfg, hemi_mode)
    nested = run_nested_cv_single_tree(X, y, subject_ids, cfg, os.path.join(out_dir, "nested_cv"),
                                       label_desc=f"{feature_set_name}/{hemi_mode}")

    final_tree, best_params, best_cv = fit_tree(X, y, cfg)
    save_tree_artifacts(final_tree, out_names, X, y, out_dir, cfg, task_name, feature_set_name)
    save_leaf_assignments(final_tree, X, y, subject_ids, out_dir)

    summary = {
        "hemi_mode": hemi_mode,
        "feature_set": feature_set_name,
        "n_total": int(X.shape[0]),
        "fwhm": cfg["fwhm"],
        "nested_cv": nested,
        "final_model": {"best_params": best_params, "best_cv_balanced_accuracy": best_cv},
        "best_params": best_params,
        "best_cv_balanced_accuracy": best_cv,
        "held_out_balanced_accuracy": nested["mean_balanced_accuracy"],
        "held_out_balanced_accuracy_std": nested["std_balanced_accuracy"],
        "classes": sorted(set(y)),
    }
    with open(os.path.join(out_dir, "run_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def compute_balanced_accuracy_deltas(cfg, per_feature_set):
    """Mean nested-CV balanced accuracy per feature set and the differences to atrophy_regional."""
    acc = {fs: {s["hemi_mode"]: s["held_out_balanced_accuracy"] for s in o["per_hemi_mode"]}
           for fs, o in per_feature_set.items()}
    deltas = {}
    for hemi_mode in cfg["hemi_modes"]:
        a = acc.get("atrophy_regional", {}).get(hemi_mode)
        c = acc.get("celltype", {}).get(hemi_mode)
        m = acc.get("combined", {}).get(hemi_mode)
        deltas[hemi_mode] = {
            "atrophy_regional_balanced_accuracy": a,
            "celltype_balanced_accuracy": c,
            "combined_balanced_accuracy": m,
            "atrophy_regional_minus_celltype_balanced_accuracy": a - c if a is not None and c is not None else None,
            "atrophy_regional_minus_combined_balanced_accuracy": a - m if a is not None and m is not None else None,
        }
    return deltas


def run_task(cfg, task_name, task_meta, feature_set_names, task_out_dir_name):
    if task_meta["label"].nunique() < 2:
        raise ValueError(f"Task '{task_name}': fewer than 2 classes - nothing to classify.")
    per_feature_set = {}
    for fs in feature_set_names:
        task_out_dir = os.path.join(cfg["out_dir"], f"{task_out_dir_name}_{fs}")
        os.makedirs(task_out_dir, exist_ok=True)
        summaries = [run_single_config(cfg, task_meta, fs, hm, task_out_dir, task_name) for hm in cfg["hemi_modes"]]
        overall = {
            "task": task_name,
            "feature_set": fs,
            "n_subjects_total": int(len(task_meta)),
            "label_counts": task_meta["label"].value_counts().to_dict(),
            "per_hemi_mode": summaries,
        }
        with open(os.path.join(task_out_dir, "run_summary_all_modes.json"), "w") as f:
            json.dump(overall, f, indent=2)
        per_feature_set[fs] = overall

    deltas = compute_balanced_accuracy_deltas(cfg, per_feature_set)
    with open(os.path.join(cfg["out_dir"], f"{task_out_dir_name}_deltas.json"), "w") as f:
        json.dump({"task": task_name, "per_hemi_mode_deltas": deltas}, f, indent=2)
    return {"per_feature_set": per_feature_set, "deltas": deltas}


def main(cfg):
    print(f"[info] Profile {cfg['profile']!r} -> {cfg['out_dir']}")
    os.makedirs(cfg["out_dir"], exist_ok=True)
    with open(os.path.join(cfg["out_dir"], "config_used.json"), "w") as f:
        json.dump(cfg, f, indent=2, default=str)

    meta = load_meta(cfg)
    feature_sets = cfg["all_feature_sets"]
    results = {
        "01_healthy_vs_sick": run_task(cfg, "healthy_vs_sick", prepare_healthy_vs_sick(meta, cfg),
                                       feature_sets, "01_healthy_vs_sick"),
    }
    if cfg.get("ftd_expected_subtypes"):
        results["02_ftd_subtype_classification"] = run_task(
            cfg, "ftd_subtype_classification", prepare_ftd_subtype_classification(meta, cfg),
            feature_sets, "02_ftd_subtype_classification")
    else:
        print(f"[info] Profile {cfg['profile']!r} has no FTD subtype labels - subtype task skipped.")

    with open(os.path.join(cfg["out_dir"], "run_summary_all_tasks.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[done] All tasks complete -> {cfg['out_dir']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decision-tree classification with nested cross-validation.")
    parser.add_argument("--profile", required=True, choices=sorted(PROFILES))
    parser.add_argument("--n-outer-repeats", type=int, default=None, help="override (default 10)")
    args = parser.parse_args()
    config = build_config(args.profile)
    if args.n_outer_repeats is not None:
        config["n_outer_repeats"] = args.n_outer_repeats
    main(config)
