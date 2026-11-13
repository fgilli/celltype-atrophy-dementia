"""
Lollipop plots of the CN-referenced cell-type scores.

One panel per group (diagnostic family or subtype) and one row per cell type
or composite. Points are subjects, the diamond is the group mean and the gray
band is +-1 SD of the CN group. Also writes the share of subjects within
+-1 CN SD per group and a small demographics table.

Needs the output of 06_cell_type_scores/compute_cell_type_scores.py.

Output: outputs/{profile}/group_profiles/plots and /tables
Usage:  python cell_type_profiles.py --profile tum      (tum | nifd | nacc)
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
OUTPUTS = REPO / "outputs"

PROFILES = {
    "tum": {
        "meta_path": DATA / "metadata" / "meta_TUM.csv",
        "scores_dir": OUTPUTS / "tum",
        "subtype_col": "Subtype",
    },
    "nifd": {
        "meta_path": DATA / "metadata" / "meta_NIFD.csv",
        "scores_dir": OUTPUTS / "nifd",
        "subtype_col": "Subtype",
    },
    "nacc": {
        "meta_path": DATA / "metadata" / "meta_NACC.csv",
        "scores_dir": OUTPUTS / "nacc",
        "subtype_col": None,
    },
}
SIDES = ["lh", "rh"]
FWHM = 5
COMPOSITE_COLS = ["glial_disease_z", "vascular_disease_z", "neuronal_disease_z"]
SEED = 42

DX_COLORS = {
    "CN": "#ff7f0e",
    "AD": "#1f77b4",
    "FTD": "#2ca02c",
    "MCI": "#bcbd22",
    "LBD": "#b3d627",
    "VAD": "#7f7f7f",
    "BVFTD": "#d62728",
    "SD": "#9467bd",
    "PCA": "#8c564b",
    "PNFA": "#e377c2",
    "LPA": "#17becf",
}
DX_COLOR_ALIASES = {
    "CON": "CN", "CONTROL": "CN", "HC": "CN", "NC": "CN",
    "BV": "BVFTD", "BV-FTD": "BVFTD",
    "SV": "SD", "SVPPA": "SD",
    "NFVPPA": "PNFA",
    "LVPPA": "LPA",
}
_FALLBACK_COLORS = ["#7f7f7f", "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173", "#3182bd", "#e6550d"]
_LABEL_UPPER = {"opc": "OPC", "cn": "CN", "ad": "AD", "ftd": "FTD"}


def dx_color(label):
    """Fixed color per diagnosis so a group looks the same in every cohort."""
    key = str(label).strip().upper()
    key = DX_COLOR_ALIASES.get(key, key)
    if key in DX_COLORS:
        return DX_COLORS[key]
    return _FALLBACK_COLORS[sum(ord(ch) for ch in key) % len(_FALLBACK_COLORS)]


def pretty_label(text):
    """'glial_disease' -> 'Glial disease', 'opc' -> 'OPC'."""
    words = [_LABEL_UPPER.get(w.lower(), w) for w in str(text).replace("_", " ").split()]
    out = " ".join(words)
    return out[:1].upper() + out[1:] if out else out


def cell_score_cols(table):
    return [c for c in table.columns if c.endswith("_cn_z")]


def score_label(col):
    return pretty_label(col.replace("_cn_z", "").replace("_z", ""))


def load_deviation_tables(cfg, ds):
    """{hemi: subject-level deviation table} with the subtype column attached via the kept rows."""
    meta = pd.read_csv(cfg["meta_path"])
    kept_path = Path(cfg["scores_dir"]) / "scores" / f"{ds}_kept_rows.npy"
    kept = np.load(kept_path) if kept_path.exists() else np.arange(len(meta))

    tables = {}
    for hemi in SIDES:
        path = Path(cfg["scores_dir"]) / "cn_reference" / f"fwhm{FWHM}" / hemi / f"{ds}_subject_level_deviation_scores.csv"
        if not path.exists():
            print(f"Missing {path} - skipped")
            continue
        table = pd.read_csv(path)
        if len(table) != len(kept):
            raise ValueError(f"{path}: {len(table)} rows, but {len(kept)} kept metadata rows")
        if cfg["subtype_col"]:
            table["subtype"] = meta[cfg["subtype_col"]].iloc[kept].to_numpy()
        tables[hemi] = table
    return tables


def plot_lollipop(table, score_cols, group_col, title_suffix, out_path):
    """Faceted lollipop chart (one panel per group, shared x-axis)."""
    table = table[table[group_col].notna()]
    labels = [score_label(c) for c in score_cols]
    groups = sorted(table[group_col].unique())
    n_groups, n_scores = len(groups), len(score_cols)

    fig, axes = plt.subplots(1, n_groups, figsize=(4.2 * n_groups, max(5, 0.55 * n_scores)),
                             sharey=True, sharex=True)
    axes = np.atleast_1d(axes)
    y_pos = np.arange(n_scores)

    all_vals = table[score_cols].dropna().values
    x_min = min(-2.5, all_vals.min() - 0.5) if all_vals.size else -2.5
    x_max = max(2.5, all_vals.max() + 0.5) if all_vals.size else 2.5

    for ax, grp_label in zip(axes, groups):
        grp = table[table[group_col] == grp_label]
        color = dx_color(grp_label)
        ax.axvspan(-1, 1, color="#bbbbbb", alpha=0.15, zorder=0)
        ax.axvline(0, color="#888888", lw=0.8, zorder=1)
        for col, y in zip(score_cols, y_pos):
            vals = grp[col].dropna().values
            jitter = np.random.uniform(-0.18, 0.18, size=len(vals))
            ax.scatter(vals, y + jitter, color=color, alpha=0.35, s=18, zorder=3, linewidths=0)
            mean_val = vals.mean() if len(vals) else 0
            ax.plot([0, mean_val], [y, y], color=color, lw=1.4, alpha=0.7, zorder=4)
            ax.scatter(mean_val, y, marker="D", color=color, s=55, zorder=5, edgecolors="white", linewidths=0.8)
        ax.set_xlim(x_min, x_max)
        ax.set_title(grp_label, fontsize=12, fontweight="bold", color=color, pad=6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlabel(pretty_label("z-score vs. CN"), fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", labelsize=9)

    cn_patch = mpatches.Patch(color="#bbbbbb", alpha=0.4, label="CN reference (±1 z)")
    fig.legend(handles=[cn_patch], loc="upper center", fontsize=9, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path} ({title_suffix})")


def in_band_stats(tables, ds, score_cols, group_col):
    """Share of subjects within +-1 CN SD per group and score."""
    group_header = "Subtype" if group_col != "dx" else "Diagnosis"
    score_header = "Cell Type" if score_cols is None else "Score"
    records = []
    for hemi, table in tables.items():
        if group_col not in table.columns:
            continue
        table = table[table[group_col].notna()]
        cols = cell_score_cols(table) if score_cols is None else [c for c in score_cols if c in table.columns]
        for grp_label in sorted(table[group_col].unique()):
            grp = table[table[group_col] == grp_label]
            for col in cols:
                vals = grp[col].dropna().values
                n = len(vals)
                count = int(np.sum((vals >= -1.0) & (vals <= 1.0))) if n else 0
                record = {"Dataset": ds, "Hemisphere": hemi.upper()}
                if group_col != "dx":
                    record["Diagnosis"] = ", ".join(sorted(grp["dx"].dropna().astype(str).unique()))
                record.update({
                    group_header: grp_label,
                    score_header: score_label(col),
                    "Total Patients": n,
                    "Count Within ±1z": count,
                    "Pct Within ±1z (%)": round(count / n * 100, 1) if n else 0.0,
                    "Mean z": round(float(vals.mean()), 3) if n else np.nan,
                })
                records.append(record)
    return pd.DataFrame(records)


def demographics(meta_path, kept):
    meta = pd.read_csv(meta_path)
    meta.columns = meta.columns.str.upper()
    meta = meta.iloc[kept]
    rows = []
    for dx, grp in meta.groupby("DX_FAMILY"):
        n = len(grp)
        age = pd.to_numeric(grp["AGE"], errors="coerce")
        sex = ", ".join(f"{s}: {c} ({c / n * 100:.1f}%)" for s, c in grp["SEX"].value_counts().items())
        rows.append({"DX": dx, "Total N": n,
                     "Age (Mean ± SD)": f"{age.mean():.1f} ± {age.std():.1f}" if age.notna().any() else "N/A",
                     "Sex Distribution": sex})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="CN-referenced cell-type profiles (lollipop plots).")
    parser.add_argument("--profile", choices=sorted(PROFILES), required=True)
    parser.add_argument("--output-dir", default=None, help="default: outputs/<profile>/group_profiles")
    args = parser.parse_args()

    np.random.seed(SEED)
    ds = args.profile
    cfg = PROFILES[ds]
    out_dir = Path(args.output_dir or OUTPUTS / ds / "group_profiles")
    plots_dir, tables_dir = out_dir / "plots", out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    tables = load_deviation_tables(cfg, ds)
    if not tables:
        raise FileNotFoundError("No deviation-score tables found - run compute_cell_type_scores.py first.")

    kept_path = Path(cfg["scores_dir"]) / "scores" / f"{ds}_kept_rows.npy"
    kept = np.load(kept_path) if kept_path.exists() else np.arange(len(pd.read_csv(cfg["meta_path"])))
    demo = demographics(cfg["meta_path"], kept)
    demo.to_csv(tables_dir / f"{ds}_demographics.csv", index=False)
    print(demo.to_string(index=False))

    groupings = [("dx", "family")] + ([("subtype", "subtype")] if cfg["subtype_col"] else [])
    for group_col, group_name in groupings:
        for hemi, table in tables.items():
            plot_lollipop(table, cell_score_cols(table), group_col, "cell types",
                          plots_dir / f"lollipop_cell_{group_name}_{ds}_{hemi}.png")
            plot_lollipop(table, COMPOSITE_COLS, group_col, "composites",
                          plots_dir / f"lollipop_composite_{group_name}_{ds}_{hemi}.png")
        for score_cols, score_name in [(None, "cell"), (COMPOSITE_COLS, "composite")]:
            stats = in_band_stats(tables, ds, score_cols, group_col)
            stats.to_csv(tables_dir / f"{ds}_in_band_stats_{score_name}_{group_name}.csv", index=False)

    print(f"\nPlots in {plots_dir}\nTables in {tables_dir}")


if __name__ == "__main__":
    main()
