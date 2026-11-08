"""
Heterogeneity of regional atrophy patterns (barcode plots).

Adapts the spatial-heterogeneity analysis of Thalhammer et al., Nat Commun
2025 (https://doi.org/10.1038/s41467-025-63967-1; code:
https://github.com/Melissa1909/preterm-brain-heterogeneity, file
code/spatial_heterogeneity.py) to the regional atrophy masses of the 34
Desikan-Killiany parcels (hemispheres averaged).

Per cohort:
  1. every parcel mass is converted into an empirical centile relative to
     the CN group of the same cohort (to_centile_scores),
  2. values below the 5th or above the 95th CN centile are coded as
     extranormal (1), all others as 0 (binarize_extranormal),
  3. per diagnostic family and per clinical subtype, the binary
     subject x parcel matrix is plotted (barcode), the subject x subject
     Spearman correlation matrix is computed, and its row means are
     summarized (with and without the self-correlation).

Functions marked "Thalhammer et al." are taken from the original code.

Outputs (CONFIG["output_dir"])
    {cohort}_{grouping}_{group}_binarized.png / _corrmatrix.png
    heterogeneity_summary.csv     mean/median within-group Spearman rho,
                                  % of subjects with any extranormal parcel
    parcel_prevalence.csv         % of subjects with an extranormal value per parcel
    thesis_fig6_barcodes_main.pdf, thesis_supp_barcodes.pdf, thesis_supp_corr.pdf

Usage
    python atrophy_heterogeneity.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path

import seaborn as sns
from matplotlib.colors import ListedColormap

sns.set_style("white")

REPO = Path(__file__).resolve().parents[1]

CONFIG = {
    "outputs_root": REPO / "outputs",
    "hemi": ["lh", "rh"],   # averaged per parcel; a single string uses one hemisphere
    "fwhm": 5,
    "datasets": {
        "TUM": {
            "ds_key": "tum",
            "meta_csv": REPO / "data" / "metadata" / "meta_TUM.csv",
            "id_col": "PTID",
            "dx_col": "DX_family",
            "subtype_col": "Subtype",
        },
        "NIFD": {
            "ds_key": "nifd",
            "meta_csv": REPO / "data" / "metadata" / "meta_NIFD.csv",
            "id_col": "PTID",
            "dx_col": "DX_family",
            "subtype_col": "Subtype",
        },
        "NACC": {
            "ds_key": "nacc",
            "meta_csv": REPO / "data" / "metadata" / "meta_NACC.csv",
            "id_col": "PTID",
            "dx_col": "DX_family",
            "subtype_col": None,
        },
    },
    "grouping_vars": ["dx_family", "subtype"],
    "reference_group": "CN",
    "min_group_size": 5,
    "output_dir": REPO / "outputs" / "heterogeneity",
    "make_thesis_figures": True,
    # thesis figures: (cohort, grouping_var, group, panel title)
    "fig6_barcodes": [
        ("TUM", "subtype", "CN", "TUM\nCN"),
        ("TUM", "subtype", "bvFTD", "TUM\nbvFTD"),
        ("TUM", "subtype", "SD", "TUM\nSD"),
        ("NIFD", "subtype", "bvFTD", "NIFD\nbvFTD"),
        ("NIFD", "subtype", "SD", "NIFD\nSD"),
    ],
    "fig6_prevalence": [
        ("TUM", "subtype", "AD", "TUM typ. AD"),
        ("TUM", "subtype", "PCA", "TUM PCA"),
        ("TUM", "subtype", "bvFTD", "TUM bvFTD"),
        ("TUM", "subtype", "SD", "TUM SD"),
        ("TUM", "subtype", "nfvPPA", "TUM nfvPPA"),
        ("NIFD", "subtype", "bvFTD", "NIFD bvFTD"),
        ("NIFD", "subtype", "SD", "NIFD SD"),
        ("NIFD", "subtype", "nfvPPA", "NIFD nfvPPA"),
        ("NACC", "dx_family", "FTD", "NACC FTD"),
    ],
    "supp_barcodes": [
        ("TUM", "dx_family", "CN", "TUM\nCN"),
        ("TUM", "dx_family", "AD", "TUM\nAD"),
        ("TUM", "dx_family", "FTD", "TUM\nFTD"),
        ("TUM", "subtype", "AD", "TUM\nAD (typical)"),
        ("TUM", "subtype", "PCA", "TUM\nPCA"),
        ("TUM", "subtype", "nfvPPA", "TUM\nnfvPPA"),
        ("NIFD", "dx_family", "CN", "NIFD\nCN"),
        ("NIFD", "dx_family", "FTD", "NIFD\nFTD"),
        ("NIFD", "subtype", "nfvPPA", "NIFD\nnfvPPA"),
        ("NACC", "dx_family", "FTD", "NACC\nFTD"),
        ("NACC", "dx_family", "LBD", "NACC\nLBD"),
        ("NACC", "dx_family", "VAD", "NACC\nVAD"),
    ],
    "supp_corr": [
        ("TUM", "subtype", "bvFTD", "TUM bvFTD"),
        ("TUM", "subtype", "SD", "TUM SD"),
        ("NIFD", "subtype", "bvFTD", "NIFD bvFTD"),
        ("NIFD", "subtype", "SD", "NIFD SD"),
    ],
}


# Desikan-Killiany parcels grouped into the 8 regions (display order of the barcodes)
REGIONS = [
    ("Inferior frontal", ["lateralorbitofrontal", "medialorbitofrontal", "parsorbitalis",
                          "parstriangularis", "parsopercularis", "frontalpole"]),
    ("Ventral frontal", ["superiorfrontal", "rostralmiddlefrontal", "caudalmiddlefrontal"]),
    ("Central", ["precentral", "postcentral", "paracentral"]),
    ("Limbic", ["rostralanteriorcingulate", "caudalanteriorcingulate", "posteriorcingulate",
                "isthmuscingulate", "insula"]),
    ("Superior temporal", ["superiortemporal", "middletemporal", "transversetemporal", "bankssts"]),
    ("Inf./medial temporal", ["temporalpole", "entorhinal", "parahippocampal", "fusiform",
                              "inferiortemporal"]),
    ("Parietal", ["superiorparietal", "inferiorparietal", "supramarginal", "precuneus"]),
    ("Occipital", ["lateraloccipital", "lingual", "cuneus", "pericalcarine"]),
]
PRETTY = {
    "lateralorbitofrontal": "lateral orbitofrontal", "medialorbitofrontal": "medial orbitofrontal",
    "parsorbitalis": "pars orbitalis", "parstriangularis": "pars triangularis",
    "parsopercularis": "pars opercularis", "frontalpole": "frontal pole",
    "superiorfrontal": "superior frontal", "rostralmiddlefrontal": "rostral middle frontal",
    "caudalmiddlefrontal": "caudal middle frontal", "precentral": "precentral",
    "postcentral": "postcentral", "paracentral": "paracentral",
    "rostralanteriorcingulate": "rostral ant. cingulate",
    "caudalanteriorcingulate": "caudal ant. cingulate", "posteriorcingulate": "posterior cingulate",
    "isthmuscingulate": "isthmus cingulate", "insula": "insula",
    "superiortemporal": "superior temporal", "middletemporal": "middle temporal",
    "transversetemporal": "transverse temporal", "bankssts": "banks STS",
    "temporalpole": "temporal pole", "entorhinal": "entorhinal",
    "parahippocampal": "parahippocampal", "fusiform": "fusiform",
    "inferiortemporal": "inferior temporal", "superiorparietal": "superior parietal",
    "inferiorparietal": "inferior parietal", "supramarginal": "supramarginal",
    "precuneus": "precuneus", "lateraloccipital": "lateral occipital", "lingual": "lingual",
    "cuneus": "cuneus", "pericalcarine": "pericalcarine",
}
PARCEL_ORDER = [p for _, ps in REGIONS for p in ps]
REGION_BOUNDS = np.cumsum([len(ps) for _, ps in REGIONS])[:-1]


def load_dataset(name, cfg):
    """
    Parcel atrophy masses (hemispheres averaged) plus diagnosis/subtype of
    one cohort. Metadata rows are aligned via {ds}_kept_rows.npy written by
    compute_cell_type_scores.py.
    """
    ds = cfg["datasets"][name]
    scores_dir = Path(cfg["outputs_root"]) / ds["ds_key"] / "scores"
    hemis = cfg["hemi"] if isinstance(cfg["hemi"], (list, tuple)) else [cfg["hemi"]]
    arrays = [np.load(scores_dir / f"{ds['ds_key']}_{h}_atrophy_dk.fwhm{cfg['fwhm']}.npy") for h in hemis]
    if any(a.shape != arrays[0].shape for a in arrays):
        raise ValueError(f"[{name}] hemisphere arrays differ in shape")
    scores = np.mean(arrays, axis=0)
    region_names = [str(r) for r in np.load(scores_dir / "atrophy_dk_names.npy", allow_pickle=True)]

    meta = pd.read_csv(ds["meta_csv"])
    kept_path = scores_dir / f"{ds['ds_key']}_kept_rows.npy"
    if kept_path.exists():
        meta = meta.iloc[np.load(kept_path)].reset_index(drop=True)
    if len(meta) != scores.shape[0]:
        raise ValueError(f"[{name}] metadata has {len(meta)} rows but the scores have {scores.shape[0]}")

    df = pd.DataFrame(scores, columns=region_names)
    df["dx_family"] = meta[ds["dx_col"]].values
    if ds["subtype_col"] is not None:
        df["subtype"] = meta[ds["subtype_col"]].values
    df["subject_id"] = meta[ds["id_col"]].values
    return df, region_names


def to_centile_scores(df, region_names, reference_group):
    """
    Empirical centile in [0, 1] of every subject's parcel mass relative to
    the reference (CN) group of the cohort (the analogue of the normative
    centile scores used by Thalhammer et al.). Falls back to the full
    sample if the reference group has fewer than 5 subjects.
    """
    ref = df[df["dx_family"] == reference_group]
    if len(ref) < 5:
        print(f"  reference group '{reference_group}' too small/missing "
              f"-> using the full sample as reference instead")
        ref = df

    cent = pd.DataFrame(index=df.index)
    for region in region_names:
        ref_vals = np.sort(ref[region].values)
        cent[region] = np.searchsorted(ref_vals, df[region].values, side="right") / len(ref_vals)
    return cent


# ----------------------------------------------------------------------
# Functions from Thalhammer et al. (code/spatial_heterogeneity.py)
# ----------------------------------------------------------------------
def binarize_extranormal(data):
    """
    Binarize extranormal deviations so that infra- or supranormal values
    are coded as 1, otherwise 0. (Thalhammer et al.)
    """
    data_bin = pd.DataFrame(
        np.where((data < 0.05) | (data > 0.95), 1, 0), columns=data.columns
    )
    return data_bin


def plot_binarized_extranormal(data_bin, title, outname, bin=True, fsize=10):
    """Plot the binarized extranormal deviations. (Thalhammer et al.)"""
    if bin:
        fig = plt.figure(figsize=(2.5, 4))
        heatmap = sns.heatmap(data_bin, cmap=["white", "k"], cbar=False, figure=fig)
        plt.xlabel("ROIs", fontsize=fsize)
        plt.ylabel("Subject-IDs", fontsize=fsize)
    else:
        fig = plt.figure(figsize=(5, 2.5))
        heatmap = sns.heatmap(
            data_bin.T, cmap="RdBu_r", center=0.5, cbar=True,
            cbar_kws={"label": "Deviation score"}, figure=fig,
        )
        heatmap.collections[0].colorbar.ax.yaxis.label.set_size(fsize - 2)
        plt.xlabel("Subject-IDs", fontsize=fsize)
        plt.ylabel("ROIs", fontsize=fsize)

    plt.title(title)
    heatmap.set_xticklabels("")
    heatmap.set_yticklabels("")

    plt.tight_layout()
    plt.savefig(outname, dpi=300)
    plt.close(fig)


def plot_correlation_matrix_kde(corr_matrix, outname, fsize=12, across="subjects"):
    """Heatmap of the correlation matrix + KDE of the average Spearman rho. (Thalhammer et al.)"""
    fig = plt.figure(figsize=(8, 6))
    gs = GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[15, 0.5], figure=fig)
    ax_heatmap = fig.add_subplot(gs[0, 0])
    ax_cbar = fig.add_subplot(gs[1, 0])

    sns.heatmap(
        corr_matrix, cmap="vlag", vmax=1, vmin=-1, square=True,
        ax=ax_heatmap, cbar_ax=ax_cbar,
        cbar_kws={"orientation": "horizontal", "label": "Spearman rho"},
    )

    sns.set_style("whitegrid")
    ax_kde = fig.add_subplot(gs[0, 1])
    row_means = np.mean(corr_matrix, axis=1)
    sns.kdeplot(y=row_means, ax=ax_kde, fill=False, color="k")
    ax_kde.set_xlabel("Density", fontsize=fsize)
    ax_kde.tick_params(axis="x", labelsize=fsize - 3)
    ax_kde.tick_params(axis="y", labelsize=fsize - 3)
    ax_kde.set_ylim(-1, 1)
    ax_kde.set_xticks(np.linspace(0, 10, 3))

    if across == "subjects":
        ax_heatmap.set_xlabel("Subject-IDs", fontsize=fsize)
        ax_heatmap.set_ylabel("Subject-IDs", fontsize=fsize)
        ax_kde.set_ylabel("Average Spearman rho per subject", fontsize=fsize)
    elif across == "regions":
        ax_heatmap.set_xlabel("ROIs", fontsize=fsize)
        ax_heatmap.set_ylabel("ROIs", fontsize=fsize)
        ax_kde.set_ylabel("Average Spearman rho per region", fontsize=fsize)

    ax_heatmap.set_xticklabels("")
    ax_heatmap.set_yticklabels("")

    plt.tight_layout()
    plt.savefig(outname, dpi=300)
    plt.close(fig)
    sns.set_style("white")


def plot_mean_rho(corr_matrix, title, outname, fsize=12):
    """Plot the distribution of mean Spearman rho. (Thalhammer et al.)"""
    row_means = corr_matrix.mean(axis=1)

    sns.set_style("whitegrid")
    plt.figure(figsize=(5, 2))
    sns.kdeplot(x=row_means, linewidth=2, color="k")
    plt.xlim(-1, 1)
    sns.despine()
    plt.title(title, fontsize=fsize)
    plt.xlabel("Spearman rho", fontsize=fsize)
    plt.ylabel("Density", fontsize=fsize)
    plt.xticks(fontsize=fsize - 2)
    plt.yticks(fontsize=fsize - 2)
    plt.gca().set_yticks(np.linspace(0, 15, 4))

    plt.tight_layout()
    plt.savefig(outname, dpi=300)
    plt.close()
    sns.set_style("white")


def percent_extranormal(data_bin):
    """Percentage of subjects with at least one extranormal region."""
    return (data_bin.sum(axis=1) > 0).mean() * 100


# ----------------------------------------------------------------------
# Additional statistics and thesis figures
# ----------------------------------------------------------------------
def row_means_excl_self(corr):
    """Row means of the correlation matrix without the diagonal."""
    c = corr.values.astype(float).copy()
    np.fill_diagonal(c, np.nan)
    return pd.Series(np.nanmean(c, axis=1), index=corr.index)


def _ordered(bin_sub):
    """Columns in regional order, rows sorted by number of extranormal parcels."""
    X = bin_sub[PARCEL_ORDER].values
    return X[np.argsort(-X.sum(axis=1), kind="stable")]


def _barcode_panel(ax, bin_sub, title, ylabels):
    X = _ordered(bin_sub)
    ax.imshow(X.T, aspect="auto", cmap=ListedColormap(["white", "black"]),
              interpolation="nearest", vmin=0, vmax=1)
    for b in REGION_BOUNDS:
        ax.axhline(b - 0.5, color="#d0473c", lw=0.6)
    ax.set_xticks([])
    ax.set_title(title, fontsize=7)
    ax.set_xlabel(f"$n={X.shape[0]}$", fontsize=6)
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color("#999"); sp.set_linewidth(0.5)
    if ylabels:
        ax.set_yticks(range(len(PARCEL_ORDER)))
        ax.set_yticklabels([PRETTY[p] for p in PARCEL_ORDER], fontsize=5.6)
    else:
        ax.set_yticks([])


def plot_thesis_fig6(bins, cfg, outname):
    """Figure 6: (a) labeled barcodes, (b) per-parcel prevalence heatmap."""
    plt.rcParams.update({"font.size": 7})
    fig = plt.figure(figsize=(6.5, 5.0))
    gs = fig.add_gridspec(1, 7, width_ratios=[1, 1, 1, 1, 1, 0.3, 3.0], wspace=0.08,
                          left=0.15, right=0.87, top=0.86, bottom=0.15)
    for i, (ds, gv, grp, title) in enumerate(cfg["fig6_barcodes"]):
        ax = fig.add_subplot(gs[0, i])
        _barcode_panel(ax, bins[(ds, gv, grp)], title, i == 0)

    ax = fig.add_subplot(gs[0, 6])
    cols = cfg["fig6_prevalence"]
    P = np.array([bins[(ds, gv, grp)][PARCEL_ORDER].mean(axis=0).values * 100
                  for ds, gv, grp, _ in cols]).T
    im = ax.imshow(P, aspect="auto", cmap="Reds", vmin=0, vmax=100)
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            ax.text(j, i, f"{P[i, j]:.0f}", ha="center", va="center", fontsize=4.6,
                    color="white" if P[i, j] > 60 else "#222")
    for b in REGION_BOUNDS:
        ax.axhline(b - 0.5, color="#333", lw=0.6)
    ax.set_yticks([])
    ax.set_xticks(range(len(cols)))
    ax.xaxis.tick_top()
    ax.set_xticklabels([c[3] for c in cols], rotation=60, ha="left", fontsize=5.6)
    for sp in ax.spines.values():
        sp.set_visible(False)
    start = 0
    for rn, ps in REGIONS:
        ax.text(len(cols) - 0.3, start + (len(ps) - 1) / 2, rn, ha="left", va="center",
                fontsize=5.8, color="#444")
        start += len(ps)
    cax = fig.add_axes([0.6, 0.07, 0.25, 0.016])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("% of subjects with extranormal parcel", fontsize=6)
    cb.ax.tick_params(labelsize=6)
    fig.text(0.01, 0.95, "a", fontsize=9, weight="bold")
    fig.text(0.57, 0.95, "b", fontsize=9, weight="bold")
    fig.savefig(outname, dpi=300)
    fig.savefig(outname.replace(".pdf", ".png"), dpi=300)
    plt.close(fig)


def plot_thesis_supp_barcodes(bins, cfg, outname, max_rows=500):
    plt.rcParams.update({"font.size": 7})
    entries = [e for e in cfg["supp_barcodes"]
               if e[:3] in bins and len(bins[e[:3]]) <= max_rows]
    ncol = 6
    nrow = int(np.ceil(len(entries) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.5, 3.8 * nrow), squeeze=False,
                             gridspec_kw=dict(wspace=0.08, hspace=0.25, left=0.17,
                                              right=0.99, top=0.95, bottom=0.05))
    for k, ax in enumerate(axes.flat):
        if k >= len(entries):
            ax.axis("off"); continue
        ds, gv, grp, title = entries[k]
        _barcode_panel(ax, bins[(ds, gv, grp)], title, k % ncol == 0)
    fig.savefig(outname, dpi=300)
    fig.savefig(outname.replace(".pdf", ".png"), dpi=300)
    plt.close(fig)


def plot_thesis_supp_corr(bins, cfg, outname):
    """Correlation matrices + histogram of row means (incl. self, as in the summary)."""
    plt.rcParams.update({"font.size": 7})
    entries = [e for e in cfg["supp_corr"] if e[:3] in bins]
    fig, axes = plt.subplots(2, len(entries), figsize=(6.5, 3.6), squeeze=False,
                             gridspec_kw=dict(height_ratios=[1, 0.55], hspace=0.45, wspace=0.35,
                                              left=0.07, right=0.92, top=0.92, bottom=0.12))
    for j, (ds, gv, grp, title) in enumerate(entries):
        X = _ordered(bins[(ds, gv, grp)])
        corr = pd.DataFrame(X.T).corr(method="spearman")
        ax = axes[0, j]
        imc = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=7)
        m = corr.mean(axis=1)
        ax = axes[1, j]
        ax.hist(m.dropna(), bins=np.linspace(-0.2, 0.8, 21), color="#555", edgecolor="white", lw=0.4)
        ax.axvline(m.mean(), color="#d0473c", lw=1)
        ax.set_xlim(-0.2, 0.8); ax.set_xlabel(r"mean $\rho$ per subject", fontsize=6)
        if j == 0:
            ax.set_ylabel("subjects", fontsize=6)
    cb = fig.colorbar(imc, ax=axes[0, :].tolist(), fraction=0.015, pad=0.01)
    cb.ax.tick_params(labelsize=5); cb.set_label(r"Spearman $\rho$", fontsize=6)
    fig.savefig(outname, dpi=300)
    fig.savefig(outname.replace(".pdf", ".png"), dpi=300)
    plt.close(fig)


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------
def main():
    cfg = CONFIG
    summary_rows = []
    prevalence_rows = []
    bins = {}
    os.makedirs(str(cfg["output_dir"]), exist_ok=True)

    for dataset_name in cfg["datasets"]:
        print(f"\n=== {dataset_name} ===")
        df, region_names = load_dataset(dataset_name, cfg)
        centiles = to_centile_scores(df, region_names, cfg["reference_group"])
        bin_full = binarize_extranormal(centiles)

        for grouping_var in cfg["grouping_vars"]:
            if grouping_var not in df.columns:
                print(f"  skip {grouping_var} (not in metadata)")
                continue
            groups = df[grouping_var].dropna().unique()
            for group_label in sorted(groups, key=str):
                sub_idx = df.index[df[grouping_var] == group_label]
                if len(sub_idx) < cfg["min_group_size"]:
                    print(f"  skip {grouping_var}={group_label} "
                          f"(n={len(sub_idx)} < min_group_size)")
                    continue

                bin_sub = bin_full.loc[sub_idx].reset_index(drop=True)
                bins[(dataset_name, grouping_var, str(group_label))] = bin_sub
                safe = f"{dataset_name}_{grouping_var}_{str(group_label).replace('/', '-')}"
                out_e = os.path.join(str(cfg["output_dir"]), f"{safe}_binarized.png")
                out_fg = os.path.join(str(cfg["output_dir"]), f"{safe}_corrmatrix.png")

                print(f"  {grouping_var} = {group_label} (n={len(sub_idx)})")

                plot_binarized_extranormal(bin_sub, title="", outname=out_e)
                corr = bin_sub.T.corr(method="spearman")
                plot_correlation_matrix_kde(corr, out_fg, fsize=16)

                mean_rho = corr.mean(axis=1)
                mean_rho_excl = row_means_excl_self(corr)
                summary_rows.append({
                    "dataset": dataset_name,
                    "grouping_var": grouping_var,
                    "group": group_label,
                    "n_subjects": len(sub_idx),
                    "mean_spearman_rho": mean_rho.mean(),
                    "median_spearman_rho": mean_rho.median(),
                    "mean_spearman_rho_excl_self": mean_rho_excl.mean(),
                    "median_spearman_rho_excl_self": mean_rho_excl.median(),
                    "pct_subjects_with_any_extranormal_region": percent_extranormal(bin_sub),
                })
                prev = (bin_sub.mean(axis=0) * 100).to_dict()
                prevalence_rows.append({"dataset": dataset_name, "grouping_var": grouping_var,
                                        "group": group_label, "n_subjects": len(sub_idx), **prev})

    summary = pd.DataFrame(summary_rows)
    summary_path = os.path.join(str(cfg["output_dir"]), "heterogeneity_summary.csv")
    summary.to_csv(summary_path, index=False)
    pd.DataFrame(prevalence_rows).to_csv(
        os.path.join(str(cfg["output_dir"]), "parcel_prevalence.csv"), index=False)

    if cfg.get("make_thesis_figures", False):
        plot_thesis_fig6(bins, cfg, os.path.join(str(cfg["output_dir"]), "thesis_fig6_barcodes_main.pdf"))
        plot_thesis_supp_barcodes(bins, cfg, os.path.join(str(cfg["output_dir"]), "thesis_supp_barcodes.pdf"))
        plot_thesis_supp_corr(bins, cfg, os.path.join(str(cfg["output_dir"]), "thesis_supp_corr.pdf"))

    print(f"\nSaved plots + summary table to: {cfg['output_dir']}/")
    if not summary.empty:
        print(summary.sort_values(["grouping_var", "mean_spearman_rho"]).to_string(index=False))


if __name__ == "__main__":
    main()