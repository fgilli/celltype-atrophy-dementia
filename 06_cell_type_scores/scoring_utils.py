"""Helper functions for compute_cell_type_scores.py."""

import os

import numpy as np
import pandas as pd

COMPOSITES = {
    "glial_disease_z": ["astrocytes", "microglia", "opc", "oligodendrocytes"],
    "vascular_disease_z": ["endothelial", "pericytes"],
    "neuronal_disease_z": ["excitatory", "inhibitory"],
}
MIN_CN_SUBJECTS = 20


def load_cell_maps(cell_dir, file_dict, standardize=False):
    """Return (names, C) with C of shape (n_cell_types, n_vertices)."""
    names = list(file_dict.keys())
    maps = []
    for name in names:
        path = os.path.join(cell_dir, file_dict[name])
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing cell map: {path}")
        v = np.load(path).astype(np.float32)
        if v.ndim != 1:
            raise ValueError(f"Cell map {name} must be 1D. Got {v.shape}.")
        if standardize:
            sd = np.nanstd(v)
            v = (v - np.nanmean(v)) / (sd if np.isfinite(sd) and sd != 0 else 1.0)
        maps.append(v)
    return names, np.stack(maps, axis=0)


def compute_weights(z_row, sign="neg", zthr=0.0):
    """Atrophy weights of one subject: thinning (sign='neg') or thickening (sign='pos') only."""
    if sign == "neg":
        w = -z_row
    elif sign == "pos":
        w = z_row
    else:
        raise ValueError("sign must be 'neg' or 'pos'")
    if zthr > 0:
        w = np.where(w >= zthr, w, 0.0)
    else:
        w = np.maximum(w, 0.0)
    return w.astype(np.float32)


def compute_cn_reference_scores(df, dataset_name, cell_names):
    """
    CN-referenced deviation scores within one dataset.

    For every cell type c, z_cn = (S_c - mean_CN) / SD_CN with mean and SD
    of the CN subjects of the same dataset. The composites are the means of
    z_cn over the glial, vascular and neuronal cell types and are then
    re-standardized against CN (CN mean 0, SD 1).

    Returns (subjects, reference_table), or (subjects, None) if the dataset
    has fewer than MIN_CN_SUBJECTS CN subjects. Added columns:
    {cell}_cn_z, glial_disease_z, vascular_disease_z, neuronal_disease_z.
    """
    sub = df[df["dataset"] == dataset_name].copy()
    for c in cell_names:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")

    cn_mask = sub["dx"] == "CN"
    if cn_mask.sum() < MIN_CN_SUBJECTS:
        print(f"[{dataset_name}] Too few CN subjects ({int(cn_mask.sum())}) for a stable reference.")
        return sub, None

    mu_cn = sub.loc[cn_mask, cell_names].mean()
    sd_cn = sub.loc[cn_mask, cell_names].std(ddof=1).replace(0, np.nan)
    print(f"[{dataset_name}] CN reference: n={int(cn_mask.sum())}")

    ref_tbl = pd.DataFrame({"cell_type": cell_names, "cn_mean": mu_cn.values, "cn_std": sd_cn.values})
    for dx_level in sorted(sub["dx"].dropna().unique()):
        ref_tbl[f"{dx_level}_mean"] = sub.loc[sub["dx"] == dx_level, cell_names].mean().values

    for c in cell_names:
        sub[f"{c}_cn_z"] = (sub[c] - mu_cn[c]) / sd_cn[c]

    for comp_name, cells in COMPOSITES.items():
        cols = [f"{c}_cn_z" for c in cells if c in cell_names]
        if not cols:
            continue
        raw = sub[cols].mean(axis=1)
        cn_vals = raw[cn_mask].dropna()
        comp_sd = cn_vals.std(ddof=1)
        if len(cn_vals) < 2 or not np.isfinite(comp_sd) or comp_sd == 0:
            print(f"[{dataset_name}] Cannot re-standardize {comp_name}; left unstandardized.")
            sub[comp_name] = raw
        else:
            sub[comp_name] = (raw - cn_vals.mean()) / comp_sd

    return sub, ref_tbl
