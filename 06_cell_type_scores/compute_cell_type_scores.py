"""
Atrophy masses and cell-type scores per subject.

For every subject and hemisphere the smoothed deviation map z is turned into
atrophy weights w = max(-z, 0). From these:
    atrophy mass          sum of w (total, per DK region group (8), per DK parcel (34))
    raw cell-type score   sum of cell map * w
    normalized score      raw score / atrophy mass
    CN-referenced score   z-score against the CN group of the same cohort,
                          plus glial / vascular / neuronal composites

Input:  data/deviation_maps/{ds}/zscores.fwhm{fwhm}.fsavg5.{hemi}.npy
        data/metadata/meta_{DS}.csv   (same row order, columns AGE, SEX, DX_family)
        data/cell_maps/fsavg5_{hemi}_{cell}.npy
        data/fsaverage5/label/{hemi}.aparc.annot
Output: outputs/{ds}/scores/        score arrays (.npy) and their column names
        outputs/{ds}/cn_reference/  CN reference table and subject-level csv

Usage:  python compute_cell_type_scores.py [--datasets tum nifd nacc]
"""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

import scoring_utils as utils

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
OUTPUTS = REPO / "outputs"

CELL_DIR = DATA / "cell_maps"
ANNOT_DIR = DATA / "fsaverage5" / "label"

ATROPHY_SIGN = "neg"
Z_THRESHOLD = 0.0
STANDARDIZE_CELL_MAPS = False
HEMISPHERES = ["lh", "rh"]
FWHM_LEVELS = [0, 5, 10, 15]

CELL_TYPES = ["astrocytes", "endothelial", "microglia", "opc",
              "oligodendrocytes", "excitatory", "inhibitory", "pericytes"]

DK_GROUPS = {
    "central": [b"precentral", b"postcentral", b"paracentral"],
    "superiortemporal": [b"bankssts", b"transversetemporal", b"middletemporal", b"superiortemporal"],
    "inferior_medial_temporal": [b"parahippocampal", b"inferiortemporal", b"entorhinal", b"fusiform",
                                 b"temporalpole"],
    "limbic": [b"rostralanteriorcingulate", b"posteriorcingulate", b"caudalanteriorcingulate",
               b"isthmuscingulate", b"insula"],
    "parietal": [b"inferiorparietal", b"precuneus", b"supramarginal", b"superiorparietal"],
    "occipital": [b"lingual", b"cuneus", b"lateraloccipital", b"pericalcarine"],
    "ventral_frontal": [b"caudalmiddlefrontal", b"rostralmiddlefrontal", b"superiorfrontal"],
    "inferior_frontal": [b"lateralorbitofrontal", b"medialorbitofrontal", b"parsorbitalis",
                         b"parstriangularis", b"parsopercularis", b"frontalpole"],
}
REGION_GROUP_NAMES = list(DK_GROUPS.keys())
EXCLUDED_DK_REGIONS = {b"unknown", b"corpuscallosum"}

# "exclude": {column: [values]} removes matching subjects from the whole
# analysis (column "*" searches all text columns).
DATASETS = {
    "tum": {
        "zmap_dir": DATA / "deviation_maps" / "tum",
        "meta_path": DATA / "metadata" / "meta_TUM.csv",
        "out_dir": OUTPUTS / "tum",
    },
    "nifd": {
        "zmap_dir": DATA / "deviation_maps" / "nifd",
        "meta_path": DATA / "metadata" / "meta_NIFD.csv",
        "out_dir": OUTPUTS / "nifd",
    },
    "nacc": {
        "zmap_dir": DATA / "deviation_maps" / "nacc",
        "meta_path": DATA / "metadata" / "meta_NACC.csv",
        "out_dir": OUTPUTS / "nacc",
    },
}

_VERTEX_MAP_CACHE = {}


def get_cell_files(hemi):
    return {cell: f"fsavg5_{hemi}_{cell}.npy" for cell in CELL_TYPES}


def compute_keep_mask(meta_raw, exclude, ds_name):
    """Boolean mask of subjects to keep after applying the {column: [values]} exclusion rules."""
    keep = np.ones(len(meta_raw), dtype=bool)
    if not exclude:
        return keep

    def norm(col):
        return meta_raw[col].astype(str).str.strip().str.upper()

    for col, values in exclude.items():
        vals = {str(v).strip().upper() for v in values}
        if col in (None, "*"):
            candidate_cols = [c for c in meta_raw.columns
                              if meta_raw[c].dtype == object or pd.api.types.is_string_dtype(meta_raw[c])]
        else:
            col_u = str(col).upper()
            if col_u not in meta_raw.columns:
                raise KeyError(f"[{ds_name}] exclusion column {col!r} not in metadata columns")
            candidate_cols = [col_u]

        hit_cols = [c for c in candidate_cols if norm(c).isin(vals).any()]
        if not hit_cols:
            raise ValueError(f"[{ds_name}] exclusion values {sorted(vals)} not found in {candidate_cols}")
        for c in hit_cols:
            hit = norm(c).isin(vals).to_numpy()
            print(f"[{ds_name}] Excluding {int((hit & keep).sum())} subjects with {c} in {sorted(vals)}")
            keep &= ~hit

    print(f"[{ds_name}] Keeping {int(keep.sum())}/{len(keep)} subjects after exclusions.")
    return keep


def load_meta(meta_path, n_rows, ds_name, exclude=None):
    """Return (metadata with columns dataset/age/sex/dx, keep mask for the deviation maps)."""
    meta = pd.read_csv(meta_path)
    meta.columns = meta.columns.str.upper()
    if len(meta) != n_rows:
        raise ValueError(f"Meta rows {len(meta)} != z-map rows {n_rows} ({meta_path})")

    df_meta = pd.DataFrame({
        "dataset": ds_name,
        "age": meta["AGE"].astype(float),
        "sex": meta["SEX"].astype(str),
        "dx": meta["DX_FAMILY"].astype(str),
    })

    keep = compute_keep_mask(meta, exclude, ds_name)
    df_meta = df_meta.loc[keep].reset_index(drop=True)
    df_meta["sex"] = df_meta["sex"].str.strip().str.capitalize()
    df_meta["dx"] = df_meta["dx"].str.strip().str.upper()
    df_meta["dataset"] = df_meta["dataset"].str.strip()
    return df_meta, keep


def build_vertex_maps(annot_path):
    """
    Per-vertex index into REGION_GROUP_NAMES (-1 = not in a group) and into
    the list of DK parcels (-1 = excluded), plus the parcel names.
    """
    labels, _, names = nib.freesurfer.read_annot(annot_path)
    name_to_group = {region: gi for gi, g in enumerate(REGION_GROUP_NAMES) for region in DK_GROUPS[g]}

    label_to_group = np.full(len(names), -1, dtype=np.int64)
    label_to_parcel = np.full(len(names), -1, dtype=np.int64)
    dk_parcel_names, unmapped = [], []
    for li, nm in enumerate(names):
        if nm in EXCLUDED_DK_REGIONS:
            continue
        label_to_parcel[li] = len(dk_parcel_names)
        dk_parcel_names.append(nm.decode("utf-8") if isinstance(nm, bytes) else str(nm))
        if nm in name_to_group:
            label_to_group[li] = name_to_group[nm]
        else:
            unmapped.append(nm)
    if unmapped:
        print(f"[WARNING] DK regions not assigned to any of the 8 groups: {unmapped}")

    labels = np.asarray(labels)
    valid = labels >= 0
    vertex_group = np.full(labels.shape[0], -1, dtype=np.int64)
    vertex_group[valid] = label_to_group[labels[valid]]
    vertex_parcel = np.full(labels.shape[0], -1, dtype=np.int64)
    vertex_parcel[valid] = label_to_parcel[labels[valid]]
    return vertex_group, vertex_parcel, dk_parcel_names


def get_vertex_maps(hemi, n_vertices):
    if hemi not in _VERTEX_MAP_CACHE:
        annot_path = ANNOT_DIR / f"{hemi}.aparc.annot"
        if not annot_path.is_file():
            raise FileNotFoundError(f"DK annotation file not found: {annot_path}")
        vertex_group, vertex_parcel, dk_names = build_vertex_maps(str(annot_path))
        if vertex_group.shape[0] != n_vertices:
            raise ValueError(f"Vertex mismatch between annot ({vertex_group.shape[0]}) and z-maps ({n_vertices})")
        _VERTEX_MAP_CACHE[hemi] = (vertex_group, vertex_parcel, dk_names)
    return _VERTEX_MAP_CACHE[hemi]


def run_one(ds_name, cfg, hemi, fwhm):
    """Compute and save all scores for one (dataset, hemisphere, FWHM) combination."""
    out_scores_dir = Path(cfg["out_dir"]) / "scores"
    out_ref_dir = Path(cfg["out_dir"]) / "cn_reference" / f"fwhm{fwhm}" / hemi
    out_scores_dir.mkdir(parents=True, exist_ok=True)
    out_ref_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}\n  Dataset={ds_name}  hemi={hemi}  fwhm={fwhm}\n{'=' * 60}")
    Z = np.load(Path(cfg["zmap_dir"]) / f"zscores.fwhm{fwhm}.fsavg5.{hemi}.npy").astype(np.float32)
    if Z.ndim != 2:
        raise ValueError(f"Expected z-maps of shape (N, V). Got {Z.shape}")
    df_meta, keep = load_meta(cfg["meta_path"], Z.shape[0], ds_name, cfg.get("exclude"))
    Z = Z[keep]
    N, V = Z.shape
    np.save(out_scores_dir / f"{ds_name}_kept_rows.npy", np.flatnonzero(keep))
    print(f"Z-maps: N={N}, V={V}")
    print(df_meta["dx"].value_counts().to_string())

    cell_names, C = utils.load_cell_maps(CELL_DIR, get_cell_files(hemi), standardize=STANDARDIZE_CELL_MAPS)
    C = np.where(np.isfinite(C), C, 0.0).astype(np.float32)
    if C.shape[1] != V:
        raise ValueError(f"Vertex mismatch: z-maps V={V}, cell maps V={C.shape[1]}")

    vertex_group, vertex_parcel, dk_names = get_vertex_maps(hemi, V)
    region_masks = [vertex_group == g for g in range(len(REGION_GROUP_NAMES))]
    parcel_masks = [vertex_parcel == p for p in range(len(dk_names))]

    scores_raw = np.zeros((N, len(cell_names)), dtype=np.float32)
    scores_norm = np.zeros((N, len(cell_names)), dtype=np.float32)
    atrophy_mass = np.zeros(N, dtype=np.float32)
    atrophy_regional = np.zeros((N, len(REGION_GROUP_NAMES)), dtype=np.float32)
    atrophy_dk = np.zeros((N, len(dk_names)), dtype=np.float32)
    eps = 1e-8
    for i in range(N):
        w = utils.compute_weights(Z[i], sign=ATROPHY_SIGN, zthr=Z_THRESHOLD)
        mass = float(np.sum(w))
        atrophy_mass[i] = mass
        atrophy_regional[i] = [float(w[m].sum()) for m in region_masks]
        atrophy_dk[i] = [float(w[m].sum()) for m in parcel_masks]
        raw = C @ w
        scores_raw[i] = raw
        scores_norm[i] = raw / (mass + eps) if mass > 0 else np.nan

    dk_names_path = out_scores_dir / "atrophy_dk_names.npy"
    if dk_names_path.exists() and np.load(dk_names_path, allow_pickle=True).tolist() != dk_names:
        raise ValueError("DK parcel names/order differ between hemispheres; check the annot files.")

    prefix = f"{ds_name}_{hemi}"
    np.save(out_scores_dir / f"{prefix}_scores_raw.fwhm{fwhm}.npy", scores_raw)
    np.save(out_scores_dir / f"{prefix}_scores_norm.fwhm{fwhm}.npy", scores_norm)
    np.save(out_scores_dir / f"{prefix}_scores_norm_all.fwhm{fwhm}.npy",
            np.hstack((scores_norm, atrophy_mass[:, None])))
    np.save(out_scores_dir / f"{prefix}_atrophy.fwhm{fwhm}.npy", atrophy_mass)
    np.save(out_scores_dir / f"{prefix}_atrophy_regional.fwhm{fwhm}.npy", atrophy_regional)
    np.save(out_scores_dir / f"{prefix}_atrophy_dk.fwhm{fwhm}.npy", atrophy_dk)
    np.save(out_scores_dir / "feature_names.npy", cell_names)
    np.save(out_scores_dir / "feature_names_all.npy", cell_names + ["atrophy_mass"])
    np.save(out_scores_dir / "atrophy_regional_names.npy", np.array(REGION_GROUP_NAMES))
    np.save(dk_names_path, np.array(dk_names))

    df = df_meta.copy()
    for j, nm in enumerate(cell_names):
        df[f"{nm}_raw"] = scores_raw[:, j]
        df[f"{nm}_norm"] = scores_norm[:, j]
        df[nm] = scores_norm[:, j]
    df["atrophy_mass"] = atrophy_mass

    print("\nMean atrophy mass by diagnosis:")
    print(df.groupby("dx")["atrophy_mass"].agg(["count", "mean", "std"]).to_string())

    for data in df["dataset"].dropna().unique():
        ds_dev, ds_ref = utils.compute_cn_reference_scores(df, data, cell_names)
        if ds_ref is None:
            continue
        ds_ref.to_csv(out_ref_dir / f"{data}_CN_reference_table.csv", index=False)
        ds_dev.to_csv(out_ref_dir / f"{data}_subject_level_deviation_scores.csv", index=False)
        summary_cols = [f"{c}_cn_z" for c in cell_names] + list(utils.COMPOSITES)
        print(f"\n{data}: mean CN-referenced scores by diagnosis")
        print(ds_dev.groupby("dx")[summary_cols].mean().round(3).to_string())


def parse_args():
    p = argparse.ArgumentParser(description="Cell-type atrophy scores for all datasets, hemispheres and FWHM levels.")
    p.add_argument("--datasets", nargs="*", default=None, help=f"subset of {list(DATASETS)} (default: all)")
    return p.parse_args()


def main():
    args = parse_args()
    for ds_name in args.datasets or list(DATASETS):
        cfg = DATASETS.get(ds_name)
        if cfg is None:
            print(f"[WARNING] Unknown dataset, skipping: {ds_name}")
            continue
        if not Path(cfg["zmap_dir"]).is_dir():
            print(f"[WARNING] Directory not found, skipping: {cfg['zmap_dir']}")
            continue
        for hemi in HEMISPHERES:
            for fwhm in FWHM_LEVELS:
                if not (Path(cfg["zmap_dir"]) / f"zscores.fwhm{fwhm}.fsavg5.{hemi}.npy").is_file():
                    continue
                run_one(ds_name, cfg, hemi, fwhm)
    print("\nAll done.")


if __name__ == "__main__":
    main()
