"""
Semi-manual preselection of the TUM cohort by atrophy-pattern specificity.

Applied to the SCSR deviation maps of the assembled TUM cohort
(build_tum_dataset.py). Scores are computed once at CONFIG["selection_fwhm"]
(5 mm) from the mean z-score of every Desikan-Killiany parcel (fsaverage5
aparc; "unknown" and "corpuscallosum" excluded).

Patients (AD, PCA, bvFTD, SD) are scored against a literature-based core
ROI of their diagnosis (CONFIG["core_rois"], sources in
core_roi_sources.txt):
    Z_in, Z_out          mean z inside / outside the core ROI (hemispheres
                         weighted by CONFIG["hemi_weights"])
    specificity          Z_out - Z_in (positive = atrophy concentrated in the ROI)
    topk_overlap_frac    fraction of the k most atrophic parcels that lie in the ROI
A patient is flagged for manual review if the specificity is in the lowest
CONFIG["specificity_flag_percentile"] % of the diagnosis or the top-k overlap
is below CONFIG["topk_overlap_flag_threshold"]. AD patients without subtype
are scored against the union of the AD and PCA ROIs. Diagnoses in
CONFIG["pass_through_diagnoses"] (nfvPPA) have no ROI and are kept without
review.

All CN subjects are sent to manual review; their global burden and an IQR
outlier flag are reported only to help sort the review.

Stages
    1. select:  compute scores and flags, write subject_scores.csv,
                flagged_for_review.csv and meta_annotated.csv, and render the
                deviation maps of all flagged subjects, all CN subjects and a
                random sample of unflagged patients for visual inspection.
    2. (manual) fill in the column "manual_keep" (True/False) of
                flagged_for_review.csv.
    3. filter:  keep all unflagged subjects and the flagged subjects with
                manual_keep = True; write the filtered metadata and z-score
                arrays for all smoothing levels to output_dir/filtered_dataset/.

Usage
    python preselect_atrophy_pattern.py select
    python preselect_atrophy_pattern.py filter --reviewed path/to/flagged_for_review.csv
"""

import argparse
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import nibabel as nib
except ImportError:
    nib = None

REPO = Path(__file__).resolve().parents[1]

CONFIG = {
    # input: output folder of build_tum_dataset.py + a metadata table that is
    # row-aligned with its arrays
    "data_dir": str(REPO / "data" / "tum_sources" / "combined"),
    "meta_path": str(REPO / "data" / "tum_sources" / "combined" / "meta_combined.csv"),
    "output_dir": str(REPO / "outputs" / "tum_preselection"),
    "fsaverage5_annot_dir": str(REPO / "data" / "fsaverage5" / "label"),
    "fsaverage5_surf_dir": str(REPO / "data" / "fsaverage5" / "surf"),
    "surface_fname_pattern": "{hemi}.pial",

    "hemis": ["lh", "rh"],
    "fwhm_levels": [0, 5, 10, 15],
    "zscore_fname_pattern": "zscores.fwhm{fwhm}.fsavg5.{hemi}.npy",
    "id_col": "id",
    "diagnosis_col": "subtype",
    "dx_family_col": "dx_family",

    # AD-family subjects without a subtype are scored against the union of these ROIs
    "unknown_subtype_merge": {
        "AD": ["AD", "PCA"],
    },

    "selection_fwhm": 5,
    # apply the final exclusion to the arrays of all FWHM levels
    "propagate_to_other_fwhm": True,

    "diagnosis_aliases": {
        "CN": "CN", "cn": "CN",
        "AD": "AD", "ad": "AD",
        "bvFTD": "bvFTD",
        "SD": "SD", "sd": "SD",
        "PCA": "PCA", "pca": "PCA",
        "nfvPPA": "nfvPPA",
    },
    "pass_through_diagnoses": ["nfvPPA"],

    # Desikan-Killiany names without hemisphere prefix, applied bilaterally
    "core_rois": {
        "AD": ["entorhinal", "inferiortemporal", "middletemporal", "superiortemporal",
               "inferiorparietal", "fusiform", "precuneus", "parahippocampal",
               "isthmuscingulate", "rostralanteriorcingulate", "caudalanteriorcingulate",
               "posteriorcingulate", "supramarginal", "bankssts"],
        "bvFTD": ["superiorfrontal", "rostralmiddlefrontal", "caudalmiddlefrontal",
                  "parsorbitalis", "parstriangularis", "parsopercularis",
                  "middletemporal", "superiortemporal", "frontalpole", "lateralorbitofrontal",
                  "medialorbitofrontal", "inferiortemporal", "temporalpole", "entorhinal",
                  "insula", "fusiform"],
        "SD": ["temporalpole", "entorhinal", "superiortemporal", "middletemporal",
               "inferiortemporal", "lateralorbitofrontal", "insula", "fusiform"],
        "PCA": ["superiorparietal", "inferiorparietal", "precuneus",
                "cuneus", "lateraloccipital", "supramarginal"],
    },
    # diagnoses not listed are weighted 0.5 / 0.5
    "hemi_weights": {
        "SD": {"lh": 0.5, "rh": 0.5},
        "PCA": {"lh": 0.5, "rh": 0.5},
    },
    "exclude_parcels": ["unknown", "corpuscallosum"],

    # patients
    "topk_parcels": 8,
    "specificity_flag_percentile": 20,
    "topk_overlap_flag_threshold": 0.25,

    # CN (informational only, all CN subjects are reviewed)
    "cn_burden_metric": "mean_z",          # "mean_z" or "frac_severe"
    "cn_severe_z_threshold": -2.0,
    "cn_iqr_multiplier": 1.5,
    "cn_flag_direction": "lower",          # "lower", "upper" or "both"

    # renders for the manual review
    "render_flagged_subjects": True,
    "render_unflagged_qc_samples": True,
    "n_unflagged_qc_samples": 10,
    "unflagged_qc_random_seed": 42,
    "render_dir_name": "renders",
    "cn_render_dir_name": "renders_cn",
    "unflagged_qc_render_dir_name": "renders_unflagged_qc",
    "render_window_size": (1000, 1000),
    # flip to -1 if lateral and medial views come out swapped
    "render_lateral_sign": 1,

    "overwrite": False,
}


# =============================================================================
# I/O helpers
# =============================================================================


def _zscore_path(hemi, fwhm):
    fname = CONFIG["zscore_fname_pattern"].format(hemi=hemi, fwhm=fwhm)
    return os.path.join(CONFIG["data_dir"], fname)


_META_CACHE = None


def get_meta():
    """Load the single shared meta CSV once and cache it."""
    global _META_CACHE
    if _META_CACHE is None:
        meta_path = CONFIG["meta_path"]
        meta = pd.read_csv(meta_path)
        id_col = CONFIG["id_col"]
        if id_col not in meta.columns:
            raise KeyError(
                f"id_col '{id_col}' not found in {meta_path}; "
                f"available columns: {list(meta.columns)}"
            )
        _META_CACHE = meta
    return _META_CACHE


def load_zscore_array(hemi, fwhm):
    """
    Load the (n_subjects, 10242) z-score array for one hemi/fwhm.
    Row i is assumed to correspond to row i of the shared meta table
    (get_meta()) - raises if the row counts don't match.
    """
    z = np.load(_zscore_path(hemi, fwhm))
    meta = get_meta()
    if z.shape[0] != len(meta):
        raise ValueError(
            f"Row mismatch for {hemi} fwhm{fwhm}: {z.shape[0]} z-score rows "
            f"vs {len(meta)} rows in shared meta ({CONFIG['meta_path']}). "
            f"All .npy arrays must be row-aligned to the shared meta table."
        )
    return z


def load_aparc_labels(hemi):
    """Return (labels_per_vertex, {parcel_name: label_value}) for fsaverage5 aparc."""
    if nib is None:
        raise ImportError("nibabel is required to read aparc.annot files")
    annot_path = os.path.join(CONFIG["fsaverage5_annot_dir"], f"{hemi}.aparc.annot")
    labels, ctab, names = nib.freesurfer.read_annot(annot_path)
    names = [n.decode("utf-8") if isinstance(n, bytes) else n for n in names]
    name_to_label = {name: i for i, name in enumerate(names)}
    return labels, name_to_label


# =============================================================================
# Parcel-level z-score table
# =============================================================================


def compute_parcel_table(hemi, fwhm):
    """
    Return a DataFrame (index = subject_id, row-order = shared meta order) of
    mean z-score per aparc parcel for one hemisphere at a given FWHM, columns
    named "{hemi}_{parcel}". Excluded parcels (unknown, corpuscallosum, ...)
    are dropped.
    """
    z = load_zscore_array(hemi, fwhm)
    meta = get_meta()
    labels, name_to_label = load_aparc_labels(hemi)

    if labels.shape[0] != z.shape[1]:
        raise ValueError(
            f"aparc label length ({labels.shape[0]}) does not match "
            f"z-score vertex count ({z.shape[1]}) for {hemi}"
        )

    exclude = set(CONFIG["exclude_parcels"])
    cols = {}
    for name, label_val in name_to_label.items():
        if name in exclude:
            continue
        mask = labels == label_val
        if not mask.any():
            continue
        cols[f"{hemi}_{name}"] = z[:, mask].mean(axis=1)

    df = pd.DataFrame(cols)
    df.index = meta[CONFIG["id_col"]].values
    df.index.name = CONFIG["id_col"]
    return df


def build_combined_parcel_table(fwhm):
    """
    Concatenate lh + rh parcel tables for one FWHM level. Both are built from
    the same shared meta table, so they are already row-aligned (same
    subject_id order) - this is verified, not just assumed.
    Returns (parcel_df, meta_df).
    """
    lh_df = compute_parcel_table("lh", fwhm)
    rh_df = compute_parcel_table("rh", fwhm)

    if not lh_df.index.equals(rh_df.index):
        if set(lh_df.index) == set(rh_df.index):
            rh_df = rh_df.loc[lh_df.index]  # same set, different order -> realign
        else:
            raise ValueError(
                f"lh/rh subject_id mismatch at fwhm{fwhm}, even though both "
                f"were built from the shared meta table. Check that the "
                f"{{hemi}}_fwhm{fwhm}.npy arrays are both row-aligned to "
                f"{CONFIG['meta_path']}."
            )

    parcel_df = pd.concat([lh_df, rh_df], axis=1)
    return parcel_df, get_meta()


# =============================================================================
# Diseased-subject scoring: pattern specificity
# =============================================================================


def _normalize_diagnosis(raw):
    return CONFIG["diagnosis_aliases"].get(raw, raw)


def _hemi_weights_for(diagnosis):
    w = CONFIG["hemi_weights"].get(diagnosis, {"lh": 0.5, "rh": 0.5})
    total = sum(w.values())
    return {h: v / total for h, v in w.items()}


def _merged_core_names(members):
    """Union of core ROI parcel names across several diagnoses, order-preserving."""
    merged = []
    for m in members:
        for name in CONFIG["core_rois"][m]:
            if name not in merged:
                merged.append(name)
    return merged


def score_diseased_group(parcel_df, subject_ids, diagnosis, core_names=None, weight_key=None):
    """
    Compute Z_in / Z_out / Z_global / specificity / topk_overlap for one
    diagnostic group's subjects, using that diagnosis's core ROI.

    core_names: override the ROI list normally looked up via
        CONFIG["core_rois"][diagnosis] (used for merged-ROI scoring when the
        specific subtype is unknown, see CONFIG["unknown_subtype_merge"]).
    weight_key: override which CONFIG["hemi_weights"] entry to use (defaults
        to `diagnosis`; relevant when `diagnosis` isn't itself a hemi_weights
        key, e.g. a merged family label).

    Returns a DataFrame indexed by subject_id.
    """
    core_names = CONFIG["core_rois"][diagnosis] if core_names is None else core_names
    weight_key = diagnosis if weight_key is None else weight_key
    hemis = CONFIG["hemis"]
    weights = _hemi_weights_for(weight_key)

    core_cols = {h: [f"{h}_{n}" for n in core_names if f"{h}_{n}" in parcel_df.columns]
                 for h in hemis}
    all_cols = {h: [c for c in parcel_df.columns if c.startswith(f"{h}_")] for h in hemis}
    for h in hemis:
        missing = set(f"{h}_{n}" for n in core_names) - set(core_cols[h])
        if missing:
            warnings.warn(f"[{diagnosis}] core ROI parcels not found for {h}: {missing}")
    other_cols = {h: [c for c in all_cols[h] if c not in core_cols[h]] for h in hemis}

    sub = parcel_df.loc[subject_ids]

    z_in = sum(weights[h] * sub[core_cols[h]].mean(axis=1) for h in hemis)
    z_out = sum(weights[h] * sub[other_cols[h]].mean(axis=1) for h in hemis)
    z_global = sum(weights[h] * sub[all_cols[h]].mean(axis=1) for h in hemis)
    specificity = z_out - z_in  # more positive = atrophy more concentrated in core ROI

    k = CONFIG["topk_parcels"]
    core_col_set = set(core_cols["lh"]) | set(core_cols["rh"])
    # weight parcels by hemi relevance before ranking, so lateralized diagnoses
    # (e.g. SD) preferentially rank the trusted hemisphere's parcels
    weighted = sub.copy()
    for h in hemis:
        weighted[all_cols[h]] = weighted[all_cols[h]] * weights[h]

    topk_overlap = []
    for sid in sub.index:
        row = weighted.loc[sid]
        worst_k = row.sort_values(ascending=True).index[:k]  # most negative = most atrophic
        overlap = len(set(worst_k) & core_col_set) / k
        topk_overlap.append(overlap)

    out = pd.DataFrame({
        "Z_in": z_in, "Z_out": z_out, "Z_global": z_global,
        "specificity": specificity, "topk_overlap_frac": topk_overlap,
    }, index=sub.index)

    pct = CONFIG["specificity_flag_percentile"]
    spec_cutoff = np.percentile(out["specificity"], pct)
    out["specificity_percentile_rank"] = out["specificity"].rank(pct=True) * 100
    out["specificity_flag"] = out["specificity"] <= spec_cutoff
    out["topk_flag"] = out["topk_overlap_frac"] < CONFIG["topk_overlap_flag_threshold"]
    out["review_flag"] = out["specificity_flag"] | out["topk_flag"]
    out["flag_reason"] = np.where(
        out["review_flag"],
        np.where(out["specificity_flag"] & out["topk_flag"], "specificity+topk",
                 np.where(out["specificity_flag"], "specificity", "topk")),
        ""
    )
    return out


# =============================================================================
# CN scoring: burden context for manual review (no automatic pass/fail)
# =============================================================================


def score_cn_group(parcel_df, subject_ids):
    """
    Burden + IQR context for CN subjects.

    Every CN subject gets review_flag=True (all healthy subjects go to
    manual review - automatic IQR-outlier detection alone wasn't reliable
    enough to trust as a filter). The burden/fence/median columns are kept
    purely as sorting/context aids for that review; `iqr_outlier` records
    what the old automatic rule would have flagged, for reference only.
    """
    sub = parcel_df.loc[subject_ids]
    all_cols = [c for c in sub.columns]

    if CONFIG["cn_burden_metric"] == "frac_severe":
        thr = CONFIG["cn_severe_z_threshold"]
        burden = (sub[all_cols] < thr).mean(axis=1)  # higher = worse
    else:
        burden = sub[all_cols].mean(axis=1)  # more negative = worse

    q1, median, q3 = np.percentile(burden, [25, 50, 75])
    iqr = q3 - q1
    k = CONFIG["cn_iqr_multiplier"]
    lower_fence = q1 - k * iqr
    upper_fence = q3 + k * iqr

    direction = CONFIG["cn_flag_direction"]
    if direction == "lower":
        iqr_outlier = burden < lower_fence
    elif direction == "upper":
        iqr_outlier = burden > upper_fence
    elif direction == "both":
        iqr_outlier = (burden < lower_fence) | (burden > upper_fence)
    else:
        iqr_outlier = pd.Series(False, index=burden.index)

    out = pd.DataFrame({
        "burden": burden,
        "burden_lower_fence": lower_fence,
        "burden_median": median,
        "burden_upper_fence": upper_fence,
        "iqr_outlier": iqr_outlier,
        "review_flag": True,  # every CN subject goes to manual review
        "flag_reason": np.where(iqr_outlier, "cn_manual_review+iqr_outlier", "cn_manual_review"),
    }, index=sub.index)
    return out


# =============================================================================
# Stage 1: run selection at CONFIG["selection_fwhm"]
# =============================================================================


def run_selection():
    fwhm = CONFIG["selection_fwhm"]
    out_dir = Path(CONFIG["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    scores_path = out_dir / "subject_scores.csv"

    if scores_path.exists() and not CONFIG["overwrite"]:
        print(f"[skip] {scores_path} already exists (set CONFIG['overwrite']=True to redo)")
        scores = pd.read_csv(scores_path)
        if CONFIG["render_unflagged_qc_samples"]:
            render_unflagged_qc(scores, fwhm)
        return scores

    parcel_df, meta = build_combined_parcel_table(fwhm)
    id_col = CONFIG["id_col"]
    dx_col = CONFIG["diagnosis_col"]
    family_col = CONFIG.get("dx_family_col")
    meta = meta.copy()

    if family_col and family_col in meta.columns:
        family_series = meta[family_col]
    else:
        family_series = pd.Series([np.nan] * len(meta), index=meta.index)

    def _resolve_scoring_group(subtype_raw, family_raw):
        if pd.notna(subtype_raw) and str(subtype_raw).strip() != "":
            return _normalize_diagnosis(subtype_raw)
        if pd.notna(family_raw):
            family = str(family_raw).strip()
            if family in CONFIG.get("unknown_subtype_merge", {}):
                return f"{family}__merged"
        return np.nan  # unresolvable - subtype missing, no family merge rule

    meta["scoring_group"] = [
        _resolve_scoring_group(s, f) for s, f in zip(meta[dx_col], family_series)
    ]

    unresolved_ids = meta.loc[meta["scoring_group"].isna(), id_col].tolist()
    if unresolved_ids:
        warnings.warn(
            f"{len(unresolved_ids)} subjects have no resolvable scoring group "
            f"(empty '{dx_col}' and no matching '{family_col}' merge rule) - "
            f"not scored: {unresolved_ids}"
        )

    all_scores = []
    for group in sorted(meta["scoring_group"].dropna().unique()):
        ids = meta.loc[meta["scoring_group"] == group, id_col].values
        if group == "CN":
            scored = score_cn_group(parcel_df, ids)
            roi_desc = ""
        elif group in CONFIG["core_rois"]:
            scored = score_diseased_group(parcel_df, ids, group)
            roi_desc = group
        elif group.endswith("__merged"):
            family = group[: -len("__merged")]
            members = CONFIG["unknown_subtype_merge"][family]
            merged_names = _merged_core_names(members)
            scored = score_diseased_group(
                parcel_df, ids, diagnosis=family,
                core_names=merged_names, weight_key=family,
            )
            roi_desc = "+".join(members)
        elif group in CONFIG.get("pass_through_diagnoses", []):
            # no mask for this diagnosis - skip scoring entirely, never flag,
            # never send to manual review; just carry the subjects through.
            scored = pd.DataFrame({
                "review_flag": False,
                "flag_reason": "no_mask_passthrough",
            }, index=ids)
            roi_desc = "none (no mask - passthrough)"
        else:
            warnings.warn(f"No scoring rule for group '{group}' (n={len(ids)}) - skipping")
            continue
        scored = scored.reset_index().rename(columns={"index": id_col})
        scored["diagnosis"] = group
        scored["scoring_rois"] = roi_desc
        scored[id_col] = ids  # ensure exact dtype/order match
        all_scores.append(scored)

    scores = pd.concat(all_scores, ignore_index=True, sort=False)
    extra_cols = [c for c in meta.columns if c not in (id_col, dx_col, "scoring_group")]
    scores = scores.merge(meta[[id_col] + extra_cols], on=id_col, how="left")
    scores["selection_fwhm"] = fwhm
    scores.to_csv(scores_path, index=False)

    n_flagged = int(scores["review_flag"].sum())
    print(f"[run_selection] scored {len(scores)} subjects at fwhm={fwhm}, "
          f"{n_flagged} flagged for manual review -> {scores_path}")

    flagged_path = out_dir / "flagged_for_review.csv"
    flagged = scores[scores["review_flag"]].copy()
    flagged["manual_keep"] = np.nan  # to be filled in during manual review
    flagged.to_csv(flagged_path, index=False)
    print(f"[run_selection] flagged subjects saved to {flagged_path} "
          f"(add a 'manual_keep' column with True/False after review)")

    # CN subjects are ALL flagged by design (see score_cn_group) - render them
    # into their own folder, separate from the diseased-pattern-mismatch renders.
    flagged_diseased_ids = flagged.loc[flagged["diagnosis"] != "CN", id_col].tolist()
    flagged_cn_ids = flagged.loc[flagged["diagnosis"] == "CN", id_col].tolist()

    if CONFIG["render_flagged_subjects"]:
        if flagged_diseased_ids:
            render_flagged(flagged_diseased_ids, fwhm)
        if flagged_cn_ids:
            render_cn(flagged_cn_ids, fwhm)

    if CONFIG["render_unflagged_qc_samples"]:
        render_unflagged_qc(scores, fwhm)

    return scores


# =============================================================================
# Optional: PyVista renders of subjects (best-effort helper)
# =============================================================================


def _camera_for_view(mesh, hemi, view):
    """
    Compute a camera position looking along the x-axis at the mesh center,
    from the side appropriate for {hemi} x {view}.

    Assumes FreeSurfer RAS-like coordinates where x separates hemispheres
    (this is the standard fsaverage5 convention). "lateral" = viewed from
    the hemisphere's own outer side (what you'd see looking at the brain
    from outside the head); "medial" = viewed from the midline (the cut
    surface facing the other hemisphere).

    If your first render comes out mirrored (e.g. lateral view actually
    shows the medial surface), flip CONFIG["render_lateral_sign"] to -1.
    """
    center = np.array(mesh.center)
    bounds = mesh.bounds  # (xmin,xmax,ymin,ymax,zmin,zmax)
    diag = np.linalg.norm([bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]])
    dist = diag * 2.0

    hemi_sign = -1 if hemi == "lh" else 1
    direction_sign = hemi_sign if view == "lateral" else -hemi_sign
    direction_sign *= CONFIG["render_lateral_sign"]

    cam_loc = center + np.array([direction_sign * dist, 0, 0])
    view_up = (0, 0, 1)
    return [tuple(cam_loc), tuple(center), view_up]


def render_subjects(subject_ids, fwhm, render_dir_name=None, log_label="render_subjects"):
    """
    Render one PNG per subject: a 2x2 grid of
        [lh lateral, lh medial]
        [rh lateral, rh medial]
    Skipped gracefully if pyvista or the surface files are unavailable.
    """
    try:
        import pyvista as pv
    except ImportError:
        print(f"[{log_label}] pyvista not available - skipping renders")
        return
    if nib is None:
        print(f"[{log_label}] nibabel not available - skipping renders")
        return

    if render_dir_name is None:
        render_dir_name = CONFIG["render_dir_name"]
    render_dir = Path(CONFIG["output_dir"]) / render_dir_name
    render_dir.mkdir(parents=True, exist_ok=True)

    meshes, zdata = {}, {}
    for hemi in CONFIG["hemis"]:
        surf_path = os.path.join(
            CONFIG["fsaverage5_surf_dir"],
            CONFIG["surface_fname_pattern"].format(hemi=hemi),
        )
        if not os.path.exists(surf_path):
            print(f"[{log_label}] surface file not found: {surf_path} - skipping {hemi}")
            continue
        verts, faces = nib.freesurfer.read_geometry(surf_path)
        faces_pv = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int64)
        meshes[hemi] = pv.PolyData(verts, faces_pv)
        zdata[hemi] = load_zscore_array(hemi, fwhm)

    if not meshes:
        print(f"[{log_label}] no surface files found - skipping all renders")
        return

    meta = get_meta()
    id_to_row = {sid: i for i, sid in enumerate(meta[CONFIG["id_col"]].values)}
    grid_pos = {("lh", "lateral"): (0, 0), ("lh", "medial"): (0, 1),
                ("rh", "lateral"): (1, 0), ("rh", "medial"): (1, 1)}

    i = 1
    for sid in subject_ids:
        row = id_to_row.get(sid)
        if row is None:
            continue
        out_png = render_dir / f"{i}_{sid}_fwhm{fwhm}.png"
        if out_png.exists() and not CONFIG["overwrite"]:
            continue

        plotter = pv.Plotter(shape=(2, 2), off_screen=True,
                              window_size=CONFIG["render_window_size"], border=False)
        for hemi in CONFIG["hemis"]:
            for view in ("lateral", "medial"):
                r, c = grid_pos[(hemi, view)]
                plotter.subplot(r, c)
                if hemi not in meshes:
                    plotter.add_text(f"{hemi} surface missing", font_size=10)
                    continue
                mesh = meshes[hemi]
                mesh["z"] = zdata[hemi][row]
                plotter.add_mesh(mesh, scalars="z", cmap="coolwarm_r", clim=[-5, 0],
                                  show_scalar_bar=(r == 0 and c == 1))
                plotter.camera_position = _camera_for_view(mesh, hemi, view)
                plotter.add_text(f"{hemi.upper()} {view}", font_size=10, position="upper_left")

        family = meta[CONFIG["dx_family_col"]].iloc[row]
        subtype = meta[CONFIG["diagnosis_col"]].iloc[row]
        subtype = "N/A" if pd.isna(subtype) else subtype
        plotter.add_text(f"Diagnosis family: {family}, subtype: {subtype}, index: {row}",
                         font_size=13, position="lower_left")
        plotter.screenshot(str(out_png))
        plotter.close()
        i += 1
    print(f"[{log_label}] {len(subject_ids)} renders (one file each) -> {render_dir}")


def render_flagged(subject_ids, fwhm):
    render_subjects(
        subject_ids,
        fwhm,
        render_dir_name=CONFIG["render_dir_name"],
        log_label="render_flagged",
    )


def render_cn(subject_ids, fwhm):
    """Render every CN subject into its own folder, separate from renders/."""
    render_subjects(
        subject_ids,
        fwhm,
        render_dir_name=CONFIG["cn_render_dir_name"],
        log_label="render_cn",
    )


def render_unflagged_qc(scores, fwhm):
    """Render a reproducible random sample of subjects not flagged for review."""
    id_col = CONFIG["id_col"]
    n_samples = CONFIG["n_unflagged_qc_samples"]
    if n_samples <= 0:
        return

    unflagged = scores[~scores["review_flag"].fillna(False)].copy()
    if unflagged.empty:
        print("[render_unflagged_qc] no unflagged subjects available - skipping")
        return

    n = min(n_samples, len(unflagged))
    sampled = unflagged.sample(
        n=n,
        random_state=CONFIG["unflagged_qc_random_seed"],
    )

    out_dir = Path(CONFIG["output_dir"])
    sample_path = out_dir / "unflagged_qc_sample.csv"
    sampled.to_csv(sample_path, index=False)
    print(f"[render_unflagged_qc] sampled {n}/{len(unflagged)} unflagged subjects -> {sample_path}")

    render_subjects(
        sampled[id_col].tolist(),
        fwhm,
        render_dir_name=CONFIG["unflagged_qc_render_dir_name"],
        log_label="render_unflagged_qc",
    )


# =============================================================================
# Stage 2: propagate scores to the other FWHM levels
# =============================================================================


def propagate_scores(scores=None):
    """
    Merge subject_scores.csv onto the single shared meta table, producing one
    meta_annotated.csv for reference/inspection. (Since there is only one
    meta table, this is a straightforward merge - the FWHM-scoping decision
    that matters happens in apply_final_filter, via
    CONFIG["propagate_to_other_fwhm"].)
    """
    out_dir = Path(CONFIG["output_dir"])
    scores_path = out_dir / "subject_scores.csv"
    if scores is None:
        if not scores_path.exists():
            raise FileNotFoundError(f"{scores_path} not found - run run_selection() first")
        scores = pd.read_csv(scores_path)

    id_col = CONFIG["id_col"]
    score_cols = [c for c in scores.columns if c != id_col]
    meta = get_meta()

    annotated = meta.merge(scores[[id_col] + score_cols], on=id_col, how="left")
    n_missing = annotated["review_flag"].isna().sum()
    if n_missing:
        warnings.warn(
            f"{n_missing} subjects in the shared meta have no score from "
            f"fwhm{CONFIG['selection_fwhm']} (not present in the selection dataset)"
        )

    out_path = out_dir / "meta_annotated.csv"
    annotated.to_csv(out_path, index=False)
    print(f"[propagate_scores] {out_path}")
    return annotated


# =============================================================================
# Stage 3: apply final filter after manual review
# =============================================================================


def apply_final_filter(reviewed_csv_path):
    """
    reviewed_csv_path: the flagged_for_review.csv, after you've filled in a
    'manual_keep' column (True/False). Unreviewed flagged subjects (blank
    manual_keep) are excluded by default (conservative) with a warning.

    CONFIG["propagate_to_other_fwhm"] controls scope:
      - True  -> filter the .npy arrays for ALL fwhm_levels (both hemis)
      - False -> filter only the .npy arrays for selection_fwhm

    Since there is a single shared meta table, exactly one filtered meta CSV
    is written (the row mask is identical for every hemi/fwhm - it's a
    subject-level decision).
    """
    out_dir = Path(CONFIG["output_dir"])
    scores_path = out_dir / "subject_scores.csv"
    scores = pd.read_csv(scores_path)
    reviewed = pd.read_csv(reviewed_csv_path)

    id_col = CONFIG["id_col"]
    if "manual_keep" not in reviewed.columns:
        raise KeyError("reviewed CSV must contain a 'manual_keep' column (True/False)")

    unresolved = reviewed[reviewed["manual_keep"].map(parse_bool).isna()][id_col].tolist()
    if unresolved:
        warnings.warn(
            f"{len(unresolved)} flagged subjects have no manual_keep decision "
            f"and will be EXCLUDED by default: {unresolved}"
        )

    manual_decisions = reviewed.set_index(id_col)["manual_keep"]

    def keep_subject(row):
        if not row["review_flag"]:
            return True
        return parse_bool(manual_decisions.get(row[id_col], np.nan)) is True

    scores = scores.copy()
    scores["final_keep"] = scores.apply(keep_subject, axis=1)
    keep_ids = set(scores.loc[scores["final_keep"], id_col])
    print(f"[apply_final_filter] keeping {len(keep_ids)}/{len(scores)} subjects overall")

    filtered_dir = out_dir / "filtered_dataset"
    filtered_dir.mkdir(parents=True, exist_ok=True)

    meta = get_meta()
    keep_mask = meta[id_col].isin(keep_ids).values
    meta_filt = meta[keep_mask].reset_index(drop=True)
    meta_filt.to_csv(filtered_dir / "meta.csv", index=False)
    print(f"[apply_final_filter] meta: {keep_mask.sum()}/{len(meta)} kept -> "
          f"{filtered_dir / 'meta.csv'}")

    fwhm_scope = (CONFIG["fwhm_levels"] if CONFIG["propagate_to_other_fwhm"]
                  else [CONFIG["selection_fwhm"]])
    for fwhm in fwhm_scope:
        for hemi in CONFIG["hemis"]:
            if not os.path.exists(_zscore_path(hemi, fwhm)):
                print(f"[apply_final_filter] {_zscore_path(hemi, fwhm)} not found - skipped")
                continue
            z = load_zscore_array(hemi, fwhm)
            np.save(filtered_dir / f"zscores.fwhm{fwhm}.fsavg5.{hemi}.npy", z[keep_mask])
            print(f"[apply_final_filter] {hemi} fwhm{fwhm}: "
                  f"{keep_mask.sum()}/{len(meta)} kept -> {filtered_dir}")

    if not CONFIG["propagate_to_other_fwhm"]:
        skipped = [f for f in CONFIG["fwhm_levels"] if f != CONFIG["selection_fwhm"]]
        print(f"[apply_final_filter] propagate_to_other_fwhm=False - "
              f"fwhm levels {skipped} left unfiltered")


def parse_bool(value):
    """Interpret a manual_keep entry (True/False, 1/0, yes/no); empty -> None."""
    if pd.isna(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "1.0", "yes", "y"}:
        return True
    if text in {"false", "0", "0.0", "no", "n"}:
        return False
    raise ValueError(f"Cannot interpret manual_keep value {value!r} as True/False")


def main():
    parser = argparse.ArgumentParser(description="Preselection of the TUM cohort.")
    sub = parser.add_subparsers(dest="stage", required=True)
    sub.add_parser("select", help="stage 1: score, flag and render subjects")
    p_filter = sub.add_parser("filter", help="stage 3: apply the manual review")
    p_filter.add_argument("--reviewed", required=True, help="flagged_for_review.csv with manual_keep filled in")
    args = parser.parse_args()

    if args.stage == "select":
        scores = run_selection()
        propagate_scores(scores)
    else:
        apply_final_filter(args.reviewed)


if __name__ == "__main__":
    main()
