"""
Group-mean deviation maps (thinning only) per diagnosis with the core regions
outlined in black, to check that the chosen core regions fit the data.
Paths and settings come from preselect_atrophy_pattern.CONFIG.

Output: outputs/tum_preselection/core_roi_check/{diagnosis}_group_mean_vs_core_roi.png
Usage:  python plot_group_mean_vs_core_roi.py
"""

from pathlib import Path
import nibabel as nib
import pyvista as pv
import numpy as np

import preselect_atrophy_pattern as pap

CONFIG = pap.CONFIG


def group_mean_zscore(hemi, fwhm, diagnosis):
    """Vertex-wise mean z-score across all subjects with this diagnosis."""
    z = pap.load_zscore_array(hemi, fwhm)
    np.minimum(z, 0, out=z)
    meta = pap.get_meta()
    dx_norm = meta[CONFIG["diagnosis_col"]].map(pap._normalize_diagnosis)
    if diagnosis == "AD+PCA":
        mask = dx_norm.isna().values
    else:
        mask = (dx_norm == diagnosis).values
    if mask.sum() == 0:
        raise ValueError(f"No subjects found for diagnosis '{diagnosis}'")
    return z[mask].mean(axis=0), int(mask.sum())


def roi_vertex_mask(hemi, diagnosis):
    """Binary vertex mask: True where the vertex belongs to the diagnosis's core ROI."""
    labels, name_to_label = pap.load_aparc_labels(hemi)
    if diagnosis == "AD+PCA":
        core_names = CONFIG["core_rois"].get("AD", []) + CONFIG["core_rois"].get("PCA", [])
    else:
        core_names = CONFIG["core_rois"].get(diagnosis, [])
    label_vals = [name_to_label[n] for n in core_names if n in name_to_label]
    return np.isin(labels, label_vals)


def render_group_map(diagnosis, fwhm=None, out_dir=None, clim=(-2, 0)):

    fwhm = CONFIG["selection_fwhm"] if fwhm is None else fwhm
    out_dir = Path(out_dir) if out_dir else Path(CONFIG["output_dir"]) / "core_roi_check"
    out_dir.mkdir(parents=True, exist_ok=True)

    meshes, n_subj = {}, None
    for hemi in CONFIG["hemis"]:
        surf_path = Path(CONFIG["fsaverage5_surf_dir"]) / CONFIG["surface_fname_pattern"].format(hemi=hemi)
        verts, faces = nib.freesurfer.read_geometry(str(surf_path))
        faces_pv = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int64)
        mesh = pv.PolyData(verts, faces_pv)
        mesh["z"], n_subj = group_mean_zscore(hemi, fwhm, diagnosis)
        mesh["roi"] = roi_vertex_mask(hemi, diagnosis).astype(float)
        meshes[hemi] = mesh

    grid_pos = {("lh", "lateral"): (0, 0), ("lh", "medial"): (0, 1),
                ("rh", "lateral"): (1, 0), ("rh", "medial"): (1, 1)}

    plotter = pv.Plotter(shape=(2, 2), off_screen=True, window_size=(1000, 1000), border=False)
    for hemi in CONFIG["hemis"]:
        mesh = meshes[hemi]
        try:
            roi_outline = mesh.contour(isosurfaces=[0.5], scalars="roi")
        except Exception:
            roi_outline = None

        for view in ("lateral", "medial"):
            r, c = grid_pos[(hemi, view)]
            plotter.subplot(r, c)
            plotter.add_mesh(mesh, scalars="z", cmap="coolwarm_r", clim=list(clim),
                              show_scalar_bar=(r == 0 and c == 1))
            if roi_outline is not None and roi_outline.n_points > 0:
                plotter.add_mesh(roi_outline, color="black", line_width=6)
            plotter.camera_position = pap._camera_for_view(mesh, hemi, view)
            plotter.add_text(f"{hemi.upper()} {view}", font_size=10, position="upper_left")

    plotter.add_text(
        f"{diagnosis}  (n={n_subj}, fwhm={fwhm}) \n black outline = literature core ROI",
        font_size=10, position="lower_edge",
    )
    out_png = out_dir / f"{diagnosis}_group_mean_vs_core_roi.png"
    plotter.screenshot(str(out_png))
    plotter.close()
    print(f"[render_group_map] {diagnosis}: n={n_subj} -> {out_png}")
    return out_png


if __name__ == "__main__":
    for dx in CONFIG["core_rois"]:
        render_group_map(dx)
    render_group_map("AD+PCA")
