"""
Shows the core regions of every diagnosis on the fsaverage5 surface
(core parcels in red, parcel borders in black).

Output: outputs/tum_preselection/roi_maps/{diagnosis}_core_roi.png
Usage:  python plot_core_roi_parcels.py
"""

from pathlib import Path

import numpy as np

import preselect_atrophy_pattern as pap

CONFIG = pap.CONFIG

ROI_COLOR = "firebrick"
BG_COLOR = "white"
BORDER_COLOR = "black"


def roi_vertex_mask(hemi, diagnosis):
    """Binary vertex mask: True where the vertex belongs to the diagnosis's core ROI."""
    labels, name_to_label = pap.load_aparc_labels(hemi)
    core_names = CONFIG["core_rois"].get(diagnosis, [])
    label_vals = [name_to_label[n] for n in core_names if n in name_to_label]
    return labels, np.isin(labels, label_vals)


def parcel_boundary_lines(mesh, labels):
    """PolyData of edges where two adjacent vertices belong to different parcels."""
    import pyvista as pv

    faces = mesh.faces.reshape(-1, 4)[:, 1:4]  # (n_faces, 3), triangles only
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = np.sort(edges, axis=1)
    boundary = labels[edges[:, 0]] != labels[edges[:, 1]]
    boundary_edges = np.unique(edges[boundary], axis=0)
    if boundary_edges.size == 0:
        return None
    lines = np.hstack([np.full((len(boundary_edges), 1), 2), boundary_edges]).astype(np.int64)
    return pv.PolyData(mesh.points, lines=lines)


def render_diagnosis_roi(diagnosis, out_dir=None):
    import nibabel as nib
    import pyvista as pv

    out_dir = Path(out_dir) if out_dir else Path(CONFIG["output_dir"]) / "roi_maps"
    out_dir.mkdir(parents=True, exist_ok=True)

    meshes, masks, boundaries = {}, {}, {}
    for hemi in CONFIG["hemis"]:
        surf_path = Path(CONFIG["fsaverage5_surf_dir"]) / CONFIG["surface_fname_pattern"].format(hemi=hemi)
        verts, faces = nib.freesurfer.read_geometry(str(surf_path))
        faces_pv = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int64)
        mesh = pv.PolyData(verts, faces_pv)

        labels, mask = roi_vertex_mask(hemi, diagnosis)
        mesh["roi"] = mask.astype(float)
        meshes[hemi] = mesh
        masks[hemi] = mask
        boundaries[hemi] = parcel_boundary_lines(mesh, labels)

    grid_pos = {("lh", "lateral"): (0, 0), ("lh", "medial"): (0, 1),
                ("rh", "lateral"): (1, 0), ("rh", "medial"): (1, 1)}

    plotter = pv.Plotter(shape=(2, 2), off_screen=True, window_size=(1000, 1000), border=False)
    for hemi in CONFIG["hemis"]:
        mesh = meshes[hemi]
        for view in ("lateral", "medial"):
            r, c = grid_pos[(hemi, view)]
            plotter.subplot(r, c)
            plotter.add_mesh(mesh, scalars="roi", cmap=[BG_COLOR, ROI_COLOR], clim=[0, 1],
                              show_scalar_bar=False)
            if boundaries[hemi] is not None:
                plotter.add_mesh(boundaries[hemi], color=BORDER_COLOR, line_width=1)
            plotter.camera_position = pap._camera_for_view(mesh, hemi, view)
            plotter.add_text(f"{hemi.upper()} {view}", font_size=10, position="upper_left")

    rois = CONFIG["core_rois"].get(diagnosis, [])
    core_names = ",\n".join(", ".join(rois[i : i + 3]) for i in range(0, len(rois), 3))
    plotter.add_text(f"{diagnosis} core ROI: {core_names}", font_size=13, position="lower_left")

    out_png = out_dir / f"{diagnosis}_core_roi.png"
    plotter.screenshot(str(out_png))
    plotter.close()
    print(f"[render_diagnosis_roi] {diagnosis} -> {out_png}")
    return out_png


if __name__ == "__main__":
    for dx in CONFIG["core_rois"]:
        render_diagnosis_roi(dx)
