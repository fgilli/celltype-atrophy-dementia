"""
Reference render of the Desikan-Killiany parcellation on fsaverage5.

Every parcel gets its own color and is labeled with its name (lateral and
medial views of both hemispheres, labels only for parcels facing the
camera), plus unlabeled anterior, posterior, superior and inferior views of
the whole brain. Used as a look-up chart when defining the core ROIs of the
TUM preselection. Paths and settings are taken from
preselect_atrophy_pattern.CONFIG.

Output
    {CONFIG["output_dir"]}/dk_atlas_reference.png

Usage
    python plot_dk_atlas.py
"""

from pathlib import Path

import numpy as np
import matplotlib
from matplotlib.colors import ListedColormap, to_hex

import preselect_atrophy_pattern as pap

CONFIG = pap.CONFIG
LABEL_FONT_SIZE = 18


def _shared_parcel_names():
    """Union of parcel names across hemispheres, minus excluded parcels."""
    names = set()
    for hemi in CONFIG["hemis"]:
        _, name_to_label = pap.load_aparc_labels(hemi)
        names |= set(name_to_label.keys())
    exclude = set(CONFIG["exclude_parcels"])
    return sorted(n for n in names if n not in exclude)


def _distinct_colors(n):
    cmap = matplotlib.colormaps["nipy_spectral"].resampled(n)
    return [to_hex(cmap(i)) for i in range(n)]


def _load_hemi_mesh(hemi):
    import nibabel as nib
    import pyvista as pv

    surf_path = Path(CONFIG["fsaverage5_surf_dir"]) / CONFIG["surface_fname_pattern"].format(hemi=hemi)
    verts, faces = nib.freesurfer.read_geometry(str(surf_path))
    faces_pv = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int64)
    mesh = pv.PolyData(verts, faces_pv)
    mesh.compute_normals(point_normals=True, cell_normals=False, inplace=True)
    labels, name_to_label = pap.load_aparc_labels(hemi)
    return mesh, labels, name_to_label


def _color_mesh_by_parcel(mesh, labels, name_to_label, name_to_idx):
    """Assign each vertex a parcel color-index (NaN for excluded parcels)."""
    lookup = np.full(int(labels.max()) + 1, -1, dtype=int)
    for name, lbl in name_to_label.items():
        if name in name_to_idx:
            lookup[lbl] = name_to_idx[name]
    idx = lookup[labels].astype(float)
    idx[idx < 0] = np.nan
    mesh["parcel"] = idx


def _parcel_label_anchors(mesh, labels, name_to_label):
    """
    One on-surface anchor point per parcel, used for both the label position
    and the facing-camera test. Using the plain mean of member vertices can
    land inside a fold for buried regions (e.g. transversetemporal, tucked
    into the Sylvian fissure) since the mean of two opposing fold walls is
    an interior point, not a surface point. Instead we use the *medoid*:
    the actual member vertex closest to the mean, which is guaranteed to
    lie on the mesh surface.
    """
    exclude = set(CONFIG["exclude_parcels"])
    anchors = {}
    for name, lbl in name_to_label.items():
        if name in exclude:
            continue
        mask = labels == lbl
        if not mask.any():
            continue
        pts = mesh.points[mask]
        mean_pos = pts.mean(axis=0)
        medoid = pts[np.linalg.norm(pts - mean_pos, axis=1).argmin()]
        anchors[name] = medoid
    return anchors


def _facing_parcel_names(mesh, labels, name_to_label, direction_sign):
    """
    Names of parcels whose average vertex-normal points towards the camera
    for a lateral/medial view (camera looks along the x-axis only), so each
    panel only labels the side of the hemisphere actually visible there.
    """
    exclude = set(CONFIG["exclude_parcels"])
    normals = mesh.point_data["Normals"]
    facing = []
    for name, lbl in name_to_label.items():
        if name in exclude:
            continue
        mask = labels == lbl
        if not mask.any():
            continue
        mean_normal_x = normals[mask, 0].mean()
        if mean_normal_x * direction_sign > 0:
            facing.append(name)
    return set(facing)


def _parcel_boundary_lines(mesh, labels):
    """PolyData of edges where two adjacent vertices belong to different parcels."""
    import pyvista as pv

    faces = mesh.faces.reshape(-1, 4)[:, 1:4]
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = np.sort(edges, axis=1)
    boundary = labels[edges[:, 0]] != labels[edges[:, 1]]
    boundary_edges = np.unique(edges[boundary], axis=0)
    if boundary_edges.size == 0:
        return None
    lines = np.hstack([np.full((len(boundary_edges), 1), 2), boundary_edges]).astype(np.int64)
    return pv.PolyData(mesh.points, lines=lines)


def _camera(center, bounds, direction, view_up):
    diag = np.linalg.norm([bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]])
    dist = diag * 2.0
    cam_loc = np.array(center) + np.array(direction) * dist
    return [tuple(cam_loc), tuple(center), view_up]


def _direction_sign(hemi, view):
    """Mirrors preselect_atrophy_pattern.py's _camera_for_view sign logic."""
    hemi_sign = -1 if hemi == "lh" else 1
    direction_sign = hemi_sign if view == "lateral" else -hemi_sign
    direction_sign *= CONFIG["render_lateral_sign"]
    return direction_sign


def render_atlas_reference(out_path=None):
    import pyvista as pv

    out_path = Path(out_path) if out_path else Path(CONFIG["output_dir"]) / "dk_atlas_reference.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    parcel_names = _shared_parcel_names()
    name_to_idx = {n: i for i, n in enumerate(parcel_names)}
    cmap = ListedColormap(_distinct_colors(len(parcel_names)))

    hemi_data = {}
    for hemi in CONFIG["hemis"]:
        mesh, labels, name_to_label = _load_hemi_mesh(hemi)
        _color_mesh_by_parcel(mesh, labels, name_to_label, name_to_idx)
        hemi_data[hemi] = dict(
            mesh=mesh,
            labels=labels,
            name_to_label=name_to_label,
            boundary=_parcel_boundary_lines(mesh, labels),
            centroids=_parcel_label_anchors(mesh, labels, name_to_label),
        )

    meshes = [hemi_data[h]["mesh"] for h in CONFIG["hemis"]]
    combined = meshes[0].merge(meshes[1:]) if len(meshes) > 1 else meshes[0]
    center, bounds = np.array(combined.center), combined.bounds

    # panel definitions: per-hemi lateral/medial views, then whole-brain angles
    panels = []
    for hemi in CONFIG["hemis"]:
        panels.append((f"{hemi.upper()} lateral", hemi, "lateral", None))
        panels.append((f"{hemi.upper()} medial", hemi, "medial", None))
    panels += [
        ("Anterior", None, (0, 1, 0), (0, 0, 1)),
        ("Posterior", None, (0, -1, 0), (0, 0, 1)),
        ("Superior", None, (0, 0, 1), (0, 1, 0)),
        ("Inferior", None, (0, 0, -1), (0, -1, 0)),
    ]

    ncols = 4
    nrows = int(np.ceil(len(panels) / ncols))
    plotter = pv.Plotter(shape=(nrows, ncols), off_screen=True,
                          window_size=(ncols * 900, nrows * 900), border=False)
    plotter.enable_anti_aliasing("fxaa")  # cheap shader-based AA, safer than SSAA off-screen

    for i, (title, hemi, a, b) in enumerate(panels):
        r, c = divmod(i, ncols)
        plotter.subplot(r, c)

        if hemi is not None:  # per-hemisphere lateral/medial panel, with labels
            view = a
            d = hemi_data[hemi]
            plotter.add_mesh(d["mesh"], scalars="parcel", cmap=cmap,
                              clim=[0, len(parcel_names) - 1], nan_color="lightgray",
                              show_scalar_bar=False, smooth_shading=True)
            if d["boundary"] is not None:
                plotter.add_mesh(d["boundary"], color="black", line_width=1.5)

            # set the camera BEFORE adding labels so any occlusion / clipping
            # PyVista does internally uses the correct view
            plotter.camera_position = pap._camera_for_view(d["mesh"], hemi, view)

            dsign = _direction_sign(hemi, view)
            facing = _facing_parcel_names(d["mesh"], d["labels"], d["name_to_label"], dsign)
            wanted = {n: xyz for n, xyz in d["centroids"].items() if n in facing}
            if wanted:
                points = np.array(list(wanted.values()))
                names = list(wanted.keys())
                plotter.add_point_labels(
                    points, names,
                    font_size=LABEL_FONT_SIZE, text_color="black",
                    shape="rounded_rect", shape_color="white", shape_opacity=0.55,
                    show_points=True, point_color="black", point_size=8,
                    always_visible=True,   # don't rely on PyVista's occlusion test
                    render=False,
                )
        else:  # whole-brain angle panel, both hemis, no labels (would be too dense)
            direction, view_up = a, b
            for h in CONFIG["hemis"]:
                d = hemi_data[h]
                plotter.add_mesh(d["mesh"], scalars="parcel", cmap=cmap,
                                  clim=[0, len(parcel_names) - 1], nan_color="lightgray",
                                  show_scalar_bar=False, smooth_shading=True)
                if d["boundary"] is not None:
                    plotter.add_mesh(d["boundary"], color="black", line_width=1.5)
            plotter.camera_position = _camera(center, bounds, direction, view_up)

        plotter.add_text(title, font_size=16, position="upper_left")

    plotter.screenshot(str(out_path))
    plotter.close()
    print(f"[render_atlas_reference] {len(parcel_names)} parcels -> {out_path}")
    return out_path


if __name__ == "__main__":
    render_atlas_reference()
