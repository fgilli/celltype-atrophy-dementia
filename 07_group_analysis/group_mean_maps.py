"""
Group-average SCSR deviation maps on the fsaverage5 surface.

For every diagnostic group (column --group-col of the metadata), the
vertex-wise mean of the thinning-only deviation maps, min(z, 0), is
rendered in lateral and medial views of both hemispheres (color range
-2 to 0, FWHM = 5 mm maps).

Inputs
    deviation maps   data/deviation_maps/{cohort}/zscores.fwhm5.fsavg5.{lh,rh}.npy
    metadata         data/metadata/meta_{COHORT}.csv (row-aligned)
    surfaces         data/fsaverage5/surf/{lh,rh}.pial

Output
    outputs/{cohort}/group_mean_maps/{group}_group_mean.png

Usage
    python group_mean_maps.py --profile tum --group-col Subtype
    python group_mean_maps.py --profile nacc --group-col DX_family
"""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
OUTPUTS = REPO / "outputs"

PROFILES = {
    "tum": {"zscore_dir": DATA / "deviation_maps" / "tum", "meta_path": DATA / "metadata" / "meta_TUM.csv"},
    "nifd": {"zscore_dir": DATA / "deviation_maps" / "nifd", "meta_path": DATA / "metadata" / "meta_NIFD.csv"},
    "nacc": {"zscore_dir": DATA / "deviation_maps" / "nacc", "meta_path": DATA / "metadata" / "meta_NACC.csv"},
}

CONFIG = {
    "fwhm": 5,
    "surf_dir": DATA / "fsaverage5" / "surf",
    "surface_fname_pattern": "{hemi}.pial",
    "hemis": ["lh", "rh"],
    "clim": (-2, 0),
    # flip to -1 if lateral and medial views come out swapped
    "render_lateral_sign": 1,
    "window_size": (1000, 1000),
}


def load_data(cfg, group_col):
    zscores = {hemi: np.load(Path(cfg["zscore_dir"]) / f"zscores.fwhm{CONFIG['fwhm']}.fsavg5.{hemi}.npy")
               for hemi in CONFIG["hemis"]}
    meta = pd.read_csv(cfg["meta_path"])
    for hemi, z in zscores.items():
        if z.shape[0] != len(meta):
            raise ValueError(f"Row mismatch: {hemi}={z.shape[0]}, meta={len(meta)}")
    if group_col not in meta.columns:
        raise KeyError(f"Column '{group_col}' not found in {cfg['meta_path']}")
    return zscores, meta


def group_mean_zscore(z, mask):
    """Vertex-wise mean over the subjects in mask, thinning only (z clipped to <= 0)."""
    return np.minimum(z, 0)[mask].mean(axis=0)


def camera_for_view(mesh, hemi, view):
    """Camera on the x-axis (FreeSurfer RAS) looking at the lateral or medial surface."""
    center = np.array(mesh.center)
    b = mesh.bounds
    dist = 2.0 * np.linalg.norm([b[1] - b[0], b[3] - b[2], b[5] - b[4]])
    hemi_sign = -1 if hemi == "lh" else 1
    direction = (hemi_sign if view == "lateral" else -hemi_sign) * CONFIG["render_lateral_sign"]
    return [tuple(center + np.array([direction * dist, 0, 0])), tuple(center), (0, 0, 1)]


def render_group_map(label, zscores, mask, out_dir):
    import pyvista as pv

    n_subj = int(mask.sum())
    meshes = {}
    for hemi in CONFIG["hemis"]:
        surf_path = Path(CONFIG["surf_dir"]) / CONFIG["surface_fname_pattern"].format(hemi=hemi)
        verts, faces = nib.freesurfer.read_geometry(str(surf_path))
        mesh = pv.PolyData(verts, np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int64))
        mesh["z"] = group_mean_zscore(zscores[hemi], mask)
        meshes[hemi] = mesh

    grid_pos = {("lh", "lateral"): (0, 0), ("lh", "medial"): (0, 1),
                ("rh", "lateral"): (1, 0), ("rh", "medial"): (1, 1)}
    plotter = pv.Plotter(shape=(2, 2), off_screen=True, window_size=CONFIG["window_size"], border=False)
    for hemi, mesh in meshes.items():
        for view in ("lateral", "medial"):
            r, c = grid_pos[(hemi, view)]
            plotter.subplot(r, c)
            plotter.add_mesh(mesh, scalars="z", cmap="coolwarm_r", clim=list(CONFIG["clim"]),
                             show_scalar_bar=(r == 0 and c == 1))
            plotter.camera_position = camera_for_view(mesh, hemi, view)
            plotter.add_text(f"{hemi.upper()} {view}", font_size=10, position="upper_left")
    plotter.add_text(f"{label}  (n={n_subj})", font_size=10, position="lower_edge")

    out_png = out_dir / f"{label}_group_mean.png"
    plotter.screenshot(str(out_png))
    plotter.close()
    for hemi, mesh in meshes.items():
        np.save(out_dir / f"{label}_{hemi}_group_mean.npy", np.asarray(mesh["z"]))
    print(f"{label}: n={n_subj} -> {out_png}")


def main():
    parser = argparse.ArgumentParser(description="Group-average deviation maps.")
    parser.add_argument("--profile", choices=sorted(PROFILES), required=True)
    parser.add_argument("--group-col", default="DX_family", help="metadata column defining the groups")
    parser.add_argument("--groups", nargs="*", default=None, help="subset of group labels (default: all)")
    args = parser.parse_args()

    cfg = PROFILES[args.profile]
    zscores, meta = load_data(cfg, args.group_col)
    out_dir = OUTPUTS / args.profile / "group_mean_maps"
    out_dir.mkdir(parents=True, exist_ok=True)

    groups = args.groups or sorted(meta[args.group_col].dropna().unique().tolist())
    for label in groups:
        mask = (meta[args.group_col] == label).values
        if mask.sum() == 0:
            print(f"No subjects for '{label}' - skipped")
            continue
        render_group_map(label, zscores, mask, out_dir)


if __name__ == "__main__":
    main()
