"""
Resample FreeSurfer cortical thickness to fsaverage6 / fsaverage5.

For every subject listed in the ID file, the native-space thickness map
(surf/{hemi}.thickness of a FreeSurfer 7.2 subject directory) is mapped to
fsaverage6 with ``mri_surf2surf`` and smoothed on the target surface with a
Gaussian kernel (default FWHM = 5 mm, matching the Ecker et al. expression
maps). Because the icosahedral fsaverage surfaces are nested, the first
10,242 fsaverage6 vertices are exactly the fsaverage5 (ico5) vertices, so
the fsaverage5 map is obtained by truncation.

If a subject has no ``label/{hemi}.cortex.label``, it is created from the
subject's aparc annotation (all vertices not labeled "unknown") so that
``--cortex`` can restrict smoothing to the cortex.

Inputs
    --subjects-dir   one or more FreeSurfer SUBJECTS_DIRs (searched in order)
    --id-file        CSV with an IMAGEUID column (FreeSurfer subject folder
                     name) and optionally PTID and DX columns
    --output-dir     output folder
    --freesurfer-home  FreeSurfer installation (default: $FREESURFER_HOME)

Outputs (HDF5, one group per subject, datasets "lh", "rh" and optionally "dx")
    thickness.fwhm{FWHM}.fsavg6.h5   (40,962 vertices per hemisphere)
    thickness.fwhm{FWHM}.fsavg5.h5   (10,242 vertices per hemisphere)

Subjects already present in the output file are skipped, so interrupted
runs can be resumed. The fsaverage5 tables were then exported to the
row-per-subject input tables used by SCSR (03_scsr).

Usage
    python resample_thickness.py --subjects-dir /path/to/subjects \
        --id-file meta.csv --output-dir out/ [--fwhm 5]
"""

import argparse
import contextlib
import os
import subprocess
from pathlib import Path

import h5py
import nibabel as nib
import nibabel.freesurfer.io as fsio
import numpy as np
import pandas as pd
from tqdm import tqdm

HEMIS = ["lh", "rh"]
MORPH = "thickness"
N_ICO6 = 40962
N_ICO5 = 10242

DX_MAPS = {
    "adni": {"CN": 0, "MCI": 1, "Dementia": 2},
    "tum": {"CN": 0, "FTD": 1, "AD": 2},
    "tum_342": {"CN": 0, "FTD": 1, "AD": 2},
    "cth_diff": {"HC": 0, "FTD": 1, "AD": 2, "PCA": 3, "SD": 4},
}


def parse_id_file(id_file: Path):
    """Return (image_uids, uid -> PTID, uid -> integer DX code or None)."""
    df = pd.read_csv(id_file, dtype=str)
    df.columns = [c.strip() for c in df.columns]

    uid_col = df["IMAGEUID"] if "IMAGEUID" in df.columns else df.iloc[:, 0]
    imageuids = uid_col.str.strip().tolist()
    if {"IMAGEUID", "PTID"} <= set(df.columns):
        uid_to_ptid = dict(zip(df["IMAGEUID"].str.strip(), df["PTID"].str.strip()))
    else:
        uid_to_ptid = {uid: uid for uid in imageuids}

    uid_to_dx = None
    if {"IMAGEUID", "DX"} <= set(df.columns):
        dx_map = DX_MAPS.get(id_file.stem.lower())
        if dx_map is None:
            print(f'[warn] DX column present but "{id_file.stem}" not in DX_MAPS - DX not stored')
        else:
            uid_to_dx = {}
            for _, row in df.iterrows():
                uid, label = row["IMAGEUID"].strip(), row["DX"].strip()
                if label not in dx_map:
                    raise ValueError(f"Unknown DX label {label!r} for dataset {id_file.stem}")
                uid_to_dx[uid] = dx_map[label]

    return imageuids, uid_to_ptid, uid_to_dx


def find_subjects_dir(subj: str, subjects_dirs):
    for d in subjects_dirs:
        if (Path(d) / subj).exists():
            return d
    return None


def ensure_cortex_label(subj: str, subjects_dir: str, hemi: str) -> bool:
    """Create label/{hemi}.cortex.label from {hemi}.aparc.annot if it does not exist."""
    subj_dir = Path(subjects_dir) / subj
    label_path = subj_dir / "label" / f"{hemi}.cortex.label"
    if label_path.exists():
        return True

    annot_path = subj_dir / "label" / f"{hemi}.aparc.annot"
    surf_path = subj_dir / "surf" / f"{hemi}.white"
    if not annot_path.exists() or not surf_path.exists():
        print(f"  WARN: cannot create cortex label for {subj} {hemi} - annot or surf missing")
        return False

    labels, _, names = fsio.read_annot(str(annot_path))
    names_str = [n.decode() if isinstance(n, bytes) else n for n in names]
    if "unknown" in names_str:
        cortex_verts = np.where(labels != names_str.index("unknown"))[0]
    else:
        cortex_verts = np.where(labels >= 0)[0]

    coords, _ = fsio.read_geometry(str(surf_path))
    with open(label_path, "w") as f:
        f.write("#!ascii label\n")
        f.write(f"{len(cortex_verts)}\n")
        for v in cortex_verts:
            x, y, z = coords[v]
            f.write(f"{v}  {x:.3f}  {y:.3f}  {z:.3f}  0.000000\n")
    return True


def resample_one(subj: str, hemi: str, args):
    """Map one subject/hemisphere to fsaverage6 with mri_surf2surf; returns (40962,) array or None."""
    subjects_dir = find_subjects_dir(subj, args.subjects_dir)
    if subjects_dir is None:
        print(f"  ERROR: {subj} not found in any SUBJECTS_DIR")
        return None

    ensure_cortex_label(subj, subjects_dir, hemi)

    fs_home = Path(args.freesurfer_home)
    fs_bin = fs_home / "bin"
    env = os.environ.copy()
    env["FREESURFER_HOME"] = str(fs_home)
    env["SUBJECTS_DIR"] = subjects_dir
    env["PATH"] = str(fs_bin) + ":" + env.get("PATH", "")

    tmp_path = Path(args.output_dir) / f"_tmp_{subj}_{hemi}.mgh"
    result = subprocess.run([
        str(fs_bin / "mri_surf2surf"),
        "--srcsubject", subj,
        "--trgsubject", "fsaverage6",
        "--hemi", hemi,
        "--sval", MORPH,
        "--src_type", "curv",
        "--tval", str(tmp_path),
        "--trg_type", "mgh",
        "--fwhm-trg", str(args.fwhm),
        "--cortex",
    ], env=env, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  ERROR: {subj} {hemi}\n{result.stderr[-800:]}")
        tmp_path.unlink(missing_ok=True)
        return None

    data = nib.load(tmp_path).get_fdata(dtype=np.float32).flatten()
    tmp_path.unlink(missing_ok=True)
    if len(data) != N_ICO6:
        print(f"  WARN: {subj} {hemi}: expected {N_ICO6} vertices, got {len(data)}")
        return None
    return data


def parse_args():
    p = argparse.ArgumentParser(description="Resample cortical thickness to fsaverage6/fsaverage5.")
    p.add_argument("--subjects-dir", nargs="+", required=True)
    p.add_argument("--id-file", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--freesurfer-home", default=os.environ.get("FREESURFER_HOME", "/usr/local/freesurfer"))
    p.add_argument("--fwhm", type=float, default=5)
    p.add_argument("--no-ico5", action="store_true", help="do not write the fsaverage5 file")
    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fwhm_tag = int(args.fwhm) if float(args.fwhm).is_integer() else args.fwhm
    out_h5 = args.output_dir / f"thickness.fwhm{fwhm_tag}.fsavg6.h5"
    out_h5_ico5 = args.output_dir / f"thickness.fwhm{fwhm_tag}.fsavg5.h5"
    save_ico5 = not args.no_ico5

    subjects, uid_to_ptid, uid_to_dx = parse_id_file(args.id_file)
    print(f"Processing {len(subjects)} subjects, morph={MORPH}, fwhm={args.fwhm} mm -> fsaverage6")

    failed = []
    ctx_ico5 = h5py.File(out_h5_ico5, "a") if save_ico5 else contextlib.nullcontext()
    with h5py.File(out_h5, "a") as hf, ctx_ico5 as hf5:
        for uid in tqdm(subjects, desc="Subjects"):
            ptid = uid_to_ptid.get(uid, uid)
            if ptid in hf and all(h in hf[ptid] for h in HEMIS):
                continue

            arrays = {}
            for hemi in HEMIS:
                data = resample_one(uid, hemi, args)
                if data is None:
                    failed.append(f"{uid}/{hemi}")
                    break
                arrays[hemi] = data
            if len(arrays) != len(HEMIS):
                continue

            targets = [(hf, None)] + ([(hf5, N_ICO5)] if save_ico5 else [])
            for h5, n_vert in targets:
                grp = h5.require_group(ptid)
                for hemi, arr in arrays.items():
                    if hemi not in grp:
                        grp.create_dataset(hemi, data=arr[:n_vert] if n_vert else arr, compression="gzip")
                if uid_to_dx is not None and uid in uid_to_dx and "dx" not in grp:
                    grp.create_dataset("dx", data=np.int8(uid_to_dx[uid]))

    print(f"\nFinished. {len(subjects) * len(HEMIS) - len(failed)} succeeded, {len(failed)} failed.")
    if failed:
        print("Failed:", ", ".join(failed))


if __name__ == "__main__":
    main()
