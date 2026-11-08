"""
Surface smoothing of the SCSR deviation maps on fsaverage5.

The unsmoothed deviation maps produced by sample_scsr.py are smoothed on
the fsaverage5 surface with FreeSurfer's ``mri_surf2surf`` (Gaussian
kernel, default FWHM = 5 mm). Smoothing is restricted to the cortex
label without the "unknown" (medial wall) and "corpuscallosum" parcels,
so values from the excluded medial wall do not bleed into the cortex.

Inputs
    --input-npy      lh and rh .npy arrays (n_subjects, 10242), unsmoothed
    --id-file        CSV row-aligned with the arrays; the subject key is
                     taken from PTID, IMAGEUID or new_id (first one present)
    --output-dir     output folder
    --freesurfer-home  FreeSurfer installation (default: $FREESURFER_HOME);
                     fsaverage5 is read from $FREESURFER_HOME/subjects

Output
    zscores.fwhm{FWHM}.fsavg5.h5 with one group per subject and the datasets
    "lh" and "rh" (10,242 vertices each). Subjects already in the file are
    skipped. The smoothed maps were exported to
    zscores.fwhm{FWHM}.fsavg5.{hemi}.npy arrays (rows in metadata order) for
    the downstream analysis.

Usage
    python smooth_deviation_maps.py --input-npy lh.npy rh.npy \
        --id-file meta.csv --output-dir out/ [--fwhm 5]
"""

import argparse
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
N_ICO5 = 10242
EXCLUDE_PARCELS = {b"corpuscallosum", b"unknown"}


def read_subject_ids(id_file: Path):
    df = pd.read_csv(id_file, dtype=str)
    df.columns = [c.strip() for c in df.columns]
    for col in ("PTID", "IMAGEUID", "new_id"):
        if col in df.columns:
            return df[col].str.strip().tolist()
    raise KeyError(f"{id_file} needs one of the columns PTID, IMAGEUID or new_id")


def make_cortex_label_without_cc(fs_home: Path, hemi: str, out_dir: Path):
    """Write an fsaverage5 cortex label without the 'unknown' and 'corpuscallosum' vertices."""
    fs5 = fs_home / "subjects" / "fsaverage5"
    label_path = fs5 / "label" / f"{hemi}.cortex.label"
    annot_path = fs5 / "label" / f"{hemi}.aparc.annot"
    if not label_path.exists() or not annot_path.exists():
        return None

    cortex_verts = fsio.read_label(str(label_path))
    per_vertex_labels, _, names = fsio.read_annot(str(annot_path))
    coords, _ = fsio.read_geometry(str(fs5 / "surf" / f"{hemi}.white"))

    exclude_idx = [i for i, n in enumerate(names) if n in EXCLUDE_PARCELS]
    excluded_verts = np.where(np.isin(per_vertex_labels, exclude_idx))[0]
    kept = cortex_verts[~np.isin(cortex_verts, excluded_verts)]

    out_path = out_dir / f"_tmp_{hemi}.cortex_no_cc.label"
    with open(out_path, "w") as f:
        f.write("#!ascii label\n")
        f.write(f"{len(kept)}\n")
        for v in kept:
            x, y, z = coords[v]
            f.write(f"{v}  {x:.3f}  {y:.3f}  {z:.3f}  0.000000\n")
    return out_path


def smooth_one(subj: str, hemi: str, values: np.ndarray, label_path, args):
    """Smooth one (10242,) map on fsaverage5; returns the smoothed map or None."""
    fs_home = Path(args.freesurfer_home)
    fs_bin = fs_home / "bin"
    env = os.environ.copy()
    env["FREESURFER_HOME"] = str(fs_home)
    env["SUBJECTS_DIR"] = str(fs_home / "subjects")
    env["PATH"] = str(fs_bin) + ":" + env.get("PATH", "")

    src_path = args.output_dir / f"_tmp_{subj}_{hemi}.zscores"
    out_path = args.output_dir / f"_tmp_{subj}_{hemi}.mgh"
    fsio.write_morph_data(str(src_path), values.astype(np.float32))

    cortex_args = (["--label-src", str(label_path), "--label-trg", str(label_path)]
                   if label_path is not None else ["--cortex"])
    result = subprocess.run([
        str(fs_bin / "mri_surf2surf"),
        "--srcsubject", "fsaverage5",
        "--trgsubject", "fsaverage5",
        "--hemi", hemi,
        "--sval", str(src_path),
        "--src_type", "curv",
        "--tval", str(out_path),
        "--trg_type", "mgh",
        "--fwhm-trg", str(args.fwhm),
        *cortex_args,
    ], env=env, capture_output=True, text=True)
    src_path.unlink(missing_ok=True)

    if result.returncode != 0:
        print(f"  ERROR: {subj} {hemi}\n{result.stdout[-800:]}\n{result.stderr[-800:]}")
        out_path.unlink(missing_ok=True)
        return None

    data = nib.load(out_path).get_fdata(dtype=np.float32).flatten()
    out_path.unlink(missing_ok=True)
    if len(data) != N_ICO5:
        print(f"  WARN: {subj} {hemi}: expected {N_ICO5} vertices, got {len(data)}")
        return None
    return data


def parse_args():
    p = argparse.ArgumentParser(description="Smooth deviation maps on fsaverage5.")
    p.add_argument("--input-npy", nargs=2, required=True, metavar=("LH", "RH"))
    p.add_argument("--id-file", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--freesurfer-home", default=os.environ.get("FREESURFER_HOME", "/usr/local/freesurfer"))
    p.add_argument("--fwhm", type=float, default=5)
    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fwhm_tag = int(args.fwhm) if float(args.fwhm).is_integer() else args.fwhm
    out_h5 = args.output_dir / f"zscores.fwhm{fwhm_tag}.fsavg5.h5"

    subjects = read_subject_ids(args.id_file)
    maps = {hemi: np.load(path) for hemi, path in zip(HEMIS, args.input_npy)}
    for hemi, arr in maps.items():
        if arr.shape != (len(subjects), N_ICO5):
            raise ValueError(f"{hemi} array has shape {arr.shape}, expected ({len(subjects)}, {N_ICO5})")

    labels = {hemi: make_cortex_label_without_cc(Path(args.freesurfer_home), hemi, args.output_dir)
              for hemi in HEMIS}
    print(f"Smoothing {len(subjects)} subjects with FWHM = {args.fwhm} mm on fsaverage5")

    failed = []
    with h5py.File(out_h5, "a") as hf:
        for i, subj in enumerate(tqdm(subjects, desc="Subjects")):
            if subj in hf and all(h in hf[subj] for h in HEMIS):
                continue
            arrays = {}
            for hemi in HEMIS:
                data = smooth_one(subj, hemi, maps[hemi][i], labels[hemi], args)
                if data is None:
                    failed.append(f"{subj}/{hemi}")
                    break
                arrays[hemi] = data
            if len(arrays) != len(HEMIS):
                continue
            grp = hf.require_group(subj)
            for hemi, arr in arrays.items():
                if hemi not in grp:
                    grp.create_dataset(hemi, data=arr, compression="gzip")

    for path in labels.values():
        if path is not None:
            path.unlink(missing_ok=True)

    print(f"\nFinished. {len(subjects) * len(HEMIS) - len(failed)} succeeded, {len(failed)} failed.")
    if failed:
        print("Failed:", ", ".join(failed))


if __name__ == "__main__":
    main()
