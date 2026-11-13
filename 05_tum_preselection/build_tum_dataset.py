"""
Puts the TUM cohort together from its three source datasets (TUM_342,
small_ds, CTh_train), which partly contain the same subjects.

total.csv lists every subject once (column "id"). Each id is looked up in
TUM_342 first (PTID without "_anat_hc" / "_ses-<n>_T1w"; "#N" ids are
"NM_FTD_{N:02d}"), then in small_ds and then in CTh_train, and the matching
row of the deviation maps is copied. Ids that are not found are written to
tum_unmatched.csv.

Output (data/tum_sources/combined):
    zscores.fwhm{fwhm}.fsavg5.{hemi}.npy
    tum_rows.fwhm{fwhm}.fsavg5.{hemi}.npy    [total_id, dataset, ptid, source_row]

Usage:  python build_tum_dataset.py
"""

import csv
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "tum_sources"

CONFIG = {
    "dataset_dirs": {
        "TUM_342": RAW / "TUM_342",
        "small_ds": RAW / "small_ds",
        "CTh_train": RAW / "CTh_train",
    },
    "meta_paths": {
        "TUM_342": RAW / "tum_342.csv",
        "small_ds": RAW / "cth_diff.csv",
        "CTh_train": RAW / "meta_cth_train.csv",
    },
    "meta_id_column": "PTID",
    "total_csv": RAW / "total.csv",
    "output_dir": REPO / "data" / "tum_sources" / "combined",
    "fwhm_levels": [0, 5, 10, 15],
    "hemis": ["lh", "rh"],
    "npy_name": "zscores.fwhm{fwhm}.fsavg5.{hemi}.npy",
}

# Datasets in matching priority order
PRIORITY = ["TUM_342", "small_ds", "CTh_train"]
TUM_342_SUFFIX_REGEX = re.compile(r"(_anat_hc|_ses-\d+_T1w)$")
HASHTAG_RE = re.compile(r"^#(\d+)$")


def load_csv_rows(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def build_lookups(meta_rows_by_dataset):
    """{dataset: {subject_id: row_index}}; TUM_342 ids are suffix-stripped."""
    lookups = {}
    for name, rows in meta_rows_by_dataset.items():
        lookup = {}
        for idx, row in enumerate(rows):
            key = row[CONFIG["meta_id_column"]]
            if name == "TUM_342":
                key = TUM_342_SUFFIX_REGEX.sub("", key)
            if key in lookup:
                print(f"WARNING: duplicate id '{key}' in {name} (rows {lookup[key]} and {idx}) - keeping first",
                      file=sys.stderr)
                continue
            lookup[key] = idx
        lookups[name] = lookup
    return lookups


def resolve_id(total_id, lookups):
    """Return (dataset, ptid, row_index) for one total.csv id, or None."""
    m = HASHTAG_RE.match(total_id.strip())
    if m:
        ptid = f"NM_FTD_{int(m.group(1)):02d}"
        idx = lookups["TUM_342"].get(ptid)
        return ("TUM_342", ptid, idx) if idx is not None else None

    for name in PRIORITY:
        idx = lookups[name].get(total_id)
        if idx is not None:
            return (name, total_id, idx)
    return None


def main():
    out_dir = Path(CONFIG["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_rows = {name: load_csv_rows(path) for name, path in CONFIG["meta_paths"].items()}
    for name, rows in meta_rows.items():
        print(f"Loaded {len(rows)} rows from {name} ({CONFIG['meta_paths'][name]})")
    lookups = build_lookups(meta_rows)

    total_rows = load_csv_rows(CONFIG["total_csv"])
    resolved, unmatched = [], []
    for row in total_rows:
        result = resolve_id(row["id"], lookups)
        if result is None:
            unmatched.append(row)
        else:
            resolved.append((row["id"], *result))

    print(f"\nResolved {len(resolved)} / {len(total_rows)} subjects:")
    for name, cnt in Counter(r[1] for r in resolved).items():
        print(f"  {name}: {cnt}")

    if unmatched:
        path = out_dir / "tum_unmatched.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=unmatched[0].keys())
            writer.writeheader()
            writer.writerows(unmatched)
        print(f"WARNING: {len(unmatched)} ids could not be resolved -> {path}")

    row_counts = {}
    for fwhm in CONFIG["fwhm_levels"]:
        for hemi in CONFIG["hemis"]:
            arrays = {}
            for name, folder in CONFIG["dataset_dirs"].items():
                path = Path(folder) / CONFIG["npy_name"].format(fwhm=fwhm, hemi=hemi)
                if path.exists():
                    arrays[name] = np.load(path, allow_pickle=True)
                else:
                    print(f"  missing: {path}")

            rows_out, ids_out = [], []
            for total_id, name, ptid, row_index in resolved:
                arr = arrays.get(name)
                if arr is None:
                    continue
                if row_index >= arr.shape[0]:
                    print(f"WARNING: row {row_index} out of bounds for {name} fwhm={fwhm} {hemi} "
                          f"({arr.shape[0]} rows) - {total_id} skipped")
                    continue
                rows_out.append(arr[row_index])
                ids_out.append((total_id, name, ptid, row_index))

            if not rows_out:
                print(f"  skipped fwhm={fwhm} {hemi}: no source arrays available")
                continue
            stacked = np.stack(rows_out, axis=0)
            np.save(out_dir / CONFIG["npy_name"].format(fwhm=fwhm, hemi=hemi), stacked)
            np.save(out_dir / f"tum_rows.fwhm{fwhm}.fsavg5.{hemi}.npy", np.array(ids_out, dtype=object))
            row_counts[(fwhm, hemi)] = stacked.shape[0]
            print(f"  fwhm={fwhm} {hemi}: {stacked.shape}")

    if len(set(row_counts.values())) > 1:
        print("\nWARNING: the output arrays have different numbers of subjects:")
        for (fwhm, hemi), n in row_counts.items():
            print(f"  fwhm={fwhm} {hemi}: {n}")
    print(f"\nDone. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
