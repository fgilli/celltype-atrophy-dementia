"""
Compute subject-specific SCSR deviation maps (vertex-wise z-scores).

Adapted from the original SCSR inference code. For every subject of a
cohort (TUM, NIFD or NACC), the retrained model of one hemisphere
reconstructs the fsaverage5 thickness map num_predictions times, each time
from a different random subset of visible vertices (corruption_rate = 0.8
of the vertices masked). For every vertex, only the reconstructions in
which the vertex was masked are kept and their 95th percentile is used as
the subject-specific expected thickness. The deviation z-score is

    z = (observed thickness - expected thickness) / sd_pop

with sd_pop the vertex-wise residual SD of the UK Biobank test split
(written by train_scsr.py). Negative values indicate thinning. Vertices of
the parcels "unknown" and "corpuscallosum" are not reconstructed and set
to -1 in the output.

Inputs
    --input-table   feather with columns ID, age, sex, DX, thickness...
                    (fsaverage5, one row per subject)
    --aparc-labels  .npy with the aparc parcel name of every fsaverage5 vertex
    --run-name      name of the training run (checkpoints/{run}_conf.yaml,
                    checkpoints/{run}_{checkpoint}.pth, variances/{run}_sd_pop.npy)

Outputs (in --output-dir)
    zscores_corr{rate}_perc{percentile}.npy       (n_subjects, 10242)
    predictions_corr{rate}_perc{percentile}.npy   expected thickness
    targets_corr{rate}_perc{percentile}.npy       observed thickness

Usage
    python sample_scsr.py --hemi lh --run-name RUN --input-table nifd_lh.feather \
        --aparc-labels aparc_labels_10K_lh.npy --output-dir samples/nifd_lh
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from train_scsr import APARC_IGNORE, AD_ROI, build_model_from_config, load_config

HERE = Path(__file__).resolve().parent
N_VERTICES = 10242


def corrupt_data(data, corruption_rate, use_region, region_indices, always_mask_indices, rng, device):
    corrupted = data.clone()
    if use_region:
        unique_region_indices = np.unique(region_indices)
        mask = torch.zeros_like(data, dtype=torch.bool, device=data.device)
        n_regions = int(len(unique_region_indices) * corruption_rate) - len(always_mask_indices)
        candidates = np.setdiff1d(unique_region_indices, always_mask_indices)
        for i in range(data.shape[0]):
            chosen = np.union1d(rng.choice(candidates, n_regions, replace=False), always_mask_indices)
            mask[i] = torch.tensor(np.isin(region_indices, chosen))
    else:
        mask = torch.rand(corrupted.shape, device=device) < corruption_rate
        if len(always_mask_indices) > 0:
            always = torch.tensor(np.isin(region_indices, always_mask_indices)).to(device)
            mask = mask | always.repeat(data.shape[0], 1)
    corrupted[mask] = 0
    return corrupted, mask


def reconstruct(model, loader, scaler, device, num_predictions, corruption_rate, percentile,
                use_region, region_indices, always_mask_indices, rng):
    """Return (expected thickness, observed thickness) in mm, both (n_subjects, n_kept_vertices)."""
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for (inputs,) in loader:
            age_sex = inputs[:, :2].to(device)
            thickness = inputs[:, 2:].to(device)
            samples = []
            for _ in range(num_predictions):
                corrupted, mask = corrupt_data(thickness, corruption_rate, use_region, region_indices,
                                               always_mask_indices, rng, device)
                outputs = model(torch.cat([age_sex, corrupted], dim=1))[:, 2:]
                outputs[~mask] = float("nan")
                samples.append(outputs)
            predictions.append(torch.nanquantile(torch.stack(samples), percentile, dim=0))
            actuals.append(thickness)
    pred = scaler.inverse_transform(torch.cat(predictions).cpu().numpy())
    true = scaler.inverse_transform(torch.cat(actuals).cpu().numpy())
    print(f"Reconstruction MSE: {np.mean((true - pred) ** 2):.5f}")
    return pred, true


def parse_args():
    p = argparse.ArgumentParser(description="SCSR inference: subject-specific deviation maps.")
    p.add_argument("--hemi", choices=["lh", "rh"], required=True)
    p.add_argument("--run-name", required=True)
    p.add_argument("--checkpoint", default="bestLossUKB", choices=["bestLossUKB", "bestLoss", "bestAUC"])
    p.add_argument("--input-table", required=True)
    p.add_argument("--aparc-labels", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--n-subjects", type=int, default=None, help="only the first N subjects (default: all)")
    p.add_argument("--num-predictions", type=int, default=100)
    p.add_argument("--corruption-rate", type=float, default=0.8)
    p.add_argument("--percentile", type=float, default=0.95)
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(seed=42)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    training_config = load_config(HERE / "checkpoints" / f"{args.run_name}_conf.yaml")
    network_config = load_config(HERE / "model" / training_config["network_config_file"])
    scaler = joblib.load(HERE / "scaler" / f"UKB_stdScaler_{args.hemi}.joblib")
    sd_pop = np.load(HERE / "variances" / f"{args.run_name}_sd_pop.npy")

    table = pd.read_feather(args.input_table).drop(columns="DX")
    aparc_names = np.load(args.aparc_labels, allow_pickle=True)

    unique_regions, region_indices = np.unique(aparc_names, return_inverse=True)
    remove_mask = np.isin(aparc_names, APARC_IGNORE)
    region_indices = region_indices[~remove_mask]
    always_masked = AD_ROI if training_config["mask_AD_ROI"] else []
    always_mask_indices = np.array([np.where(unique_regions == r)[0][0] for r in always_masked if r in unique_regions])

    age = table.iloc[:, 1].to_numpy().reshape(-1, 1).astype(np.float32)
    sex = table.iloc[:, 2].to_numpy(dtype=object).reshape(-1, 1)
    X = table.iloc[:, 3:].to_numpy(dtype=np.float32)[:, ~remove_mask]
    if args.n_subjects is not None:
        X, age, sex = X[:args.n_subjects], age[:args.n_subjects], sex[:args.n_subjects]
    n_subjects = X.shape[0]

    if training_config["use_age_sex"]:
        preprocessor = joblib.load(HERE / "scaler" / f"UKB_AS_stdScaler_{args.hemi}.joblib")
        as_norm = preprocessor.transform(np.hstack((age, sex)))
    else:
        as_norm = np.zeros((n_subjects, 2))
    X_norm = np.hstack((as_norm, scaler.transform(X)))
    loader = DataLoader(TensorDataset(torch.tensor(X_norm, dtype=torch.float32)), batch_size=n_subjects, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network_config["layers"][0]["in_features"] = X_norm.shape[1]
    network_config["layers"][-1]["out_features"] = X_norm.shape[1]
    model = build_model_from_config(network_config)
    snapshot = torch.load(HERE / "checkpoints" / f"{args.run_name}_{args.checkpoint}.pth", map_location=device)
    model.load_state_dict(snapshot["state_dict"])
    model = model.to(device)

    predictions, targets = reconstruct(
        model, loader, scaler, device, args.num_predictions, args.corruption_rate, args.percentile,
        training_config["use_region"], region_indices, always_mask_indices, rng,
    )
    zscores = (targets - predictions) / sd_pop

    tag = f"corr{args.corruption_rate}_perc{args.percentile}"
    for name, values in [("zscores", zscores), ("predictions", predictions), ("targets", targets)]:
        full = -np.ones((n_subjects, N_VERTICES))
        full[:, ~remove_mask] = values
        np.save(out_dir / f"{name}_{tag}.npy", full)
    print(f"Deviation maps of {n_subjects} subjects saved to {out_dir}")


if __name__ == "__main__":
    main()
