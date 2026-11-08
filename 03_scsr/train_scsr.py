"""
Retrain the SCSR (Stochastic Cortical Self-Reconstruction) model.

Adapted from the original SCSR training code (Wachinger et al., Medical
Image Analysis). An MLP denoising autoencoder is trained on fsaverage5
cortical thickness of healthy UK Biobank subjects (FreeSurfer 7.2), one
model per hemisphere:

  * vertices of the parcels "unknown" and "corpuscallosum" are removed,
  * thickness is standardized per vertex (StandardScaler fit on the
    training split),
  * during training, Gaussian jitter is added and a random fraction of the
    vertices (corruption_rate_train) is masked with zeros; the MSE loss is
    computed on the masked vertices only,
  * optimizer AdamW with cosine-annealing learning-rate schedule,
  * 80/20 split of the UK Biobank table (random_state=42) for training and
    held-out testing; an external validation table (ADNI, CN and dementia)
    is evaluated for monitoring only.

Three checkpoints are kept: lowest UKB test loss ("bestLossUKB", used for
inference in sample_scsr.py), lowest UKB test + CN validation loss
("bestLoss") and highest AD-ROI AUC on the validation table ("bestAUC").
After training, the vertex-wise standard deviation of the reconstruction
residuals on the UKB test split (sd_pop) is saved; it is used to turn
residuals into deviation z-scores.

Inputs
    training_configs/<config>.yaml   training hyperparameters
    model/<network_config_file>      network architecture (layer list)
    --train-table   UKB feather: age, sex, DX, ID, full ID, thickness...
    --val-table     validation feather: age, sex, DX, thickness...
    --aparc-labels  .npy with the aparc parcel name of every fsaverage5 vertex

Outputs (relative to this folder)
    checkpoints/{run}_{bestLossUKB,bestLoss,bestAUC}.pth (+ .csv, _conf.yaml)
    scaler/UKB_stdScaler_{hemi}.joblib, scaler/UKB_AS_stdScaler_{hemi}.joblib
    variances/{run}_sd_pop.npy

Usage
    python train_scsr.py training_config.yaml --hemi lh \
        --train-table UKB_lh.feather --val-table ADNI_lh.feather \
        --aparc-labels aparc_labels_10K_lh.npy [--run-name NAME] [--wandb]
"""

import argparse
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
APARC_IGNORE = ["unknown", "corpuscallosum"]
AD_ROI = ["entorhinal", "inferiortemporal", "middletemporal", "inferiorparietal", "fusiform"]


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_model_from_config(config):
    layers = []
    last_out_features = None
    for layer in config["layers"]:
        if layer["type"] == "Linear":
            layers.append(nn.Linear(layer["in_features"], layer["out_features"]))
            last_out_features = layer["out_features"]
        elif layer["type"] == "BatchNorm1d":
            layers.append(nn.BatchNorm1d(last_out_features))
        elif layer["type"] == "SiLU":
            layers.append(nn.SiLU())
        elif layer["type"] == "Dropout":
            layers.append(nn.Dropout(p=layer.get("p", 0.5)))
    return nn.Sequential(*layers)


class Logger:
    """Thin wrapper so Weights & Biases logging is optional."""

    def __init__(self, enabled, cfg, run_name):
        self.run = None
        if enabled:
            import wandb
            self.run = wandb.init(project="SCSR", config=cfg, name=run_name)
            run_name = self.run.name
        self.name = run_name or time.strftime("scsr_%Y%m%d_%H%M%S")

    def log(self, values):
        if self.run is not None:
            self.run.log(values)

    def watch(self, model):
        if self.run is not None:
            self.run.watch(model, log_freq=2)


class SCSRTrainer:
    def __init__(self, cfg, region_indices, always_mask_indices, aparc_names, scaler, device):
        self.cfg = cfg
        self.region_indices = region_indices
        self.unique_region_indices = np.unique(region_indices)
        self.always_mask_indices = always_mask_indices
        self.aparc_names = aparc_names
        self.scaler = scaler
        self.device = device

    def corrupt_data(self, data, corruption_rate):
        """Mask a fraction of the vertices (or of whole parcels if use_region) with zeros."""
        corrupted = data.clone()
        if self.cfg["use_region"]:
            mask = torch.zeros_like(data, dtype=torch.bool, device=data.device)
            n_regions = int(len(self.unique_region_indices) * corruption_rate) - len(self.always_mask_indices)
            candidates = np.setdiff1d(self.unique_region_indices, self.always_mask_indices)
            for i in range(data.shape[0]):
                chosen = np.random.choice(candidates, n_regions, replace=False)
                chosen = np.union1d(chosen, self.always_mask_indices)
                mask[i] = torch.tensor(np.isin(self.region_indices, chosen))
        else:
            mask = torch.rand(corrupted.shape, device=self.device) < corruption_rate
            if len(self.always_mask_indices) > 0:
                always = torch.tensor(np.isin(self.region_indices, self.always_mask_indices)).to(self.device)
                mask = mask | always.repeat(data.shape[0], 1)
        corrupted[mask] = 0
        return corrupted, mask

    def test_loss(self, model, criterion, loader):
        model.eval()
        total_loss, total_count = 0.0, 0
        with torch.no_grad():
            for (full_inputs,) in loader:
                full_inputs = full_inputs.to(self.device)
                age_sex, thickness = full_inputs[:, :2], full_inputs[:, 2:]
                corrupted, mask = self.corrupt_data(thickness, self.cfg["corruption_rate_test"])
                outputs = model(torch.cat([age_sex, corrupted], dim=1))
                full_mask = torch.cat([torch.ones(age_sex.shape[0], 2).bool().to(self.device), mask], dim=1)
                loss = criterion(outputs[full_mask], full_inputs[full_mask])
                total_loss += loss.item() * thickness.size(0)
                total_count += thickness.size(0)
        return total_loss / total_count

    def reconstruct(self, model, loader):
        """Quantile of num_predictions masked reconstructions; returns residuals in mm."""
        model.eval()
        predictions, actuals = [], []
        with torch.no_grad():
            for (inputs,) in loader:
                age_sex = inputs[:, :2].to(self.device)
                thickness = inputs[:, 2:].to(self.device)
                samples = []
                for _ in range(self.cfg["num_predictions"]):
                    corrupted, mask = self.corrupt_data(thickness, self.cfg["corruption_rate_test"])
                    outputs = model(torch.cat([age_sex, corrupted], dim=1))[:, 2:]
                    outputs[~mask] = float("nan")
                    samples.append(outputs)
                predictions.append(torch.nanquantile(torch.stack(samples), self.cfg["percentile"], dim=0))
                actuals.append(thickness)
        pred = self.scaler.inverse_transform(torch.cat(predictions).cpu().numpy())
        true = self.scaler.inverse_transform(torch.cat(actuals).cpu().numpy())
        residuals = true - pred
        mse = float(np.mean(residuals ** 2))
        print(f"Reconstruction MSE: {mse:.5f}")
        return mse, residuals

    def ad_roi_auc(self, model, test_loader, val_cn_loader, val_ad_loader):
        """AUC of the summed AD-ROI deviation scores between validation CN and dementia."""
        _, res_ukb = self.reconstruct(model, test_loader)
        sd_pop = np.nanstd(res_ukb, axis=0)
        _, res_cn = self.reconstruct(model, val_cn_loader)
        _, res_ad = self.reconstruct(model, val_ad_loader)

        def roi_sum(res):
            z = pd.DataFrame(res / sd_pop, columns=self.aparc_names)
            return z.T.groupby(self.aparc_names).mean().T[AD_ROI].sum(axis=1)

        scores = pd.concat([roi_sum(res_cn), roi_sum(res_ad)], ignore_index=True)
        labels = np.r_[np.zeros(len(res_cn)), np.ones(len(res_ad))]
        return max(roc_auc_score(labels, scores), roc_auc_score(labels, -scores))

    def train(self, model, loaders, logger, training_config):
        cfg = self.cfg
        train_loader, test_loader, val_cn_loader, val_ad_loader = loaders
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["T_max"])
        ckpt_dir = HERE / "checkpoints"

        best = {"loss": (10.0, {}), "auc": (0.0, {}), "loss_ukb": (10.0, {})}

        def snapshot(epoch, **extra):
            return {"epoch": epoch + 1, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), **extra}

        for epoch in range(cfg["epochs"]):
            model.train()
            start = time.time()
            train_loss = 0.0
            for (full_inputs,) in train_loader:
                full_inputs = full_inputs.to(self.device)
                age_sex, thickness = full_inputs[:, :2], full_inputs[:, 2:]
                jittered = thickness + torch.randn_like(thickness) * cfg["jitter_sd"]
                corrupted, mask = self.corrupt_data(jittered, cfg["corruption_rate_train"])
                optimizer.zero_grad()
                outputs = model(torch.cat([age_sex, corrupted], dim=1))
                full_mask = torch.cat([torch.ones(age_sex.shape[0], 2).bool().to(self.device), mask], dim=1)
                loss = criterion(outputs[full_mask], full_inputs[full_mask])
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            scheduler.step()
            train_loss /= len(train_loader)

            test_loss = self.test_loss(model, criterion, test_loader)
            val_cn_loss = self.test_loss(model, criterion, val_cn_loader)
            val_ad_loss = self.test_loss(model, criterion, val_ad_loader)
            losses = {"test_loss": test_loss, "val_CN_loss": val_cn_loss, "val_AD_loss": val_ad_loss}

            if test_loss + val_cn_loss < best["loss"][0]:
                best["loss"] = (test_loss + val_cn_loss, snapshot(epoch, **losses))

            if epoch > 1 and epoch % 3 == 0:
                auc = self.ad_roi_auc(model, test_loader, val_cn_loader, val_ad_loader)
                print(f"Epoch [{epoch + 1}/{cfg['epochs']}], AUC: {auc:.4f}")
                logger.log({"epoch": epoch, "AUC": auc})
                if auc > best["auc"][0]:
                    best["auc"] = (auc, snapshot(epoch, AUC=auc, **losses))

            if test_loss < best["loss_ukb"][0]:
                best["loss_ukb"] = (test_loss, snapshot(epoch, test_loss=test_loss))

            if (epoch % 50 == 0 and epoch > 1) or epoch == cfg["epochs"] - 1:
                for key, suffix in [("loss", "bestLoss"), ("auc", "bestAUC"), ("loss_ukb", "bestLossUKB")]:
                    ckpt = best[key][1]
                    torch.save(ckpt, ckpt_dir / f"{logger.name}_{suffix}.pth")
                    save_config_res_to_csv(training_config, ckpt, ckpt_dir / f"{logger.name}_{suffix}.csv")

            print(f"Epoch [{epoch + 1}/{cfg['epochs']}], train {train_loss:.4f}, test {test_loss:.4f}, "
                  f"val CN {val_cn_loss:.4f}, val AD {val_ad_loss:.4f}, {time.time() - start:.1f}s")
            logger.log({"epoch": epoch, "training_loss": train_loss, **losses})


def save_config_res_to_csv(config, checkpoint, csv_path):
    keys = ["epoch", "test_loss", "val_CN_loss", "val_AD_loss", "AUC"]
    params = {k: checkpoint[k] for k in keys if k in checkpoint}
    pd.DataFrame([{**config, **params}]).to_csv(csv_path, index=False)


def parse_args():
    p = argparse.ArgumentParser(description="Train the SCSR denoising autoencoder.")
    p.add_argument("config_file", help="YAML file in training_configs/")
    p.add_argument("--hemi", choices=["lh", "rh"], required=True)
    p.add_argument("--train-table", required=True)
    p.add_argument("--val-table", required=True)
    p.add_argument("--aparc-labels", required=True)
    p.add_argument("--run-name", default=None)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--wandb", action="store_true", help="log to Weights & Biases")
    return p.parse_args()


def main():
    args = parse_args()
    training_config = load_config(HERE / "training_configs" / args.config_file)
    cfg = dict(training_config)
    network_config = load_config(HERE / "model" / cfg["network_config_file"])
    if cfg.get("harmonization", False) or cfg.get("cortex_res", "10K") != "10K":
        raise ValueError("Only the configuration used in the thesis is supported: cortex_res=10K, harmonization=false")

    for d in ["checkpoints", "scaler", "variances"]:
        (HERE / d).mkdir(exist_ok=True)

    logger = Logger(args.wandb, cfg, args.run_name)
    conf_copy = HERE / "checkpoints" / f"{logger.name}_conf.yaml"
    shutil.copy(HERE / "training_configs" / args.config_file, conf_copy)
    with open(conf_copy, "a") as f:
        f.write(f"\norig_conf_file: {args.config_file}\n")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    scaler_name = HERE / "scaler" / f"UKB_stdScaler_{args.hemi}.joblib"
    as_scaler_name = HERE / "scaler" / f"UKB_AS_stdScaler_{args.hemi}.joblib"

    data_matrix = pd.read_feather(args.train_table).values
    val_matrix = pd.read_feather(args.val_table).values
    aparc_names = np.load(args.aparc_labels, allow_pickle=True)

    unique_regions, region_indices = np.unique(aparc_names, return_inverse=True)
    remove_mask = np.isin(aparc_names, APARC_IGNORE)
    aparc_names = aparc_names[~remove_mask]
    region_indices = region_indices[~remove_mask]

    always_masked = AD_ROI if cfg["mask_AD_ROI"] else []
    always_mask_indices = np.array([np.where(unique_regions == r)[0][0] for r in always_masked if r in unique_regions])

    age = data_matrix[:, 0].reshape(-1, 1).astype(np.float32)
    sex = data_matrix[:, 1].reshape(-1, 1)
    X_train, X_test, as_train, as_test = train_test_split(
        data_matrix[:, 5:].astype(np.float32), np.hstack((age, sex)), test_size=0.2, random_state=42)

    age_val = val_matrix[:, 0].reshape(-1, 1).astype(np.float32)
    sex_val = val_matrix[:, 1].reshape(-1, 1)
    dx_val = val_matrix[:, 2]
    X_val = val_matrix[:, 3:].astype(np.float32)

    X_train, X_test, X_val = (x[:, ~remove_mask] for x in (X_train, X_test, X_val))
    del data_matrix

    preprocessor = ColumnTransformer(transformers=[
        ("age", StandardScaler(), [0]),
        ("sex", OneHotEncoder(drop="if_binary"), [1]),
    ])
    as_train_n = preprocessor.fit_transform(as_train)
    as_test_n = preprocessor.transform(as_test)
    as_val_n = preprocessor.transform(np.hstack((age_val, sex_val)))
    joblib.dump(preprocessor, as_scaler_name)
    if not cfg["use_age_sex"]:
        as_train_n, as_test_n, as_val_n = (np.zeros(a.shape) for a in (as_train_n, as_test_n, as_val_n))

    scaler = StandardScaler()
    X_train_n = np.hstack((as_train_n, scaler.fit_transform(X_train)))
    X_test_n = np.hstack((as_test_n, scaler.transform(X_test)))
    X_val_n = np.hstack((as_val_n, scaler.transform(X_val)))
    joblib.dump(scaler, scaler_name)

    to_tensor = lambda a: torch.tensor(a, dtype=torch.float32)
    bs = cfg["batch_size"]
    train_loader = DataLoader(TensorDataset(to_tensor(X_train_n)), batch_size=bs, shuffle=True)
    test_loader = DataLoader(TensorDataset(to_tensor(X_test_n)), batch_size=bs, shuffle=False)
    val_cn_loader = DataLoader(TensorDataset(to_tensor(X_val_n[dx_val == "CN"])), batch_size=bs, shuffle=False)
    val_ad_loader = DataLoader(TensorDataset(to_tensor(X_val_n[dx_val == "Dementia"])), batch_size=bs, shuffle=False)

    input_dim = X_train_n.shape[1]
    network_config["layers"][0]["in_features"] = input_dim
    network_config["layers"][-1]["out_features"] = input_dim
    model = build_model_from_config(network_config).to(device)
    print(model)
    print(f"Number of model parameters: {sum(p.numel() for p in model.parameters())}")
    logger.watch(model)

    trainer = SCSRTrainer(cfg, region_indices, always_mask_indices, aparc_names, scaler, device)
    trainer.train(model, (train_loader, test_loader, val_cn_loader, val_ad_loader), logger, training_config)

    mse, res_test = trainer.reconstruct(model, test_loader)
    mse_cn, res_cn = trainer.reconstruct(model, val_cn_loader)
    mse_ad, res_ad = trainer.reconstruct(model, val_ad_loader)
    logger.log({"mse": mse, "mse_val_cn": mse_cn, "mse_val_ad": mse_ad})

    sd_pop = np.nanstd(res_test, axis=0)
    np.save(HERE / "variances" / f"{logger.name}_sd_pop.npy", sd_pop)
    print(f"Normalized MSE - test: {np.mean((res_test / sd_pop) ** 2):.4f}, "
          f"val CN: {np.mean((res_cn / sd_pop) ** 2):.4f}, val AD: {np.mean((res_ad / sd_pop) ** 2):.4f}")


if __name__ == "__main__":
    main()
