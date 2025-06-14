# unsupervised_aae_standalone.py – with two discriminators
from __future__ import annotations

import csv
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Mapping

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import torch.nn.functional as F
from components import Decoder, Discriminator
from utils import weights_init
import os


# ---------------------------------------------------------------------------
# 1. Configuration dataclass
# ---------------------------------------------------------------------------
@dataclass
class UAAEConfig:
    input_dim: int
    ae_hidden: int
    disc_hidden: int
    latent_dim_categorical: int
    latent_dim_style: int
    use_decoder_sigmoid: bool = True

    recon_loss_fn: nn.Module = nn.MSELoss()
    adv_loss_fn: nn.Module = nn.BCEWithLogitsLoss()

    device: torch.device | str | None = None

    def __post_init__(self):
        if self.device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(self.device)


class Encoder(nn.Module):
    def __init__(self,
                 input_dim: int,
                 ae_hidden: int,
                 latent_dim_categorical: int,
                 latent_dim_style: int):
        super().__init__()
        self.input_dropout = nn.Dropout(p=0.2)

        total_latent_dim = latent_dim_categorical + latent_dim_style

        self.fc1 = nn.Linear(input_dim, ae_hidden)
        self.bn1 = nn.BatchNorm1d(ae_hidden)
        self.fc2 = nn.Linear(ae_hidden, ae_hidden)
        self.bn2 = nn.BatchNorm1d(ae_hidden)

        self.fc_latent = nn.Linear(ae_hidden, total_latent_dim)

    def forward(self, x: torch.Tensor):
        x = self.input_dropout(x)
        h = F.relu(self.bn1(self.fc1(x)))
        h = F.relu(self.bn2(self.fc2(h)))

        z = self.fc_latent(h)
        return z


class UnsupervisedAdversarialAutoencoder(nn.Module):
    def __init__(self, cfg: UAAEConfig):
        super().__init__()
        self.cfg = cfg
        self.device = cfg.device
        self.encoder = Encoder(cfg.input_dim, cfg.ae_hidden, cfg.latent_dim_categorical, cfg.latent_dim_style).to(cfg.device)
        self.decoder = Decoder(cfg.latent_dim_categorical + cfg.latent_dim_style, cfg.ae_hidden, cfg.input_dim, cfg.use_decoder_sigmoid).to(cfg.device)
        self.disc_cat = Discriminator(cfg.disc_hidden, cfg.latent_dim_categorical).to(cfg.device)
        self.disc_style = Discriminator(cfg.disc_hidden, cfg.latent_dim_style).to(cfg.device)
        self.cat_softmax = nn.Softmax(dim=1)

        self.encoder.apply(weights_init)
        self.decoder.apply(weights_init)
        self.disc_cat.apply(weights_init)
        self.disc_style.apply(weights_init)

        self.recon_opt = torch.optim.SGD(list(self.encoder.parameters()) + list(self.decoder.parameters()), momentum=0.9, lr=0.01)
        self.gen_opt = torch.optim.SGD(self.encoder.parameters(), momentum=0.1, lr=0.1)
        self.disc_cat_opt = torch.optim.SGD(self.disc_cat.parameters(), momentum=0.1, lr=0.1)
        self.disc_style_opt = torch.optim.SGD(self.disc_style.parameters(), momentum=0.1, lr=0.1)

        self.recon_loss = cfg.recon_loss_fn
        self.adv_loss = cfg.adv_loss_fn

        self.prev_cluster_assignments = None

    def forward_encoder(self, x):
        z = self.encoder(x)
        z_cat = self.cat_softmax(z[:, : self.cfg.latent_dim_categorical])
        z_style = z[:, self.cfg.latent_dim_categorical:]
        return z_cat, z_style

    def forward_reconstruction(self, x):
        z_cat, z_style = self.forward_encoder(x)
        return self.decoder(torch.cat((z_cat, z_style), dim=1))

    def sample_cat_prior(self, n):
        labels = torch.randint(0, self.cfg.latent_dim_categorical, (n,), device=self.device)
        return F.one_hot(labels, num_classes=self.cfg.latent_dim_categorical).float()

    def sample_style_prior(self, n, std=1.0):
        return torch.randn(n, self.cfg.latent_dim_style, device=self.device) * std

    def cluster_assignment_change_rate(self, loader):
        self.eval()
        new_assignments = []
        for x, _ in loader:
            x = x.to(self.device)
            preds = self.forward_encoder(x)[0].argmax(dim=1)
            new_assignments.append(preds.cpu())
        new_assignments = torch.cat(new_assignments)
        if self.prev_cluster_assignments is None:
            self.prev_cluster_assignments = new_assignments
            return 0.0  # no change on first call
        changes = (self.prev_cluster_assignments != new_assignments).sum().item()
        rate = changes / len(new_assignments) * 100
        self.prev_cluster_assignments = new_assignments
        return rate

    def fit(self, train_loader, val_loader, test_loader, epochs, result_folder, prior_std: float = 1.0):
        os.makedirs(result_folder, exist_ok=True)
        log_path = f"{result_folder}/train_log.csv"

        initial_acc = self.evaluate_clustering(val_loader, test_loader) * 100.0
        print(f"Initial Clustering Accuracy (untrained): {initial_acc:.2f}%")

        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "recon_loss", "disc_cat_loss", "gen_cat_loss", "disc_style_loss", "gen_style_loss", "test_acc"
            ])

        batch_log_path = f"{result_folder}/batch_log.csv"
        with open(batch_log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "batch", "recon_loss", "disc_cat_loss", "gen_cat_loss", "disc_style_loss", "gen_style_loss"
            ])

        for epoch in range(epochs):
            if epoch == 0:
                self.recon_opt.param_groups[0]["lr"] = 0.01
                self.gen_opt.param_groups[0]["lr"] = 0.1
                for dopt in (self.disc_cat_opt, self.disc_style_opt):
                    dopt.param_groups[0]["lr"] = 0.1
            elif epoch == 50:
                self.recon_opt.param_groups[0]["lr"] = 0.001
                self.gen_opt.param_groups[0]["lr"] = 0.01
                for dopt in (self.disc_cat_opt, self.disc_style_opt):
                    dopt.param_groups[0]["lr"] = 0.01

            self.train()
            loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]", leave=False)
            for batch_idx, (x, _) in enumerate(loop, 1):
                x = x.to(self.device)

                # ───────── fit(..) – inner batch loop ─────────
                x_hat = self.forward_reconstruction(x)
                recon_loss = self.recon_loss(x_hat, x)
                self.recon_opt.zero_grad()
                recon_loss.backward()
                self.recon_opt.step()

                # ---- categorical adversary --------------------------------------------------
                z_real_cat = self.sample_cat_prior(x.size(0))
                d_real_cat = self.adv_loss(self.disc_cat(z_real_cat),
                                        torch.ones(x.size(0), device=self.device))

                z_fake_cat, _ = self.forward_encoder(x)
                d_fake_cat = self.adv_loss(self.disc_cat(z_fake_cat.detach()),
                                        torch.zeros(x.size(0), device=self.device))

                self.disc_cat_opt.zero_grad()
                (d_real_cat + d_fake_cat).backward()
                self.disc_cat_opt.step()

                g_cat = self.adv_loss(self.disc_cat(self.forward_encoder(x)[0]),
                                    torch.ones(x.size(0), device=self.device))

                # ---- style adversary --------------------------------------------------------
                z_real_sty = self.sample_style_prior(x.size(0), prior_std)
                d_real_sty = self.adv_loss(self.disc_style(z_real_sty),
                                        torch.ones(x.size(0), device=self.device))

                _, z_fake_sty = self.forward_encoder(x)
                d_fake_sty = self.adv_loss(self.disc_style(z_fake_sty.detach()),
                                        torch.zeros(x.size(0), device=self.device))

                self.disc_style_opt.zero_grad()
                (d_real_sty + d_fake_sty).backward()
                self.disc_style_opt.step()

                g_sty = self.adv_loss(self.disc_style(self.forward_encoder(x)[1]),
                                    torch.ones(x.size(0), device=self.device))

                # ---- single generator update -------------------------------------------------
                λ_style = 5.0  # or 10.0, experiment with values
                g_loss = g_cat + λ_style * g_sty
                g_loss = g_cat + g_sty
                self.gen_opt.zero_grad()
                g_loss.backward()
                
                # # ── print gradients ─────────────────────────────────────────────
                # print("Generator gradients (after backward):")
                # for name, param in self.encoder.named_parameters():
                #     if param.grad is not None:
                #         print(f"  {name:<40}  |  grad norm = {param.grad.norm().item():.6f}")
                #     else:
                #         print(f"  {name:<40}  |  no grad")
                
                self.gen_opt.step()

                with open(batch_log_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        epoch + 1, batch_idx,
                        recon_loss.item(),
                        (d_real_cat + d_fake_cat).item(),
                        g_cat.item(),
                        (d_real_sty + d_fake_sty).item(),
                        g_sty.item()
                    ])

                loop.set_postfix({
                    "Recon": recon_loss.item(),
                    "Disc_Cat": (d_real_cat + d_fake_cat).item(),
                    "Gen_Cat": g_cat.item(),
                    "Disc_Style": (d_real_sty + d_fake_sty).item(),
                    "Gen_Style": g_sty.item(),
                })

            last_test_acc = -1.0
            if epoch == 0 or (epoch + 1) % 5 == 0:
                last_test_acc = self.evaluate_clustering(val_loader, test_loader) * 100.0
                print(f"Test Clustering Accuracy: {last_test_acc:.2f}%")

            # cluster_change = None
            # if epoch > 0:
            #     cluster_change = self.cluster_assignment_change_rate(train_loader)

            if (epoch + 1) % 5 == 0:
                # if cluster_change is not None:
                #     print(f"Cluster Assignment Change Rate: {cluster_change:.2f}%")
                # else:
                #     cluster_change = 0.0
                #     print(f"Cluster Assignment Change Rate: {cluster_change:.2f}%")

                with open(log_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        epoch + 1,
                        recon_loss.item(),
                        (d_real_cat + d_fake_cat).item(),
                        g_cat.item(),
                        (d_real_sty + d_fake_sty).item(),
                        g_sty.item(),
                        last_test_acc
                    ])

            if (epoch + 1) % 50 == 0:
                ckpt_dir = f"{result_folder}/weights_epoch_{epoch+1}"
                os.makedirs(ckpt_dir, exist_ok=True)
                self.save_weights(f"{ckpt_dir}/weights")

    @torch.no_grad()
    def predict_clusters(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        if x.dim() == 4:
            x = x.view(x.size(0), -1)
        z = self.encoder(x.to(self.device))
        z_cat = self.cat_softmax(z[:, :self.cfg.latent_dim_categorical])
        return z_cat.argmax(1)

    @torch.no_grad()
    def evaluate_clustering(self, val_loader, test_loader):
        self.eval()
        K = self.cfg.latent_dim_categorical
        cluster_labels = torch.full((K,), -1, dtype=torch.long, device=self.device)

        max_probs = torch.zeros(K, device=self.device)

        val_loop = tqdm(val_loader, desc="Scanning val (for cluster labels)", leave=False)
        for x, y in val_loop:
            x, y = x.to(self.device), y.to(self.device)
            z_cat, _ = self.forward_encoder(x)
            probs, preds = z_cat.max(dim=1)
            for i in range(x.size(0)):
                cluster = preds[i].item()
                prob = probs[i].item()
                if prob > max_probs[cluster]:
                    max_probs[cluster] = prob
                    cluster_labels[cluster] = y[i]

        correct = total = 0
        test_loop = tqdm(test_loader, desc="Evaluating on test", leave=False)
        for x, y in test_loop:
            x, y = x.to(self.device), y.to(self.device)
            preds = self.predict_clusters(x)
            mapped = cluster_labels[preds]
            correct += (mapped == y).sum().item()
            total += y.size(0)

        accuracy = correct / total
        return accuracy

    @torch.no_grad()
    def generate(self, n: int, prior_std: float = 1.0):
        z_cat = F.one_hot(torch.randint(0, self.cfg.latent_dim_categorical, (n,), device=self.device), num_classes=self.cfg.latent_dim_categorical).float()
        z_style = torch.randn(n, self.cfg.latent_dim_style, device=self.device) * prior_std
        z = torch.cat([z_cat, z_style], dim=1)
        return self.decoder(z)

    def save_weights(self, path_prefix="aae_weights"):
        torch.save(self.encoder.state_dict(), f"{path_prefix}_encoder.pth")
        torch.save(self.decoder.state_dict(), f"{path_prefix}_decoder.pth")
        torch.save(self.disc_cat.state_dict(), f"{path_prefix}_disc_categorical.pth")
        torch.save(self.disc_style.state_dict(), f"{path_prefix}_disc_style.pth")
        print(f"Weights saved to {path_prefix}_*.pth")

    def load_weights(self, path_prefix="aae_weights"):
        self.encoder.load_state_dict(torch.load(f"{path_prefix}_encoder.pth", map_location=self.device))
        self.decoder.load_state_dict(torch.load(f"{path_prefix}_decoder.pth", map_location=self.device))
        self.disc_cat.load_state_dict(torch.load(f"{path_prefix}_disc_categorical.pth", map_location=self.device))
        self.disc_style.load_state_dict(torch.load(f"{path_prefix}_disc_style.pth", map_location=self.device))
        print(f"Weights loaded from {path_prefix}_*.pth")

    def __repr__(self):
        return f"<UnsupervisedAAE {', '.join(f'{k}={v}' for k,v in asdict(self.cfg).items())}>"
