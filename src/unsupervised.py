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
    def __init__(self, input_dim: int, ae_hidden: int, output_dim: int) -> None:
        super(Encoder, self).__init__()
        self.input_dropout = nn.Dropout(p=0.2)

        self.fc1 = nn.Linear(input_dim, ae_hidden)
        self.bn1 = nn.BatchNorm1d(ae_hidden)
        self.fc2 = nn.Linear(ae_hidden, ae_hidden)
        self.bn2 = nn.BatchNorm1d(ae_hidden)
        self.fc3 = nn.Linear(ae_hidden, output_dim)
        self.bn3 = nn.BatchNorm1d(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_dropout(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.bn3(self.fc3(x))
        return x

# ---------------------------------------------------------------------------
# 2. Unsupervised AAE implementation with 2 discriminators
# ---------------------------------------------------------------------------
class UnsupervisedAdversarialAutoencoder(nn.Module):
    def __init__(self, cfg: UAAEConfig):
        super().__init__()
        self.cfg = cfg
        self.device = cfg.device
        self.encoder = Encoder(cfg.input_dim, cfg.ae_hidden, cfg.latent_dim_categorical + cfg.latent_dim_style).to(cfg.device)
        self.decoder = Decoder(cfg.latent_dim_categorical + cfg.latent_dim_style, cfg.ae_hidden, cfg.input_dim, cfg.use_decoder_sigmoid).to(cfg.device)
        self.disc_cat = Discriminator(cfg.disc_hidden, cfg.latent_dim_categorical).to(cfg.device)
        self.disc_style = Discriminator(cfg.disc_hidden, cfg.latent_dim_style).to(cfg.device)
        self.cat_softmax = nn.Softmax(dim=1)

        self.encoder.apply(weights_init)
        self.decoder.apply(weights_init)
        self.disc_cat.apply(weights_init)
        self.disc_style.apply(weights_init)

        self.recon_opt = torch.optim.SGD(list(self.encoder.parameters()) + list(self.decoder.parameters()), momentum=0.9)
        self.gen_style_opt = torch.optim.SGD(self.encoder.parameters(), momentum=0.1)
        self.gen_cat_opt = torch.optim.SGD(self.encoder.parameters(), momentum=0.1)
        self.disc_cat_opt = torch.optim.SGD(self.disc_cat.parameters(), momentum=0.1)
        self.disc_style_opt = torch.optim.SGD(self.disc_style.parameters(), momentum=0.1)

        self.recon_loss = cfg.recon_loss_fn
        self.adv_loss = cfg.adv_loss_fn

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


    def fit(self, train_loader, val_loader, epochs, result_folder, prior_std=5.0):
        os.makedirs(result_folder, exist_ok=True)
        with open(f'{result_folder}/train_log.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'recon_loss', 'disc_cat_loss', 'gen_cat_loss', 'disc_style_loss', 'gen_style_loss'])

        for epoch in range(epochs):

            # Learning rate decay schedule (optional)
            if epoch == 0:
                self.recon_opt.param_groups[0]['lr'] = 0.01
                self.gen_style_opt.param_groups[0]['lr'] = 0.1
                self.gen_cat_opt.param_groups[0]['lr'] = 0.1
                self.disc_style_opt.param_groups[0]['lr'] = 0.1
                self.disc_cat_opt.param_groups[0]['lr'] = 0.1
            elif epoch == 49:
                self.recon_opt.param_groups[0]['lr'] = 0.001
                self.gen_style_opt.param_groups[0]['lr'] = 0.01
                self.gen_cat_opt.param_groups[0]['lr'] = 0.01
                self.disc_style_opt.param_groups[0]['lr'] = 0.01
                self.disc_cat_opt.param_groups[0]['lr'] = 0.01

            total_recon_loss = 0
            total_disc_style_loss = 0
            total_gen_style_loss = 0
            total_disc_cat_loss = 0
            total_gen_cat_loss = 0

            self.train()
            loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]", leave=False)
            for batch_idx, (x, _) in enumerate(loop, 1):
                x = x.to(self.device)

                # === RECONSTRUCTION PHASE ===
                x_hat = self.forward_reconstruction(x)
                recon_loss = self.recon_loss(x_hat, x)
                self.recon_opt.zero_grad()
                recon_loss.backward()
                self.recon_opt.step()

                # === CATEGORICAL DISCRIMINATOR ===
                z_real_cat = self.sample_cat_prior(x.size(0))
                z_fake_cat, _ = self.forward_encoder(x)

                d_real_loss_cat = self.adv_loss(self.disc_cat(z_real_cat), torch.ones(x.size(0), device=self.device))
                d_fake_loss_cat = self.adv_loss(self.disc_cat(z_fake_cat.detach()), torch.zeros(x.size(0), device=self.device))
                self.disc_cat_opt.zero_grad()
                (d_real_loss_cat + d_fake_loss_cat).backward()
                self.disc_cat_opt.step()

                gen_cat_loss = self.adv_loss(self.disc_cat(z_fake_cat), torch.ones(x.size(0), device=self.device))
                self.gen_cat_opt.zero_grad()
                gen_cat_loss.backward()
                self.gen_cat_opt.step()

                # === STYLE DISCRIMINATOR ===
                z_real_style = self.sample_style_prior(x.size(0), prior_std)
                _, z_fake_style = self.forward_encoder(x)
                d_real_loss_style = self.adv_loss(self.disc_style(z_real_style), torch.ones(x.size(0), device=self.device))
                d_fake_loss_style = self.adv_loss(self.disc_style(z_fake_style.detach()), torch.zeros(x.size(0), device=self.device))
                self.disc_style_opt.zero_grad()
                (d_real_loss_style + d_fake_loss_style).backward()
                self.disc_style_opt.step()

                gen_style_loss = self.adv_loss(self.disc_style(z_fake_style), torch.ones(x.size(0), device=self.device))
                self.gen_style_opt.zero_grad()
                gen_style_loss.backward()
                self.gen_style_opt.step()

                # === Accumulate Losses ===
                total_recon_loss += recon_loss.item()
                total_disc_cat_loss += (d_real_loss_cat + d_fake_loss_cat).item()
                total_gen_cat_loss += gen_cat_loss.item()
                total_disc_style_loss += (d_real_loss_style + d_fake_loss_style).item()
                total_gen_style_loss += gen_style_loss.item()

                loop.set_postfix({
                    "Recon": total_recon_loss / batch_idx,
                    "Disc_Cat": total_disc_cat_loss / batch_idx,
                    "Gen_Cat": total_gen_cat_loss / batch_idx,
                    "Disc_Style": total_disc_style_loss / batch_idx,
                    "Gen_Style": total_gen_style_loss / batch_idx
                })

            # Logging
            avg_recon_loss = total_recon_loss / len(train_loader)
            avg_disc_cat_loss = total_disc_cat_loss / len(train_loader)
            avg_gen_cat_loss = total_gen_cat_loss / len(train_loader)
            avg_disc_style_loss = total_disc_style_loss / len(train_loader)
            avg_gen_style_loss = total_gen_style_loss / len(train_loader)

            val_acc = None
            if (epoch + 1) % 5 == 0:
                val_acc = self.evaluate_clustering(val_loader) * 100
                print(f"Validation Clustering Accuracy: {val_acc:.2f}%")

            print(f"Epoch {epoch+1}/{epochs} - Recon: {avg_recon_loss:.4f}, Disc_Cat: {avg_disc_cat_loss:.4f}, Gen_Cat: {avg_gen_cat_loss:.4f}, Disc_Style: {avg_disc_style_loss:.4f}, Gen_Style: {avg_gen_style_loss:.4f}")

            with open(f'{result_folder}/train_log.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch+1, avg_recon_loss, avg_disc_cat_loss, avg_gen_cat_loss, avg_disc_style_loss, avg_gen_style_loss, val_acc if val_acc is not None else ''])

            if (epoch + 1) % 50 == 0:
                os.makedirs(f'{result_folder}/weights_epoch_{epoch+1}', exist_ok=True)
                self.save_weights(f'{result_folder}/weights_epoch_{epoch+1}/weights')


    def predict_clusters(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        if x.dim() == 4:
            x = x.view(x.size(0), -1)
        z = self.encoder(x.to(self.device))
        z_cat = self.cat_softmax(z[:, :self.cfg.latent_dim_categorical])
        return z_cat.argmax(1)

    def evaluate_clustering(self, loader: DataLoader, n_classes: int = 10) -> float:
        self.eval()
        K = self.cfg.latent_dim_categorical
        vote = torch.zeros(K, n_classes, dtype=torch.long, device=self.device)

        for x, y in loader:
            x = x.to(self.device)
            y = y.to(self.device)
            preds = self.predict_clusters(x)
            for k in range(K):
                mask = preds == k
                if mask.any():
                    vote[k] += torch.bincount(y[mask], minlength=n_classes)

        majority = vote.argmax(1)
        correct = total = 0
        for x, y in loader:
            mapped = majority[self.predict_clusters(x)]
            correct += (mapped.cpu() == y).sum().item()
            total += y.size(0)
        return correct / total

    def generate(self, n: int, prior_std: float = 5.0):
        z_cat = F.one_hot(torch.randint(0, self.cfg.latent_dim_categorical, (n,), device=self.device), num_classes=self.cfg.latent_dim_categorical).float()
        z_style = torch.randn(n, self.cfg.latent_dim_style, device=self.device) * prior_std
        z = torch.cat([z_cat, z_style], dim=1)
        return self.decoder(z)

    def save_weights(self, path_prefix="aae_weights"):
        """
        Saves the weights of the encoder, decoder, and both discriminators.

        Args:
            path_prefix (str): Prefix for the saved file paths. Files will be saved as:
                - <path_prefix>_encoder.pth
                - <path_prefix>_decoder.pth
                - <path_prefix>_disc_categorical.pth
                - <path_prefix>_disc_style.pth
        """
        torch.save(self.encoder.state_dict(), f"{path_prefix}_encoder.pth")
        torch.save(self.decoder.state_dict(), f"{path_prefix}_decoder.pth")
        torch.save(self.discriminator_categorical.state_dict(), f"{path_prefix}_disc_categorical.pth")
        torch.save(self.discriminator_style.state_dict(), f"{path_prefix}_disc_style.pth")
        print(f"Weights saved to {path_prefix}_*.pth")

    def load_weights(self, path_prefix="aae_weights"):
        """
        Loads the weights of the encoder, decoder, and both discriminators.

        Args:
            path_prefix (str): Prefix for the saved file paths. Expected files are:
                - <path_prefix>_encoder.pth
                - <path_prefix>_decoder.pth
                - <path_prefix>_disc_categorical.pth
                - <path_prefix>_disc_style.pth
        """
        self.encoder.load_state_dict(torch.load(f"{path_prefix}_encoder.pth", map_location=self.device))
        self.decoder.load_state_dict(torch.load(f"{path_prefix}_decoder.pth", map_location=self.device))
        self.discriminator_categorical.load_state_dict(
            torch.load(f"{path_prefix}_disc_categorical.pth", map_location=self.device)
        )
        self.discriminator_style.load_state_dict(
            torch.load(f"{path_prefix}_disc_style.pth", map_location=self.device)
        )
        print(f"Weights loaded from {path_prefix}_*.pth")

    def __repr__(self):
        return f"<UnsupervisedAAE {', '.join(f'{k}={v}' for k,v in asdict(self.cfg).items())}>"