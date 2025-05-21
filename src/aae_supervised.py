import csv
import os

import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import wandb
from src.components import Encoder, Decoder, Discriminator
from src.utils import save_weights, load_weights, weights_init


class SupervisedAdversarialAutoencoder(nn.Module):
    def __init__(self, input_dim, ae_hidden, dc_hidden, latent_dim, num_classes,
                 recon_loss_fn, init_recon_lr, init_gen_lr, init_disc_lr,
                 use_decoder_sigmoid=True, device="cuda"):
        super(SupervisedAdversarialAutoencoder, self).__init__()
        self.device = device
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        self.encoder = Encoder(input_dim, ae_hidden, latent_dim).to(device)
        self.decoder = Decoder(latent_dim + num_classes, ae_hidden, input_dim, use_decoder_sigmoid).to(device)
        self.discriminator = Discriminator(dc_hidden, latent_dim).to(device)

        self.encoder.apply(weights_init)
        self.decoder.apply(weights_init)
        self.discriminator.apply(weights_init)

        # self.recon_opt = torch.optim.SGD(
        #     list(self.encoder.parameters()) + list(self.decoder.parameters()),
        #     lr=init_recon_lr,
        #     momentum=0.9
        # )
        #
        # self.gen_opt = torch.optim.SGD(
        #     self.encoder.parameters(),
        #     lr=init_gen_lr,
        #     momentum=0.1
        # )
        #
        # self.disc_opt = torch.optim.SGD(
        #     self.discriminator.parameters(),
        #     lr=init_disc_lr,
        #     momentum=0.1
        # )

        self.recon_opt = torch.optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()),
                                          lr=init_recon_lr)
        self.gen_opt = torch.optim.Adam(self.encoder.parameters(), lr=init_gen_lr)
        self.disc_opt = torch.optim.Adam(self.discriminator.parameters(), lr=init_disc_lr)

        self.recon_loss = recon_loss_fn
        self.adv_loss = nn.BCEWithLogitsLoss()

    def train_mbgd(self, data_loader, epochs, model, csv_log_path, prior_std=5.0):
        self.train()
        wandb.init(project="Generative Modeling", config={
            "epochs": epochs,
            "latent_dim": self.latent_dim,
            "prior_std": prior_std,
            "learning_rate_initial": self.recon_opt.param_groups[0]['lr'],
            "model": model
        })
        if not os.path.exists(csv_log_path):
            with open(csv_log_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Epoch", "Recon Loss", "Disc Loss", "Gen Loss"])
        for epoch in tqdm(range(epochs), desc="Training Epochs", unit="epoch"):

            if epoch == 50:
                self.recon_opt.param_groups[0]['lr'] = 0.0001
                self.gen_opt.param_groups[0]['lr'] = 0.0001
                self.disc_opt.param_groups[0]['lr'] = 0.0001
            if epoch == 1000:
                self.recon_opt.param_groups[0]['lr'] = 0.00001
                self.gen_opt.param_groups[0]['lr'] = 0.00001
                self.disc_opt.param_groups[0]['lr'] = 0.00001

            total_recon_loss = 0
            total_disc_loss = 0
            total_gen_loss = 0

            for batch_idx, (x, labels) in enumerate(data_loader):
                x = x.to(self.device)
                labels = labels.to(self.device)
                y_onehot = F.one_hot(labels, num_classes=self.num_classes).float()

                # Add gaussian noise to input
                if "noise" in model:
                    x = x + torch.randn_like(x) * 0.3

                # === RECONSTRUCTION ===
                z = self.encoder(x)
                z_cat = torch.cat([z, y_onehot], dim=1)
                x_hat = self.decoder(z_cat)
                recon_loss = self.recon_loss(x_hat, x)
                self.recon_opt.zero_grad()
                recon_loss.backward()
                self.recon_opt.step()

                # === DISCRIMINATOR ===
                z_real = torch.randn(x.size(0), self.latent_dim).to(self.device) * prior_std
                d_real = self.discriminator(z_real)
                d_real_loss = self.adv_loss(d_real, torch.ones_like(d_real))

                z_fake = self.encoder(x).detach()
                d_fake = self.discriminator(z_fake)
                d_fake_loss = self.adv_loss(d_fake, torch.zeros_like(d_fake))

                disc_loss = d_real_loss + d_fake_loss
                self.disc_opt.zero_grad()
                disc_loss.backward()
                self.disc_opt.step()

                # === GENERATOR ===
                z = self.encoder(x)
                d_pred = self.discriminator(z)
                gen_loss = self.adv_loss(d_pred, torch.ones_like(d_pred))
                self.gen_opt.zero_grad()
                gen_loss.backward()
                self.gen_opt.step()

                total_recon_loss += recon_loss.item()
                total_disc_loss += disc_loss.item()
                total_gen_loss += gen_loss.item()

            # tqdm.write(f"Epoch ({epoch + 1}/{epochs}):\t"
            #            f"Recon Loss: {total_recon_loss / len(data_loader):.4f}|\t"
            #            f"Disc Loss: {total_disc_loss / len(data_loader):.4f}|\t"
            #            f"Gen Loss: {total_gen_loss / len(data_loader):.4f}|\t"
            #            )
            avg_recon_loss = total_recon_loss / len(data_loader)
            avg_disc_loss = total_disc_loss / len(data_loader)
            avg_gen_loss = total_gen_loss / len(data_loader)
            with open(csv_log_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([epoch + 1, avg_recon_loss, avg_disc_loss, avg_gen_loss])

            wandb.log({
                "epoch": epoch + 1,
                "recon_loss": avg_recon_loss,
                "disc_loss": avg_disc_loss,
                "gen_loss": avg_gen_loss,
                "lr": self.recon_opt.param_groups[0]['lr']
            })
            # tqdm.write(
            #     f"Epoch {epoch + 1}/{epochs} - Recon: {avg_recon_loss:.4f} | Disc: {avg_disc_loss:.4f} | Gen: {avg_gen_loss:.4f}")

    def save_weights(self, path_prefix="aae_weights"):
        save_weights(self.encoder, self.decoder, self.discriminator, path_prefix=path_prefix)

    def load_weights(self, path_prefix="aae_weights"):
        load_weights(self.encoder, self.decoder, self.discriminator, self.device, path_prefix=path_prefix)



