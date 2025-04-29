

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class Encoder(nn.Module):
    def __init__(self, input_dim: int, ae_hidden: int, output_dim: int) -> None:
        super(Encoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, ae_hidden),
            nn.ReLU(),
            nn.Linear(ae_hidden, ae_hidden),
            nn.ReLU(),
            nn.Linear(ae_hidden, output_dim),  # Linear output
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class Decoder(nn.Module):
    def __init__(self, input_dim: int, ae_hidden: int, output_dim: int, use_sigmoid: bool) -> None:
        super(Decoder, self).__init__()
        layers = [
            nn.Linear(input_dim, ae_hidden),
            nn.ReLU(),
            nn.Linear(ae_hidden, ae_hidden),
            nn.ReLU(),
            nn.Linear(ae_hidden, output_dim)
        ]
        if use_sigmoid:
            layers.append(nn.Sigmoid())
        self.fc = nn.Sequential(*layers)

    def forward(self, z):
        return self.fc(z)


class Discriminator(nn.Module):
    def __init__(self, dc_hidden, latent_dim):
        super(Discriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, dc_hidden),
            nn.ReLU(),
            nn.Linear(dc_hidden, dc_hidden),
            nn.ReLU(),
            nn.Linear(dc_hidden, 1),
        )

    def forward(self, z):
        return self.fc(z).squeeze()


class AdversarialAutoencoder(nn.Module):
    def __init__(self, input_dim, ae_hidden, dc_hidden, latent_dim, recon_loss_fn, lr=2e-4, use_decoder_sigmoid=True, device="cuda"):
        super(AdversarialAutoencoder, self).__init__()
        self.device = device
        self.encoder = Encoder(input_dim, ae_hidden, latent_dim).to(device)
        self.decoder = Decoder(latent_dim, ae_hidden, input_dim, use_decoder_sigmoid).to(device)
        self.discriminator = Discriminator(dc_hidden, latent_dim).to(device)

        self.encoder.apply(weights_init)
        self.decoder.apply(weights_init)
        self.discriminator.apply(weights_init)

        self.ae_opt = optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=lr,
        )

        self.dc_opt = optim.Adam(
            self.discriminator.parameters(),
            lr=lr
        )

        self.recon_loss = recon_loss_fn
        self.adv_loss = nn.BCEWithLogitsLoss()


    # we assume gaussian prior, if you want to change this, change it.
    def train(self, data_loader, epochs, prior_std=5.0):
        for epoch in range(epochs):

            total_recon_loss = 0
            total_disc_loss = 0
            total_gen_loss = 0

            for batch_idx, (x, _) in enumerate(data_loader):
                x = x.to(self.device)

                # recon
                self.ae_opt.zero_grad()
                z = self.encoder(x)
                x_hat = self.decoder(z)

                recon_loss = self.recon_loss(x_hat, x)
                recon_loss.backward()
                self.ae_opt.step()

                # part 1: backprop through discriminator
                self.dc_opt.zero_grad()

                z_real = torch.randn(x.size(0), z.size(1)).to(self.device) * prior_std
                d_real = self.discriminator(z_real)
                d_real_loss = self.adv_loss(d_real, torch.ones_like(d_real))

                z_fake = self.encoder(x).detach()
                d_fake = self.discriminator(z_fake)
                d_fake_loss = self.adv_loss(d_fake, torch.zeros_like(d_fake))

                disc_loss = d_real_loss + d_fake_loss
                disc_loss.backward()
                self.dc_opt.step()

                # part 2: backprop through generator
                self.ae_opt.zero_grad()
                z = self.encoder(x)
                d_pred = self.discriminator(z)
                gen_loss = self.adv_loss(d_pred, torch.ones_like(d_pred))
                gen_loss.backward()
                self.ae_opt.step()

                total_recon_loss += recon_loss.item()
                total_disc_loss += disc_loss.item()
                total_gen_loss += gen_loss.item()

            print(f"Epoch ({epoch + 1}/{epochs})\t)"
                  f"Recon Loss: {total_recon_loss / len(data_loader):.4f}\t)"
                  f"Disc Loss: {total_disc_loss / len(data_loader):.4f}\t)"
                  f"Gen Loss: {total_gen_loss / len(data_loader):.4f}\t)"
            )

    def save_weights(self, path_prefix="aae_weights"):
        """
        Saves the weights of the encoder, decoder, and discriminator.

        Args:
            path_prefix (str): Prefix for the saved file paths. Files will be saved as:
                - <path_prefix>_encoder.pth
                - <path_prefix>_decoder.pth
                - <path_prefix>_discriminator.pth
        """
        torch.save(self.encoder.state_dict(), f"{path_prefix}_encoder.pth")
        torch.save(self.decoder.state_dict(), f"{path_prefix}_decoder.pth")
        torch.save(self.discriminator.state_dict(), f"{path_prefix}_discriminator.pth")
        print(f"Weights saved to {path_prefix}_*.pth")


    def load_weights(self, path_prefix="aae_weights"):
        """
        Loads the weights of the encoder, decoder, and discriminator from files.

        Args:
            path_prefix (str): Prefix for the saved file paths. Expected files are:
                - <path_prefix>_encoder.pth
                - <path_prefix>_decoder.pth
                - <path_prefix>_discriminator.pth
        """
        self.encoder.load_state_dict(torch.load(f"{path_prefix}_encoder.pth", map_location=self.device))
        self.decoder.load_state_dict(torch.load(f"{path_prefix}_decoder.pth", map_location=self.device))
        self.discriminator.load_state_dict(torch.load(f"{path_prefix}_discriminator.pth", map_location=self.device))
        print(f"Weights loaded from {path_prefix}_*.pth")



