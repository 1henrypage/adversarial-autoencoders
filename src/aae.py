
import torch
import torch.nn as nn

from src.utils import load_weights, save_weights, weights_init
from src.components import Encoder, Decoder, Discriminator


class AdversarialAutoencoder(nn.Module):
    def __init__(self, input_dim, ae_hidden, dc_hidden, latent_dim, recon_loss_fn, init_recon_lr, init_gen_lr, init_disc_lr, use_decoder_sigmoid=True, device="cuda"):
        super(AdversarialAutoencoder, self).__init__()
        self.device = device
        self.encoder = Encoder(input_dim, ae_hidden, latent_dim).to(device)
        self.decoder = Decoder(latent_dim, ae_hidden, input_dim, use_decoder_sigmoid).to(device)
        self.discriminator = Discriminator(dc_hidden, latent_dim).to(device)

        self.encoder.apply(weights_init)
        self.decoder.apply(weights_init)
        self.discriminator.apply(weights_init)

        self.recon_opt = torch.optim.SGD(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=init_recon_lr,
            momentum=0.9
        )

        self.gen_opt = torch.optim.SGD(
            self.encoder.parameters(),
            lr=init_gen_lr,
            momentum=0.1
        )

        self.disc_opt = torch.optim.SGD(
            self.discriminator.parameters(),
            lr=init_disc_lr,
            momentum=0.1
        )

        self.recon_loss = recon_loss_fn
        self.adv_loss = nn.BCEWithLogitsLoss()


    # we assume gaussian prior, if you want to change this, change it.
    def train_mbgd(self, data_loader, epochs, prior_std=5.0):
        for epoch in range(epochs):

            # adjust this if your experiment does different dynamic LRs
            if epoch == 50:
                self.recon_opt.param_groups[0]['lr'] = 0.001
                self.gen_opt.param_groups[0]['lr'] = 0.01
                self.disc_opt.param_groups[0]['lr'] = 0.01
            elif epoch == 1000:
                self.recon_opt.param_groups[0]['lr'] = 0.0001
                self.gen_opt.param_groups[0]['lr'] = 0.001
                self.disc_opt.param_groups[0]['lr'] = 0.001


            total_recon_loss = 0
            total_disc_loss = 0
            total_gen_loss = 0

            for batch_idx, (x, _) in enumerate(data_loader):
                x = x.to(self.device)

                # === RECON PHASE ====
                self.recon_opt.zero_grad()
                z = self.encoder(x)
                x_hat = self.decoder(z)
                recon_loss = self.recon_loss(x_hat, x)
                recon_loss.backward()
                self.recon_opt.step()

                # === DISCRIMINATOR REGULARISATION ===
                self.disc_opt.zero_grad()
                z_real = torch.randn(x.size(0), z.size(1)).to(self.device) * prior_std
                d_real = self.discriminator(z_real)
                d_real_loss = self.adv_loss(d_real, torch.ones_like(d_real))

                z_fake = self.encoder(x).detach()
                d_fake = self.discriminator(z_fake)
                d_fake_loss = self.adv_loss(d_fake, torch.zeros_like(d_fake))

                disc_loss = d_real_loss + d_fake_loss
                disc_loss.backward()
                self.disc_opt.step()

                #  === GENERATOR REGULARISATION ===
                self.gen_opt.zero_grad()
                z = self.encoder(x)
                d_pred = self.discriminator(z)
                gen_loss = self.adv_loss(d_pred, torch.ones_like(d_pred))
                gen_loss.backward()
                self.gen_opt.step()

                total_recon_loss += recon_loss.item()
                total_disc_loss += disc_loss.item()
                total_gen_loss += gen_loss.item()

            print(f"Epoch ({epoch + 1}/{epochs})\t)"
                  f"Recon Loss: {total_recon_loss / len(data_loader):.4f}\t)"
                  f"Disc Loss: {total_disc_loss / len(data_loader):.4f}\t)"
                  f"Gen Loss: {total_gen_loss / len(data_loader):.4f}\t)"
            )

            if (total_recon_loss / len(data_loader)) < 0.175:
                break


    def generate_samples(self, n: int , prior_std: float = 5.0) -> torch.Tensor:
        with torch.no_grad():
            z = torch.randn(n, self.encoder.fc[-1].out_features).to(self.device) * prior_std
            samples = self.decoder(z)

        return samples

    def load_weights(self, path_prefix="aae_weights"):
        load_weights(self.encoder, self.decoder, self.discriminator, self.device, path_prefix=path_prefix)

    def save_weights(self, path_prefix="aae_weights"):
        save_weights(self.encoder, self.decoder, self.discriminator, path_prefix=path_prefix)
