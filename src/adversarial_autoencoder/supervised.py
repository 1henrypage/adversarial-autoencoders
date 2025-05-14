import torch
import torch.nn.functional as F
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm

from src.adversarial_autoencoder.components import save_weights, load_weights, weights_init


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
          # Concatenate style z and class y
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
        self.train()
        for epoch in tqdm(range(epochs), desc="Training Epochs", unit="epoch"):

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

            for batch_idx, (x, labels) in enumerate(data_loader):
                x = x.to(self.device)
                labels = labels.to(self.device)
                y_onehot = F.one_hot(labels, num_classes=self.num_classes).float()
                # print(f"Labels: {labels[:4]}")
                # print(f"One-hot: {y_onehot[:4]}")
                # with torch.no_grad():
                #     z_inspect = self.encoder(x)
                #     print(f"[Epoch {epoch}] z mean: {z_inspect.mean():.4f}, z std: {z_inspect.std():.4f}")

                # # === Add Gaussian noise during training ===
                if self.training:  # Ensure noise is added only during training mode
                    noise = torch.randn_like(x) * 0.3  # Standard deviation of 0.3
                    x = x + noise  # Add the noise to the input

                # === RECONSTRUCTION ===
                self.recon_opt.zero_grad()
                z = self.encoder(x)
                z_cat = torch.cat([z, y_onehot], dim=1)
                x_hat = self.decoder(z_cat)
                recon_loss = self.recon_loss(x_hat, x)
                recon_loss.backward()
                self.recon_opt.step()

                # if batch_idx == 1:  # Display every 100 batches
                #     output_img = x_hat[0].detach().cpu().view(1, 28, 28)  # Detach from graph, move to CPU, reshape to 28x28
                #     plt.imshow(output_img.squeeze(), cmap="gray")  # Squeeze to remove unnecessary dimension
                #     plt.title(f"Reconstruction")
                #     plt.axis('off')
                #     plt.show()

                # === DISCRIMINATOR ===
                self.disc_opt.zero_grad()
                z_real = torch.randn(x.size(0), self.latent_dim).to(self.device) * prior_std
                d_real = self.discriminator(z_real)
                d_real_loss = self.adv_loss(d_real, torch.ones_like(d_real))

                z_fake = self.encoder(x).detach()
                d_fake = self.discriminator(z_fake)
                d_fake_loss = self.adv_loss(d_fake, torch.zeros_like(d_fake))

                disc_loss = (d_real_loss + d_fake_loss)/2
                disc_loss.backward()
                self.disc_opt.step()

                # === GENERATOR ===
                self.gen_opt.zero_grad()
                z = self.encoder(x)
                d_pred = self.discriminator(z)
                gen_loss = self.adv_loss(d_pred, torch.ones_like(d_pred))
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

    def generate_sample(self, x, label, prior_std: float = 1.0) -> torch.Tensor:
        """Generate samples for a specific class
        Args:
            n: Number of samples to generate
            class_id: Class ID for which to generate samples
            prior_std: Standard deviation for the Gaussian prior
        Returns:
            Tensor of generated images with shape (n, 1, 28, 28)
        """
        with (torch.no_grad()):
            x = x.to(self.device)
            label = label.to(self.device)
            y_onehot = F.one_hot(label, num_classes=self.num_classes).float()

            self.disc_opt.zero_grad()
            z = torch.randn(x.size(0), self.latent_dim).to(self.device) * prior_std
            z_cat = torch.cat([z, y_onehot], dim=1)
            sample = self.decoder(z_cat)

            return sample.view(-1, 1, 28, 28)

    def generate_sample_from_test(self, test_loader, n_variations=5, prior_std=5.0):
        """Generate sample variations for one test image"""
        self.eval()

        # Get one test sample
        test_images, test_labels = next(iter(test_loader))
        test_image = test_images[1].to(self.device)  # Keep as batch of 1
        test_image = test_image.view(test_image.size(0), -1)  # Flatten the image
        test_label = test_labels[1].to(self.device)
        y_onehot = F.one_hot(test_label, num_classes=self.num_classes).float()

        # Get deterministic encoding
        with torch.no_grad():
            z_original = self.encoder(test_image)

            # Generate variations by perturbing the latent code
            samples = []
            for _ in range(n_variations):
                # Create slightly perturbed latent code
                z_perturbed = z_original + torch.randn_like(z_original) * prior_std
                # z = torch.randn(n, self.encoder.fc[-1].out_features).to(self.device) * prior_std
                z_cat = torch.cat([z_perturbed, y_onehot], dim=1)
                sample = self.decoder(z_cat)
                samples.append(sample)

            # Add original reconstruction
            z_cat = torch.cat([z_original, y_onehot], dim=1)
            original_recon = self.decoder(z_cat)
            samples.insert(0, original_recon)

        # Prepare visualization
        samples = torch.cat(samples).view(-1, 1, 28, 28)
        original_image = test_image.view(1, 28, 28).cpu()

        # Plot comparison
        plt.figure(figsize=(n_variations + 2, 2))

        # Show original
        plt.subplot(1, n_variations + 2, 1)
        plt.imshow(original_image[0], cmap='gray')
        plt.title("Original")
        plt.axis('off')

        # Show reconstruction
        plt.subplot(1, n_variations + 2, 2)
        plt.imshow(samples[0].permute(1, 2, 0).cpu(), cmap='gray')
        plt.title("Recon")
        plt.axis('off')

        # Show variations
        for i in range(1, n_variations + 1):
            plt.subplot(1, n_variations + 2, i + 2)
            plt.imshow(samples[i].permute(1, 2, 0).cpu(), cmap='gray')
            plt.title(f"Var {i}")
            plt.axis('off')

        plt.suptitle(f"Class {test_label.item()} Samples")
        plt.tight_layout()
        plt.show()

        return samples


    def save_weights(self, path_prefix="aae_weights"):
        save_weights(self.encoder, self.decoder, self.discriminator, path_prefix=path_prefix)

    def load_weights(self, path_prefix="aae_weights"):
        load_weights(self.encoder, self.decoder, self.discriminator, self.device, path_prefix=path_prefix)