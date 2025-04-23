
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class Encoder(nn.Module):
    """
    The encoder class for the AAE. This is synonymous to the generator of the GAN component.
    """

    def __init__(self, input_dim: int, ae_hidden: int, output_dim: int) -> None:
        """
        :param input_dim: Input dimension of the encoder
        :param ae_hidden: hidden dimension of the encoder (noted as generator to keep consistency)
        :param output_dim: The latent dimension
        """
        super(Encoder, self).__init__()
        self.fc =nn.Sequential(
            nn.Linear(input_dim, ae_hidden),
            nn.ReLU(),
            nn.Linear(ae_hidden, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass on the encoder

        :param x: The input to the encoder
        :return: The latent representation of the input (the paper says this is the hidden code).
        """
        return self.fc(x)


class Decoder(nn.Module):
    """
    The decoder class for the AAE.
    """

    def __init__(self, input_dim: int, ae_hidden: int, output_dim: int, use_sigmoid: bool) -> None:
        """
        :param input_dim: The input dimension should be latent dim
        :param ae_hidden: hidden dimension (same as enc)
        :param output_dim: Should be the input of the enc.
        :param use_sigmoid: If true, use sigmoid to constrain the output to probabilities.
        """

        super(Decoder, self).__init__()
        layers = [
            nn.Linear(input_dim, ae_hidden),
            nn.ReLU(),
            nn.Linear(ae_hidden, output_dim)
        ]
        if use_sigmoid:
            layers.append(nn.Sigmoid())
        self.fc = nn.Sequential(*layers)


    def forward(self, z):
        """
        Forward pass on the decoder

        :param z: The latent representation of the input.
        :return: The reconstructed input
        """
        return self.fc(z)

class Discriminator(nn.Module):
    """
    Component responsible for adversarial loss. This acts as a regularisation on the autoencoder.
    """
    def __init__(self, dc_hidden, latent_dim):
        """
        :param dc_hidden: The hidden dimension of the discriminator
        :param latent_dim: The latent dimension
        """

        super(Discriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, dc_hidden),
            nn.ReLU(),
            nn.Linear(dc_hidden,1),
        )

    def forward(self, z):
        """
        Forward pass on the discriminator
        :param z: The latent representation of the input.
        :return: 0 if latent representation is not stemming from $p(z)$ 0 otherwise.
        """
        return self.fc(z).squeeze()


class AdversarialAutoencoder(nn.Module):

    def __init__(self,
                 input_dim: int,
                 ae_hidden: int,
                 dc_hidden: int,
                 latent_dim: int,
                 recon_loss_fn: nn.Module,
                 lr: float = 2e-4,
                 use_decoder_sigmoid: bool = True,
                 device: str = "cuda"
                 ) -> None:
        super(AdversarialAutoencoder, self).__init__()
        self.encoder = Encoder(
            input_dim=input_dim,
            ae_hidden=ae_hidden,
            output_dim=latent_dim
        ).to(device)

        self.decoder = Decoder(
            input_dim=latent_dim,
            ae_hidden=ae_hidden,
            output_dim=input_dim,
            use_sigmoid=use_decoder_sigmoid
        ).to(device)

        self.discriminator = Discriminator(
            dc_hidden=dc_hidden,
            latent_dim=latent_dim
        ).to(device)

        self.ae_opt = optim.Adam(
            list(self.encoder.parameters()) + list(self.decode.parameters()),
            lr=lr,
        )

        self.dc_opt = optim.Adam(
            self.discriminator.parameters(),
            lr=lr
        )

        self.recon_loss = recon_loss_fn
        self.adv_loss = nn.BCEWithLogitsLoss()

    def step_autoencoder(self, x):
        self.ae_opt.zero_grad()
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = self.recon_loss(x_hat, x.view(x.size(0), -1)) #TODO
        loss.backward()
        self.ae_opt.step()
        return z.detach(), loss.item()

    def step_discriminator(self, ):






