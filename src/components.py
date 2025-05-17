import torch
from torch import nn


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