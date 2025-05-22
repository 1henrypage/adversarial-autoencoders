import torch
import torch.nn as nn
import torch.nn.functional as F
import weakref

from adversarial import Decoder, weights_init

from semisupervised import (
  SemiSupervisedAdversarialAutoencoder,
  SemiSupervisedAutoEncoderOptions,
)

class ReconLossWithPenalty(nn.Module):
    """Wraps the original reconstruction loss and adds the cluster-head penalty."""

    def __init__(self, base_loss: nn.Module, owner: "DimensionalityReductionAAE"):
        super().__init__()
        self.base_loss = base_loss
        object.__setattr__(self, "_owner_ref", weakref.ref(owner))

    def forward(self, x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        owner = self._owner_ref()
        return self.base_loss(x_hat, x) + owner._cluster_head_penalty()

class DimensionalityReductionAAE(SemiSupervisedAdversarialAutoencoder):
    def __init__(
            self,
            options: SemiSupervisedAutoEncoderOptions,
            *,
            eta: float = 1.0, # Minimum allowed distance between heads
            lambda_ch: float = 1.0, # Weight of the cluster-head distance penalty
    ):
        super().__init__(options)

        # W_C Head
        self.WC = nn.Parameter(torch.randn(
            options.latent_dim_categorical, options.latent_dim_style
            ))
        
        self.decoder = Decoder(
            options.latent_dim_style, 
            options.ae_hidden_dim, 
            options.input_dim, 
            options.use_decoder_sigmoid
            ).to(options.device)
        self.decoder.apply(weights_init)
        # Add to optimiser
        self.recon_opt = torch.optim.AdamW(
            list(self.encoder.parameters()) + list(self.decoder.parameters()) + [self.WC],
            lr=options.init_recon_lr,
        )

        self.eta = eta
        self.lambda_ch = lambda_ch

        self.recon_loss = ReconLossWithPenalty(self.recon_loss, self)

    def _compose_latent(self, z_cat: torch.Tensor, z_style: torch.Tensor):
        """Create the *n-dimensional* latent used by the decoder."""
        z_cluster = z_cat @ self.WC
        z_total = z_cluster + z_style
        return z_total
    
    def forward_reconstruction(self, x):
        """Auto-encoder path used both for training and inference."""
        z_cat, z_style = self.forward_encoder(x)
        x_hat = self.decoder(self._compose_latent(z_cat, z_style))
        return x_hat
    
    # cluster-head regulariser
    def _cluster_head_penalty(self):
        diff = self.WC.unsqueeze(0) - self.WC.unsqueeze(1)
        dists = (diff.pow(2).sum(-1) + 1e-8).sqrt()
        mask = (dists < self.eta).float()
        penalty = ((self.eta - dists) * mask).sum() / (self.options.latent_dim_categorical ** 2)
        return self.lambda_ch * penalty
    
    def generate_images(self, labels, style_z=None, prior_std=5.0):
        """Same as parent, but uses the additive latent."""
        self.eval()
        with torch.no_grad():
            if isinstance(labels, int):
                labels = torch.tensor([labels], device=self.device)
            else:
                labels = labels.to(self.device)
            batch = labels.size(0)
            z_cat = F.one_hot(labels, num_classes=self.options.latent_dim_categorical).float()
            if style_z is None:
                style_z = self.sample_latent_prior_gaussian(batch, prior_std)
            z_total = self._compose_latent(z_cat, style_z)
            x_hat = self.decoder(z_total)
            return x_hat.view(batch, 1, 28, 28)
        
