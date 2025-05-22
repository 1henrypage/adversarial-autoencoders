import torch
import torch.nn as nn
import torch.nn.functional as F

from semisupervised import (
  SemiSupervisedAdversarialAutoencoder,
  SemiSUpervisedAutoEncoderOptions,
)

class DimensionalityReductionAAE(SemiSupervisedAdversarialAutoencoder):
    def __init(
            self,
            options: SemiSUpervisedAutoEncoderOptions,
            *,
            eta: float = 1.0, # Minimum allowed distance between heads
            lambda_ch: float = 1.0, # Weight of the cluster-head distance penalty
    ):
        super().__init__(options)

        # W_C Head
        self.WC = nn.Parameter(torch.randn(
            options.latent_dim_categorical, options.latent_dim_style
            ))
        
        # Add to optimiser
        self.recon_opt.add_param_group({"params": self.WC})

        self.eta = eta
        self.lambda_ch = lambda_ch

        # Include our penalty in the reconstruction loss
        self._base_recon_loss = self.recon_loss

        def _recon_with_penalty(x_hat, x, *, _self=self):
            return _self._base_recon_loss(x_hat, x) + _self._cluster_head_penalty()

        self.recon_loss = _recon_with_penalty

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
        
