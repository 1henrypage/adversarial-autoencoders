import torch
import torch.nn as nn
import torch.nn.functional as F
import weakref
import csv
import os
from tqdm.auto import tqdm
from itertools import cycle
import torch.optim as optim
import numpy as np

from aae import Decoder, weights_init
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment

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
            options.latent_dim_categorical, 
            options.latent_dim_style,
            device=options.device
            )
        )
        
        self.decoder = Decoder(
            options.latent_dim_style, 
            options.ae_hidden_dim, 
            options.input_dim, 
            options.use_decoder_sigmoid
            ).to(options.device)
        self.decoder.apply(weights_init)

        self.recon_opt = optim.SGD(
            list(self.encoder.parameters()) +
            list(self.decoder.parameters()) + [self.WC],
            lr=0.01,       momentum=0.9)

        self.semi_supervised_opt = optim.SGD(
            list(self.encoder.parameters()),
            lr=0.10,       momentum=0.9)

        self.gen_cat_opt   = optim.SGD(list(self.encoder.parameters()),
                                       lr=0.10, momentum=0.9)
        self.gen_style_opt = optim.SGD(list(self.encoder.parameters()),
                                       lr=0.10, momentum=0.9)

        self.disc_cat_opt  = optim.SGD(self.discriminator_categorical.parameters(),
                                       lr=0.10, momentum=0.1)
        self.disc_style_opt= optim.SGD(self.discriminator_style.parameters(),
                                       lr=0.10, momentum=0.1)
        
        self.adv_loss = nn.BCEWithLogitsLoss(reduction="mean")

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
    
    @staticmethod
    def _hungarian_cluster_acc(y_true, y_pred, K):
        """
        Return the best-matching accuracy between cluster ids and true labels.
        """
        cm = np.zeros((K, K), dtype=np.int64)
        for t, p in zip(y_true, y_pred):
            cm[p, t] += 1

        row_ind, col_ind = linear_sum_assignment(cm.max() - cm)
        return cm[row_ind, col_ind].sum() / y_true.size

    @torch.no_grad()
    def evaluate_unsupervised(self, loader):
        """
        Compute Hungarian accuracy, NMI and ARI on a labelled *loader*
        – but **never** uses the labels during training.
        """
        self.eval()
        preds, trues = [], []
        for x, y in loader:
            if x.dim() > 2:
                x = x.view(x.size(0), -1)
            x = x.to(self.device)
            logits, _ = self.forward_no_softmax(x)
            preds.append(logits.argmax(1).cpu().numpy())
            trues.append(y.numpy())

        y_pred = np.concatenate(preds)
        y_true = np.concatenate(trues)
        K      = self.options.latent_dim_categorical

        acc = self._hungarian_cluster_acc(y_true, y_pred, K)
        nmi = normalized_mutual_info_score(y_true, y_pred)
        ari = adjusted_rand_score(y_true, y_pred)
        print("unique preds:", np.unique(y_pred, return_counts=True))
        return acc, nmi, ari
    
    def train_mbgd_2(
        self,
        train_labeled_loader,
        val_loader,
        epochs,
        result_folder: str,
        prior_std=5.0,
        add_gaussian_noise=False,
        train_unlabeled_loader=None,
    ):
        
        assert train_unlabeled_loader is not None, \
        "Need an unlabeled loader (the outer loop runs on it)"


        os.makedirs(result_folder, exist_ok=True)
        with open(f'{result_folder}/train_log.csv', 'w', newline='') as f:
            csv.writer(f).writerow(
                ['epoch', 'Recon_U', 'Recon_L', 'SemiSup',
                'Disc_Cat', 'Gen_Cat', 'Disc_Sty', 'Gen_Sty']
            )

        # Endless iterator over the tiny labeled pool
        unsupervised = train_labeled_loader is None or len(train_labeled_loader.dataset) == 0
        if not unsupervised:
            labeled_iter = cycle(train_labeled_loader)

        def _set_lr(epoch_idx: int):
            """Piece-wise step LR schedule: 0 / 50 / 1000 epochs."""
            if   epoch_idx < 50:    factor = 1.0          #   0 –  49
            elif epoch_idx < 1000:  factor = 0.1          #  50 – 999
            else:                   factor = 0.01         # ≥ 1000

            # Paper uses    recon 0.01 ·f,   semi-sup 0.1 ·f,   adv 0.1 ·f
            lr_recon = 0.01  * factor
            lr_cls   = 0.10  * factor
            lr_adv   = 0.10  * factor

            for param_group in self.recon_opt.param_groups:          param_group["lr"] = lr_recon
            for param_group in self.semi_supervised_opt.param_groups:param_group["lr"] = lr_cls
            for opt in (self.gen_cat_opt, self.gen_style_opt,
                        self.disc_cat_opt, self.disc_style_opt):
                for param_group in opt.param_groups:
                    param_group["lr"] = lr_adv


        # ------------------------------------------------------------------ TRAIN
        for epoch in range(epochs):
            _set_lr(epoch)
            self.train()

            # ---------------------------------------------------------------- loop over *UNLABELED* loader
            sums = dict(recon_u=0, recon_l=0, cls=0,
                        d_cat=0, g_cat=0, d_sty=0, g_sty=0)

            loop = tqdm(train_unlabeled_loader,
                        desc=f"Epoch [{epoch+1}/{epochs}]",
                        leave=False)
            
            for batch_idx, (x_u, _) in enumerate(loop, 1):
                # ---------- fetch one unlabeled batch ----------
                # x_l, y_l = next(labeled_iter)
                # x_u, x_l, y_l = (t.to(self.device)
                #                 for t in (x_u, x_l, y_l))
                if not unsupervised:
                    x_l, y_l = next(labeled_iter)
                    x_u, x_l, y_l = (t.to(self.device)
                                    for t in (x_u, x_l, y_l))
                else:
                    # create a zero-sized tensor so the cat/style encoder still works
                    x_u = x_u.to(self.device)
                    x_l = x_u[0:0]                 # shape (0, …)
                    y_l = None

                # Optional Gaussian noise on inputs
                x_l_tgt, x_u_tgt = x_l.clone(), x_u.clone()
                if add_gaussian_noise:
                    x_l = x_l + torch.randn_like(x_l) * 0.3
                    x_u = x_u + torch.randn_like(x_u) * 0.3

                # =================================================
                # 1) Reconstruction phase  - LABELED + UNLABELED
                # =================================================
                x_hat_u = self.forward_reconstruction(x_u)
                x_hat_l = self.forward_reconstruction(x_l)
                loss_rec_u = self.recon_loss(x_hat_u, x_u_tgt)
                loss_rec_l = self.recon_loss(x_hat_l, x_l_tgt)
                loss_rec   = loss_rec_u + loss_rec_l

                self.recon_opt.zero_grad()
                loss_rec.backward()
                self.recon_opt.step()

                # =================================================
                # 2) Latent regularisation phase (GAN)  – BOTH batches
                # =================================================

                # put BOTH batches through the encoder once
                x_comb = torch.cat([x_u, x_l], dim=0)
                z_fake_cat, z_fake_sty = self.forward_encoder(x_comb)

                # ---- cat adv update ----
                bs = z_fake_cat.size(0)
                z_real_cat = self.sample_latent_prior_categorical(bs)

                d_real_cat = self.adv_loss(
                    self.discriminator_categorical(z_real_cat),
                    torch.ones(bs, device=self.device))

                d_fake_cat = self.adv_loss(
                    self.discriminator_categorical(z_fake_cat.detach()),
                    torch.zeros(bs, device=self.device))

                self.disc_cat_opt.zero_grad()
                (d_real_cat + d_fake_cat).backward()
                self.disc_cat_opt.step()

                g_cat = self.adv_loss(
                    self.discriminator_categorical(
                        self.forward_encoder(x_comb)[0]   # fresh forward pass
                    ),
                    torch.ones(bs, device=self.device))
                
                self.gen_cat_opt.zero_grad()
                g_cat.backward()
                self.gen_cat_opt.step()

                # ---- style adv update ----
                z_real_sty = self.sample_latent_prior_gaussian(bs, prior_std)

                d_real_sty = self.adv_loss(
                    self.discriminator_style(z_real_sty),
                    torch.ones(bs, device=self.device))

                d_fake_sty = self.adv_loss(
                    self.discriminator_style(z_fake_sty.detach()),
                    torch.zeros(bs, device=self.device))

                self.disc_style_opt.zero_grad()
                (d_real_sty + d_fake_sty).backward()
                self.disc_style_opt.step()

                g_sty = self.adv_loss(
                    self.discriminator_style(
                        self.forward_encoder(x_comb)[1]   # fresh forward pass
                    ),
                    torch.ones(bs, device=self.device))

                self.gen_style_opt.zero_grad()
                g_sty.backward()
                self.gen_style_opt.step()

                # =================================================
                # 3) Semi-supervised classification - only labeled samples
                # =================================================
                # logits_cat, _ = self.forward_no_softmax(x_l)

                # # λ: keep CE roughly same scale as unsup terms
                # lambda_cls = x_u.size(0) / x_l.size(0)
                # loss_cls = self.semi_supervised_loss(logits_cat, y_l) * lambda_cls

                # self.semi_supervised_opt.zero_grad()
                # loss_cls.backward()
                # self.semi_supervised_opt.step()
                if not unsupervised:
                    logits_cat, _ = self.forward_no_softmax(x_l)
                    lambda_cls  = x_u.size(0) / x_l.size(0)
                    loss_cls    = self.semi_supervised_loss(logits_cat, y_l) * lambda_cls

                    self.semi_supervised_opt.zero_grad()
                    loss_cls.backward()
                    self.semi_supervised_opt.step()
                else:
                    loss_cls = torch.tensor(0.0, device=self.device)  # for logging only

                # ---------------------------------------------------------- logging-in-epoch
                sums['recon_u'] += loss_rec_u.item()
                sums['recon_l'] += loss_rec_l.item()
                sums['cls']     += loss_cls.item()
                sums['d_cat']   += (d_real_cat + d_fake_cat).item()
                sums['g_cat']   += g_cat.item()
                sums['d_sty']   += (d_real_sty + d_fake_sty).item()
                sums['g_sty']   += g_sty.item()

                loop.set_postfix({
                    'Recon_U': sums['recon_u']/batch_idx,
                    'Recon_L': sums['recon_l']/batch_idx,
                    'SemiSup': sums['cls']    /batch_idx,
                    'Disc_Cat':sums['d_cat']  /batch_idx,
                    'Gen_Cat': sums['g_cat']  /batch_idx,
                    'Disc_Sty':sums['d_sty']  /batch_idx,
                    'Gen_Sty': sums['g_sty']  /batch_idx
                })

            n_batches = len(train_unlabeled_loader)
            avg = {k: v/n_batches for k, v in sums.items()}
            print(
                f"Epoch {epoch+1}/{epochs} — "
                f"Recon_U: {avg['recon_u']:.4f}, Recon_L: {avg['recon_l']:.4f}, "
                f"SemiSup: {avg['cls']:.4f}, "
                f"Disc_Cat: {avg['d_cat']:.4f}, Gen_Cat: {avg['g_cat']:.4f}, "
                f"Disc_Sty: {avg['d_sty']:.4f}, Gen_Sty: {avg['g_sty']:.4f}"
            )

            with open(f'{result_folder}/train_log.csv', 'a', newline='') as f:
                csv.writer(f).writerow(
                    [epoch+1, avg['recon_u'], avg['recon_l'], avg['cls'],
                    avg['d_cat'], avg['g_cat'], avg['d_sty'], avg['g_sty']]
                )

            if unsupervised:
                acc, nmi, ari = self.evaluate_unsupervised(val_loader)
                print(f"[VAL]  Hungarian-ACC: {acc*100:5.2f}% | NMI: {nmi:.4f} | ARI: {ari:.4f}\n")
            else:
                self.eval()
                correct = total = 0
                with torch.no_grad():
                    for vx, vy in val_loader:
                        vx = vx.view(vx.size(0), -1).to(self.device)
                        vy = vy.to(self.device)
                        preds = self.forward_no_softmax(vx)[0].argmax(1)
                        correct += (preds == vy).sum().item()
                        total   += vy.size(0)
                print(f"[VAL]  Accuracy: {100*correct/total:.2f}%\n")

            if (epoch+1) % 50 == 0:
                ckpt_dir = f'{result_folder}/weights_epoch_{epoch+1}'
                os.makedirs(ckpt_dir, exist_ok=True)
                self.save_weights(f'{ckpt_dir}/weights')
            self.save_weights(f'{result_folder}/weights')
    
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
        
    @torch.no_grad()
    def embed(self, x: torch.Tensor, *, return_parts: bool = False):
        """
        Parameters
        ----------
        x : (B, C, H, W) tensor
            A batch of input images.
        return_parts : bool, default False
            If True, also return (z_cat, z_style).

        Returns
        -------
        z      : (B, latent_dim)  –  W_c·y  +  z_style
        z_cat  : (B, K)           –  softmax/categorical part (optional)
        z_style: (B, d_style)     –  style part (optional)
        """
        z_cat, z_style = self.forward_encoder(x)
        z = self._compose_latent(z_cat, z_style)     # additive mix
        if return_parts:
            return z, z_cat, z_style
        return z
        
