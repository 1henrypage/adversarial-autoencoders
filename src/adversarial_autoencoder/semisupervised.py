import torch
import torch.nn as nn
from tqdm.auto import tqdm
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from adversarial import Encoder, Decoder, Discriminator, weights_init
from experiment import AAEExperimentBase
class SemiSupervisedAutoEncoderOptions(object):
    def __init__(
            self, 
            input_dim, 
            ae_hidden_dim, 
            disc_hidden_dim, 
            latent_dim_categorical, 
            latent_dim_style, 
            recon_loss_fn, 
            init_recon_lr,
            semi_supervised_loss_fn,
            init_semi_sup_lr,
            init_gen_lr, 
            init_disc_categorical_lr,
            init_disc_style_lr,
            use_decoder_sigmoid=True
        ):
        
        # Model dimensions
        self.input_dim = input_dim
        self.ae_hidden_dim = ae_hidden_dim
        self.disc_hidden_dim = disc_hidden_dim
        self.latent_dim_categorical = latent_dim_categorical
        self.latent_dim_style = latent_dim_style

        # Reconstruction loss fn and lr
        self.recon_loss_fn = recon_loss_fn
        self.init_recon_lr = init_recon_lr

        # Classification loss fn and lr
        self.semi_supervised_loss_fn = semi_supervised_loss_fn
        self.init_semi_sup_lr = init_semi_sup_lr

        # Generator / encoder lr
        self.init_gen_lr = init_gen_lr

        # Discriminator lrs
        self.init_disc_categorical_lr = init_disc_categorical_lr
        self.init_disc_style_lr = init_disc_style_lr

        # Decoder options
        self.use_decoder_sigmoid = use_decoder_sigmoid

        # Device for computation
        self.device = self._get_device()

    def _get_device(self):
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'


    def __repr__(self):
        return (
            f"<SemiSupervisedAutoEncoderOptions("  \
            f"input_dim={self.input_dim}, "  \
            f"encoder_hidden_dim={self.ae_hidden_dim}, "  \
            f"decoder_hidden_dim={self.disc_hidden_dim}, "  \
            f"latent_dim_categorical={self.latent_dim_categorical}, "  \
            f"latent_dim_style={self.latent_dim_style}, "  \
            f"use_decoder_sigmoid={self.use_decoder_sigmoid}, "  \
            f"device={self.device})>"
        )


class SemiSupervisedAdversarialAutoencoder(AAEExperimentBase, nn.Module):
    def __init__(self, options: SemiSupervisedAutoEncoderOptions, plot_separately: bool = False):
        
        super().__init__(plot_separately=plot_separately)
        nn.Module.__init__(self)

        self.options = options

        self.device = options.device
        self.encoder = Encoder(options.input_dim, options.ae_hidden_dim, options.latent_dim_categorical + options.latent_dim_style).to(options.device)
        self.decoder = Decoder(options.latent_dim_categorical + options.latent_dim_style, options.ae_hidden_dim, options.input_dim, options.use_decoder_sigmoid).to(options.device)
        self.cat_softmax = nn.Softmax(dim=1)

        self.discriminator_categorical = Discriminator(options.disc_hidden_dim, options.latent_dim_categorical).to(options.device)
        self.discriminator_style = Discriminator(options.disc_hidden_dim, options.latent_dim_style).to(options.device)

        self.encoder.apply(weights_init)
        self.decoder.apply(weights_init)
        self.discriminator_categorical.apply(weights_init)
        self.discriminator_style.apply(weights_init)

        # optimazer for reconstruction phase
        self.recon_opt = torch.optim.SGD(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=options.init_recon_lr,
            momentum=0.9
        )

        # optimizer for semi-supervised phase
        self.semi_supervised_opt = torch.optim.SGD(
            self.encoder.parameters(),
            lr=options.init_semi_sup_lr,
            momentum=0.9
        )

        # optimizer for generative phase
        self.gen_cat_opt = torch.optim.SGD(
            self.encoder.parameters(),
            lr=options.init_gen_lr,
            momentum=0.1
        )

        # optimizer for generative phase
        self.gen_style_opt = torch.optim.SGD(
            self.encoder.parameters(),
            lr=options.init_gen_lr,
            momentum=0.1
        )

        # optimizer for categorical discriminator 
        self.disc_cat_opt = torch.optim.SGD(
            self.discriminator_categorical.parameters(),
            lr=options.init_disc_categorical_lr,
            momentum=0.1
        )

        # optimizer for style discriminator
        self.disc_style_opt = torch.optim.SGD(
            self.discriminator_style.parameters(),
            lr=options.init_disc_style_lr,
            momentum=0.1
        )

        self.recon_loss = options.recon_loss_fn
        self.semi_supervised_loss = options.semi_supervised_loss_fn
        self.adv_loss = nn.BCEWithLogitsLoss()


    def forward_reconstruction(self, x):
        z_cat, z_style = self.forward_encoder(x)
        x_hat = self.decoder(torch.cat((z_cat, z_style), dim=1))
        return x_hat
    
    def forward_encoder(self, x):
        z = self.encoder(x)
        z_cat = self.cat_softmax(z[:, :self.options.latent_dim_categorical])
        # z_cat = z[:, :self.options.latent_dim_categorical]
        z_style = z[:, self.options.latent_dim_categorical:]
        return z_cat, z_style
    
    def forward_no_softmax(self, x):
        z = self.encoder(x)
        z_cat = z[:, :self.options.latent_dim_categorical]
        z_style = z[:, self.options.latent_dim_categorical:]
        return z_cat, z_style
        
    def sample_latent_prior_gaussian(self, n: int , prior_std: float = 5.0) -> torch.Tensor:
        return torch.randn(n, self.options.latent_dim_style).to(self.device) * prior_std
    
    def sample_latent_prior_categorical(self, n: int) -> torch.Tensor:
        latent_dim = self.options.latent_dim_categorical
        labels = torch.randint(0, latent_dim, (n,), device=self.device)
        return F.one_hot(labels, num_classes=latent_dim).float().to(self.device)


    def train_mbgd(self, train_loader, val_loader, epochs, prior_std=5.0):
        for epoch in range(epochs):

            # adjust this if your experiment does different dynamic LRs
            if epoch == 50:
                self.recon_opt.param_groups[0]['lr'] = 0.001
                self.semi_supervised_opt.param_groups[0]['lr'] = 0.001
                self.gen_style_opt.param_groups[0]['lr'] = 0.01
                self.gen_cat_opt.param_groups[0]['lr'] = 0.01
                self.disc_style_opt.param_groups[0]['lr'] = 0.01
                self.disc_cat_opt.param_groups[0]['lr'] = 0.01
                
            elif epoch == 1000:
                self.recon_opt.param_groups[0]['lr'] = 0.0001
                self.semi_supervised_opt.param_groups[0]['lr'] = 0.0001
                self.gen_style_opt.param_groups[0]['lr'] = 0.001
                self.gen_cat_opt.param_groups[0]['lr'] = 0.001
                self.disc_style_opt.param_groups[0]['lr'] = 0.001
                self.disc_cat_opt.param_groups[0]['lr'] = 0.001


            total_recon_loss = 0
            total_semi_supervised_loss = 0

            total_disc_style_loss = 0
            total_gen_style_loss = 0

            total_disc_cat_loss = 0
            total_gen_cat_loss = 0

            self.train()
            loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]", leave=False)
            self.dummy_loop(epoch)
            for batch_idx, (x, y) in enumerate(loop, start=1):
                x, y = x.to(self.device), y.to(self.device)

                # === RECONSTRUCTION PHASE ====
                self.recon_opt.zero_grad()
                x_hat = self.forward_reconstruction(x)
                recon_loss = self.recon_loss(x_hat, x)
                recon_loss.backward()
                self.recon_opt.step()

                # === CATEGORICAL DISCRIMINATOR REGULARISATION ===
                self.disc_cat_opt.zero_grad()
                z_real_cat = self.sample_latent_prior_categorical(x.size(0))
                d_real_loss_cat = self.adv_loss(self.discriminator_categorical(z_real_cat), torch.ones(x.size(0), device=self.device))
                z_fake_cat, _ = self.forward_encoder(x)
                d_fake_loss_cat = self.adv_loss(self.discriminator_categorical(z_fake_cat.detach()), torch.zeros(x.size(0), device=self.device))
                (d_real_loss_cat + d_fake_loss_cat).backward()
                self.disc_cat_opt.step()

                # === STYLE DISCRIMINATOR REGULARISATION ===
                self.disc_style_opt.zero_grad()
                z_real_style = self.sample_latent_prior_gaussian(x.size(0))
                d_real_loss_style = self.adv_loss(self.discriminator_style(z_real_style), torch.ones(x.size(0), device=self.device))
                _, z_fake_style = self.forward_encoder(x)
                d_fake_loss_style = self.adv_loss(self.discriminator_style(z_fake_style.detach()), torch.zeros(x.size(0), device=self.device))
                (d_real_loss_style + d_fake_loss_style).backward()
                self.disc_style_opt.step()

                # === GENERATOR REGULARISATION ===
                self.gen_cat_opt.zero_grad()
                d_pred_cat = self.discriminator_categorical(self.forward_encoder(x)[0])
                gen_cat_loss = self.adv_loss(d_pred_cat, torch.ones(x.size(0), device=self.device))
                gen_cat_loss.backward()
                self.gen_cat_opt.step()

                self.gen_style_opt.zero_grad()
                d_pred_style = self.discriminator_style(self.forward_encoder(x)[1])
                gen_style_loss = self.adv_loss(d_pred_style, torch.ones(x.size(0), device=self.device))
                gen_style_loss.backward()
                self.gen_style_opt.step()

                # === SEMI-SUPERVISED CLASSIFICATION ===
                self.semi_supervised_opt.zero_grad()
                # forward without softmax, since cross entropy already implements that
                y_hat, _ = self.forward_no_softmax(x)
                semi_supervised_loss = self.semi_supervised_loss(y_hat, y)
                semi_supervised_loss.backward()
                self.semi_supervised_opt.step()

                # accumulate
                total_recon_loss           += recon_loss.item()
                total_disc_cat_loss        += (d_real_loss_cat + d_fake_loss_cat).item()
                total_gen_cat_loss         += gen_cat_loss.item()
                total_disc_style_loss      += (d_real_loss_style + d_fake_loss_style).item()
                total_gen_style_loss       += gen_style_loss.item()
                total_semi_supervised_loss += semi_supervised_loss.item()

                # update tqdm display with current batch averages
                loop.set_postfix({
                    "Recon":           total_recon_loss / batch_idx,
                    "Disc_Cat":        total_disc_cat_loss / batch_idx,
                    "Gen_Cat":         total_gen_cat_loss / batch_idx,
                    "Disc_Style":      total_disc_style_loss / batch_idx,
                    "Gen_Style":       total_gen_style_loss / batch_idx,
                    "SemiSup":         total_semi_supervised_loss / batch_idx
                })

            # optionally print a summary at end of epoch
            print(
                f"Epoch {epoch+1}/{epochs} — "
                f"Recon: {total_recon_loss/len(train_loader):.4f}, "
                f"Disc_Cat: {total_disc_cat_loss/len(train_loader):.4f}, "
                f"Gen_Cat: {total_gen_cat_loss/len(train_loader):.4f}, "
                f"Disc_Style: {total_disc_style_loss/len(train_loader):.4f}, "
                f"Gen_Style: {total_gen_style_loss/len(train_loader):.4f}, "
                f"SemiSup: {total_semi_supervised_loss/len(train_loader):.4f}"
            )


            self.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for vx, vy in val_loader:
                    vx = vx.view(vx.size(0), -1).to(self.device)
                    vy = vy.to(self.device)
                    logits_cat, _ = self.forward_no_softmax(vx)
                    preds = logits_cat.argmax(dim=1)
                    correct += (preds == vy).sum().item()
                    total += vy.size(0)
            val_acc = correct / total * 100
            print(f"Validation Accuracy: {val_acc:.2f}%\n")

            self.log_epoch(
                total_recon_loss / len(train_loader),
                total_disc_cat_loss / len(train_loader),
                total_gen_cat_loss / len(train_loader),
                total_disc_style_loss / len(train_loader),
                total_gen_style_loss / len(train_loader),
                total_semi_supervised_loss / len(train_loader),
                val_acc
            )
            self.plot_training(update=True)



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

    def predict(self, x):
        """
        Predict class labels for input batch x.

        Args:
            x (torch.Tensor): shape (N, 1, 28, 28) or (N, 784)

        Returns:
            probs (torch.Tensor): shape (N, latent_dim_categorical), class probabilities
            preds (torch.Tensor): shape (N,), predicted class indices
        """
        self.eval()
        with torch.no_grad():
            if x.dim() == 4:
                x = x.view(x.size(0), -1)

            z = self.encoder(x.to(self.device))
            logits_cat = z[:, : self.options.latent_dim_categorical]
            probs = F.softmax(logits_cat, dim=1)
            preds = probs.argmax(dim=1)

        return probs, preds
    

    def generate_images(self, labels, style_z=None, prior_std=5.0):
        """
        Generate images conditioned on digit labels and style codes.

        Args:
            labels (int or torch.Tensor):
                - If int: a single digit in [0, latent_dim_categorical-1]
                - If 1D Tensor of ints: batch of digit labels
            style_z (torch.Tensor, optional):
                - If provided: Tensor of shape (batch_size, latent_dim_style)
                - If None: sampled from Gaussian prior with std=prior_std
            prior_std (float): std deviation for style prior sampling

        Returns:
            images (torch.Tensor): shape (batch_size, 1, 28, 28), values in [0,1]
        """
        self.eval()
        with torch.no_grad():
            
            if isinstance(labels, int):
                labels = torch.tensor([labels], device=self.device)
            else:
                labels = labels.to(self.device)
            batch_size = labels.size(0)

            
            z_cat = F.one_hot(labels, num_classes=self.options.latent_dim_categorical) \
                     .float().to(self.device)

            
            if style_z is None:
                style_z = self.sample_latent_prior_gaussian(batch_size, prior_std)
            else:
                style_z = style_z.to(self.device)
                assert style_z.shape == (batch_size, self.options.latent_dim_style), \
                       f"Expected style_z of shape {(batch_size, self.options.latent_dim_style)}"

            
            z = torch.cat([z_cat, style_z], dim=1)
            x_hat = self.decoder(z)  # shape (batch_size, input_dim)
            images = x_hat.view(batch_size, 1, 28, 28)

        return images
