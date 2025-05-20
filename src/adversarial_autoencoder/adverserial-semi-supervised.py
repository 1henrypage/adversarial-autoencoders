import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from adversarial import Encoder, Decoder, Discriminator, weights_init

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
            use_decoder_sigmoid=True,
            device="cuda"
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
        self.device = device

    def __repr__(self):
        return (
            f"<SemiSupervisedAutoEncoderOptions("  \
            f"input_dim={self.input_dim}, "  \
            f"encoder_hidden_dim={self.encoder_hidden_dim}, "  \
            f"decoder_hidden_dim={self.decoder_hidden_dim}, "  \
            f"latent_dim_categorical={self.latent_dim_categorical}, "  \
            f"latent_dim_style={self.latent_dim_style}, "  \
            f"use_decoder_sigmoid={self.use_decoder_sigmoid}, "  \
            f"device={self.device})>"
        )


class SemiSupervisedAdversarialAutoencoder(nn.Module):
    def __init__(self, options: SemiSupervisedAutoEncoderOptions):
        
        super(SemiSupervisedAdversarialAutoencoder, self).__init__()

        self.options = options

        self.device = options.device
        self.encoder = Encoder(options.input_dim, options.ae_hidden_dim, options.latent_dim_categorical + options.latent_dim_style).to(options.device)
        self.decoder = Decoder(options.latent_dim_categorical + options.latent_dim_style, options.ae_hidden_dim, options.input_dim, options.use_decoder_sigmoid).to(options.device)
        self.cat_softmax = nn.Softmax(dim=options.latent_dim_categorical)

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
        # self.gen_opt = torch.optim.SGD(
        #     self.encoder.parameters(),
        #     lr=options.init_gen_lr,
        #     momentum=0.1
        # )

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
            self.discriminator.parameters(),
            lr=options.init_disc_lr,
            momentum=0.1
        )

        # optimizer for style discriminator
        self.disc_style_opt = torch.optim.SGD(
            self.discriminator.parameters(),
            lr=options.init_disc_lr,
            momentum=0.1
        )

        self.recon_loss = options.recon_loss_fn
        self.semi_supervised_loss = options.semi_supervised_loss_fn
        self.adv_loss = nn.BCEWithLogitsLoss()
        # self.adv_loss_cat = nn.BCEWithLogitsLoss()
        # self.adv_loss_style = nn.BCEWithLogitsLoss()


    def foreward_reconstruction(self, x):
        z_cat, z_style = self.foreward_encoder(x)
        x_hat = self.decoder(torch.cat((z_cat, z_style), dim=1))
        return x_hat
    
    def foreward_encoder(self, x):
        z = self.encoder(x)
        z_cat = self.cat_softmax(z[:self.options.latent_dim_categorical])
        z_style = z[self.options.latent_dim_categorical:]
        return z_cat, z_style
        
    def sample_latent_prior_gaussian(self, n: int , prior_std: float = 5.0) -> torch.Tensor:
        return torch.randn(n, self.options.latent_dim_style).to(self.device) * prior_std
    
    def sample_latent_prior_categorical(self, n: int) -> torch.Tensor:
        latent_dim = self.options.latent_dim_categorical
        labels = torch.randint(0, latent_dim, (n,), device=self.device)
        return F.one_hot(labels, num_classes=latent_dim).float().to(self.device)


    def train(self, data_loader, epochs, prioir_std=5.0):
        for epoch in range(epochs):

            # adjust this if your experiment does different dynamic LRs
            if epoch == 50:
                self.recon_opt.param_groups[0]['lr'] = 0.001
                self.semi_supervised_opt.param_groups[0]['lr'] = 0.001
                self.gen_opt.param_groups[0]['lr'] = 0.01
                self.disc_style_opt.param_groups[0]['lr'] = 0.01
                self.disc_cat_opt.param_groups[0]['lr'] = 0.01
                
            elif epoch == 1000:
                self.recon_opt.param_groups[0]['lr'] = 0.0001
                self.semi_supervised_opt.param_groups[0]['lr'] = 0.0001
                self.gen_opt.param_groups[0]['lr'] = 0.001
                self.disc_style_opt.param_groups[0]['lr'] = 0.001
                self.disc_cat_opt.param_groups[0]['lr'] = 0.001


            total_recon_loss = 0
            total_semi_supervised_loss = 0

            total_disc_style_loss = 0
            total_gen_style_loss = 0

            total_disc_cat_loss = 0
            total_gen_cat_loss = 0

            for batch_idx, (x, y) in enumerate(data_loader):
                x = x.to(self.device)
                y = y.to(self.device)

                # === RECONSTRUCTION PHASE ====
                self.recon_opt.zero_grad()
                x_hat = self.foreward_reconstruction(x)
                recon_loss = self.recon_loss(x_hat, x)
                recon_loss.backward()
                self.recon_opt.step()


                # === CATEGORICAL DISCRIMINATOR REGULARISATION ===
                self.disc_cat_opt.zero_grad()

                z_real_cat = self.sample_latent_prior_categorical(x.size(0))
                d_real_cat = self.discriminator_categorical(z_real_cat)
                d_real_loss_cat = self.adv_loss(d_real_cat, torch.ones_like(d_real_cat))

                z_fake_cat, _ = self.foreward_encoder(x).detach()
                d_fake_cat = self.discriminator_categorical(z_fake_cat)
                d_fake_loss_cat = self.adv_loss(d_fake_cat, torch.zeros_like(d_fake_cat))

                disc_loss_cat = d_real_loss_cat + d_fake_loss_cat
                disc_loss_cat.backward()
                self.disc_cat_opt.step()


                # === STYLE DISCRIMINATOR REGULARISATION ===
                self.disc_style_opt.zero_grad()

                z_real_style = self.sample_latent_prior_gaussian(x.size(0))
                d_real_style = self.discriminator_style(z_real_style)
                d_real_loss_style = self.adv_loss(d_real_style, torch.ones_like(d_real_style))

                _, z_fake_style = self.foreward_encoder(x).detach()
                d_fake_style = self.discriminator_style(z_fake_style)
                d_fake_loss_style = self.adv_loss(d_fake_style, torch.zeros_like(d_fake_style))

                disc_loss_style = d_real_loss_style + d_fake_loss_style
                disc_loss_style.backward()
                self.disc_style_opt.step()

                #  === GENERATOR REGULARISATION ===
                # doing it kinda jointly, idk if that's right
                self.gen_cat_opt.zero_grad()
                z_cat, z_style = self.foreward_encoder(x)

                d_pred_cat = self.discriminator_categorical(z_cat)
                gen_cat_loss = self.adv_loss(d_pred_cat, torch.ones_like(d_pred_cat))

                gen_cat_loss.backward()
                self.gen_cat_opt.step()

                d_pred_style = self.discriminator_style(z_style)
                gen_style_loss = self.adv_loss(d_pred_style, torch.ones_like(d_pred_style))

                gen_style_loss.backward()
                self.gen_style_opt.step()

                #  === SEMI-SUPERVISED CLASSIFICATION ===

                self.semi_supervised_opt.zero_grad()
                y_hat, _ = self.foreward_encoder(x)

                semi_supervised_loss = self.semi_supervised_loss(y_hat, y)
                semi_supervised_loss.backward()
                self.semi_supervised_opt.step()

                # add up lossess

                total_recon_loss += recon_loss.item()

                total_disc_cat_loss += disc_loss_cat.item()
                total_gen_cat_loss += gen_cat_loss.item()

                total_disc_style_loss += disc_loss_style.item()
                total_gen_style_loss += gen_style_loss.item()

                total_semi_supervised_loss += semi_supervised_loss.item()


                print(f"Epoch ({epoch + 1}/{epochs})\t)")

                print(f"Recon Loss: {total_recon_loss / len(data_loader):.4f}\t)")

                print(f"Disc Cat Loss: {total_disc_cat_loss / len(data_loader):.4f}\t)")
                print(f"Gen Cat Loss: {total_gen_cat_loss / len(data_loader):.4f}\t)")

                print(f"Disc Style Loss: {total_disc_style_loss / len(data_loader):.4f}\t)")
                print(f"Gen Style Loss: {total_gen_style_loss / len(data_loader):.4f}\t)")
            
                print(f"Semi-supervised Loss: {total_semi_supervised_loss / len(data_loader):.4f}\t)")





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
