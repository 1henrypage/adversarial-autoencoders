import os
import torch
import torch.nn as nn
import torchvision
from torch.utils.tensorboard import SummaryWriter


class LinearMaxOut(nn.Module):
    def __init__(self, input_dim, output_dim, num_pieces):
        super().__init__()
        self.output_dim = output_dim
        self.num_pieces = num_pieces
        self.linear = nn.Linear(input_dim, output_dim * num_pieces)

    def forward(self, x):
        out = self.linear(x)
        out = out.view(out.size(0), self.num_pieces, self.output_dim)
        return torch.amax(out, dim=1)


def gan_weights_init_gen(m):
    if isinstance(m, nn.Linear):
        nn.init.uniform_(m.weight, -0.05, 0.05)
        nn.init.zeros_(m.bias)

def gan_weights_init_disc(m):
    if isinstance(m, nn.Linear):
        nn.init.uniform_(m.weight, -0.005, 0.005)
        nn.init.zeros_(m.bias)


class GAN(nn.Module):
    def __init__(self,
                 latent_dim: int,
                 input_size: int,
                 generator_hidden_dim: int,
                 discriminator_hidden_dim: int,
                 use_sigmoid_gen: bool,
                 device: str):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_size = input_size
        self.generator_hidden_dim = generator_hidden_dim
        self.discriminator_hidden_dim = discriminator_hidden_dim
        self.device = device

        gen_layers = [
            nn.Linear(latent_dim, generator_hidden_dim),
            nn.ReLU(),
            nn.Linear(generator_hidden_dim, generator_hidden_dim),
            nn.ReLU(),
            nn.Linear(generator_hidden_dim, input_size)
        ]
        if use_sigmoid_gen:
            gen_layers.append(nn.Sigmoid())

        self.generator = nn.Sequential(*gen_layers)

        self.discriminator = nn.Sequential(
            nn.Dropout(0.2),
            LinearMaxOut(input_size, discriminator_hidden_dim, 5),
            nn.Dropout(0.5),
            LinearMaxOut(discriminator_hidden_dim, discriminator_hidden_dim, 5),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(discriminator_hidden_dim, 1),
            nn.Sigmoid()
        )

        self.generator.apply(gan_weights_init_gen)
        self.discriminator.apply(gan_weights_init_disc)

        self.generator.to(self.device)
        self.discriminator.to(self.device)

        self.adv_loss = nn.BCELoss()

    def sample_z(self, batch_size, uniform_range):
        return torch.FloatTensor(batch_size, self.latent_dim).uniform_(
            -uniform_range, uniform_range
        ).to(self.device)

    def adjust_lr(self, gen_opt, disc_opt, min_lr, decay_factor):
        for optim in [gen_opt, disc_opt]:
            lr = optim.param_groups[0]['lr']
            if lr > min_lr:
                new_lr = max(lr * decay_factor, min_lr)
                optim.param_groups[0]['lr'] = new_lr

    def adjust_momentum(self, gen_opt, disc_opt, epoch, max_epoch, momentum, final_momentum):
        alpha = min(max(epoch / max_epoch, 0.0), 1.0)
        new_momentum = momentum * (1 - alpha) + final_momentum * alpha
        for optim in [gen_opt, disc_opt]:
            for group in optim.param_groups:
                group['momentum'] = new_momentum

    def train_mbgd(self, data_loader,
                   learning_rate: float,
                   uniform_range: float,
                   min_lr: float,
                   decay_factor: float,
                   epochs: int,
                   momentum: float = 0.5,
                   final_momentum: float = 0.9,
                   momentum_saturate: int = 100,
                   log_dir="./tmp_runs"
                   ):

        writer = SummaryWriter(log_dir=log_dir)

        gen_opt = torch.optim.SGD(self.generator.parameters(), lr=learning_rate, momentum=momentum)
        disc_opt = torch.optim.SGD(self.discriminator.parameters(), lr=learning_rate, momentum=momentum)

        for epoch in range(epochs):
            gen_loss_sum = 0
            disc_loss_sum = 0

            for batch_idx, (x_real, _) in enumerate(data_loader):
                x_real = x_real.to(self.device)
                batch_size = x_real.size(0)

                # === Discriminator update ===
                disc_opt.zero_grad()
                z = self.sample_z(batch_size, uniform_range)
                x_fake = self.generator(z).detach()

                real_out = self.discriminator(x_real)
                fake_out = self.discriminator(x_fake)

                d_loss_real = self.adv_loss(real_out, torch.ones_like(real_out))
                d_loss_fake = self.adv_loss(fake_out, torch.zeros_like(fake_out))
                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                disc_opt.step()

                # === Generator update ===
                gen_opt.zero_grad()
                z = self.sample_z(batch_size, uniform_range)
                x_fake = self.generator(z)
                fake_out = self.discriminator(x_fake)

                g_loss = self.adv_loss(fake_out, torch.ones_like(fake_out))
                g_loss.backward()
                gen_opt.step()

                gen_loss_sum += g_loss.item()
                disc_loss_sum += d_loss.item()

                if batch_idx % 100 == 0:
                    print(f"[Epoch {epoch+1}/{epochs}] Batch {batch_idx} | "
                          f"G Loss: {g_loss.item():.4f} | D Loss: {d_loss.item():.4f} | "
                          f"LR: {disc_opt.param_groups[0]['lr']:.6f} | "
                          f"Momentum: {disc_opt.param_groups[0]['momentum']:.4f}")

                self.adjust_lr(gen_opt, disc_opt, min_lr, decay_factor)

            self.adjust_momentum(gen_opt, disc_opt, epoch, momentum_saturate, momentum, final_momentum)

            avg_g = gen_loss_sum / len(data_loader)
            avg_d = disc_loss_sum / len(data_loader)

            writer.add_scalar("Loss/Generator", avg_g, epoch)
            writer.add_scalar("Loss/Discriminator", avg_d, epoch)

        writer.close()



    def save_weights(self, path_prefix="gan_weights"):
        torch.save(self.generator.state_dict(), f"{path_prefix}_generator.pth")
        torch.save(self.discriminator.state_dict(), f"{path_prefix}_discriminator.pth")
        print(f"Weights saved to {path_prefix}_*.pth")

    def load_weights(self, device, path_prefix="gan_weights"):
        self.generator.load_state_dict(torch.load(f"{path_prefix}_generator.pth", map_location=device))
        self.discriminator.load_state_dict(torch.load(f"{path_prefix}_discriminator.pth", map_location=device))
        print(f"Weights loaded from {path_prefix}_*.pth")


