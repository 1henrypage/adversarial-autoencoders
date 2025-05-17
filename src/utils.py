import torch
from torch import nn


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def save_weights(encoder, decoder, discriminator, path_prefix="aae_weights"):
    """
    Saves the weights of the encoder, decoder, and discriminator.

    Args:
        path_prefix (str): Prefix for the saved file paths. Files will be saved as:
            - <path_prefix>_encoder.pth
            - <path_prefix>_decoder.pth
            - <path_prefix>_discriminator.pth
    """
    torch.save(encoder.state_dict(), f"{path_prefix}_encoder.pth")
    torch.save(decoder.state_dict(), f"{path_prefix}_decoder.pth")
    torch.save(discriminator.state_dict(), f"{path_prefix}_discriminator.pth")
    print(f"Weights saved to {path_prefix}_*.pth")


def load_weights(encoder, decoder, discriminator, device, path_prefix="aae_weights"):
    """
    Loads the weights of the encoder, decoder, and discriminator from files.

    Args:
        path_prefix (str): Prefix for the saved file paths. Expected files are:
            - <path_prefix>_encoder.pth
            - <path_prefix>_decoder.pth
            - <path_prefix>_discriminator.pth
    """
    encoder.load_state_dict(torch.load(f"{path_prefix}_encoder.pth", map_location=device))
    decoder.load_state_dict(torch.load(f"{path_prefix}_decoder.pth", map_location=device))
    discriminator.load_state_dict(torch.load(f"{path_prefix}_discriminator.pth", map_location=device))
    print(f"Weights loaded from {path_prefix}_*.pth")
