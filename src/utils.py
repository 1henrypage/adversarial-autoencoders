import torch
from torch import nn


import numpy as np

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


def compute_mean_std(data):
    """
    Compute the mean and std across the training set for each input dimension.

    Parameters:
        data (np.ndarray): Shape (num_samples, num_pixels)

    Returns:
        mean (np.ndarray): Shape (num_pixels,)
        std (np.ndarray): Shape (num_pixels,)
    """
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0) # + 1e-8  # Add epsilon to avoid division by zero
    return mean, std


def normalize_data(data, mean, std):
    """
    Normalize data using the mean and std.

    Parameters:
        data (np.ndarray): Shape (num_samples, num_pixels)
        mean (np.ndarray): Per-pixel mean
        std (np.ndarray): Per-pixel std

    Returns:
        normalized_data (np.ndarray): Normalized data
    """
    return (data - mean) / std


def rescale_to_unit_interval_individual(normalized_data, mean, std):
    """
    Invert normalization and rescale data to [0, 1] range.

    Parameters:
        normalized_data (np.ndarray): Normalized data
        mean (np.ndarray): Mean used in normalization
        std (np.ndarray): Std used in normalization

    Returns:
        rescaled_data (np.ndarray): Rescaled to [0, 1]
    """
    if hasattr(normalized_data, 'detach'):
        normalized_data=normalized_data.detach().cpu().numpy()
    if hasattr(mean, 'detach'):
        mean = mean.detach().cpu().numpy()
    if hasattr(std, 'detach'):
        std = std.detach().cpu().numpy()


    raw_data = normalized_data * std + mean
    min_val = np.min(raw_data, axis=1, keepdims=True)
    max_val = np.max(raw_data, axis=1, keepdims=True)
    return (raw_data - min_val) / (max_val - min_val)


def rescale_to_unit_interval_global(unnormalized_data: np.ndarray) -> np.ndarray:
    min_val = unnormalized_data.min()
    max_val = unnormalized_data.max()

    if max_val == min_val:
        # Avoid division by zero if data is constant
        return np.zeros_like(unnormalized_data)

    return (unnormalized_data - min_val) / (max_val - min_val)


