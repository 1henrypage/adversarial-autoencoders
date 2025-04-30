import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np


def get_data_loaders(dataset, data_root, batch_size, labeled_fraction=1.0):
    """
    Creates DataLoaders for a dataset with optional semi-supervised split.

    :param dataset: The dataset to use.
    :param data_root: Directory where MNIST data is stored
    :param batch_size: Batch size for the DataLoader
    :param labeled_fraction: Fraction of the data used for supervised learning
    :param supervised: If True, uses labeled data for supervised learning
    :return: train_loader, test_loader
    """
    dataset_class, mean, std, train_dataset_no_norm = None, None, None, None

    # Dataset-specific settings
    # Load dataset without normalization first to compute mean and std
    if dataset == "MNIST":
        dataset_class = datasets.MNIST
        train_dataset_no_norm = dataset_class(root=data_root, train=True, download=True,
                                              transform=transforms.ToTensor())
    elif dataset == "SVHN":
        dataset_class = datasets.SVHN
        train_dataset_no_norm = dataset_class(root=data_root, split="train", download=True, transform=transforms.ToTensor())

    # Compute mean and std for normalization
    train_data = torch.stack([img[0] for img in train_dataset_no_norm])  # Extract image data
    mean = train_data.mean(dim=[0, 2, 3])  # Mean per channel (R, G, B)
    std = train_data.std(dim=[0, 2, 3])  # Standard deviation per channel (R, G, B)

    # Apply normalization
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    if dataset == "SVHN":
        train_dataset = dataset_class(root=data_root, split="train", download=True, transform=transform)
        test_dataset = dataset_class(root=data_root, split='test', download=True, transform=transform)
    else:
        train_dataset = dataset_class(root=data_root, train=True, download=True, transform=transform)
        test_dataset = dataset_class(root=data_root, train=False, download=True, transform=transform)


    # Handle labeled/unlabeled split (for semi-supervised setup)
    if labeled_fraction < 1.0:
        num_labeled = int(len(train_dataset) * labeled_fraction)
        indices = np.random.permutation(len(train_dataset))
        labeled_indices = indices[:num_labeled]

        labeled_subset = Subset(train_dataset, labeled_indices)
        unlabeled_indices = indices[num_labeled:]
        unlabeled_subset = Subset(train_dataset, unlabeled_indices)

        train_loader_labeled = DataLoader(labeled_subset, batch_size=batch_size, shuffle=True)
        train_loader_unlabeled = DataLoader(unlabeled_subset, batch_size=batch_size, shuffle=True)
    else:
        train_loader_labeled = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        train_loader_unlabeled = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Return both labeled and unlabeled loaders
    return train_loader_labeled, train_loader_unlabeled, test_loader
