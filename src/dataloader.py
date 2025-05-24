from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets


def load_mnist_data(batch_size, num_samples):
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert to [0, 1]
        transforms.Lambda(lambda x: x.view(-1)),  # Flatten the image
    ])

    # Load full MNIST training and test sets
    train_dataset = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="data", train=False, download=True, transform=transform)

    # Load all or part of data
    if num_samples == -1:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    else:
        indices = list(range(0, num_samples))
        train_subset = Subset(train_dataset, indices)
        test_subset = Subset(test_dataset, indices)

        # Create a DataLoader for the subset
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
