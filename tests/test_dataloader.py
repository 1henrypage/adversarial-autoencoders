import unittest

import numpy as np

from src.dataloader.dataloader import get_data_loaders


class TestGetDataLoaders(unittest.TestCase):

    def test_mnist_loader(self):
        # Set up
        data_root = '../data'  # path where MNIST data is stored
        batch_size = 100
        labeled_fraction = 0.02

        # Load data
        train_loader_labeled, train_loader_unlabeled, test_loader = get_data_loaders("MNIST", data_root, batch_size, labeled_fraction)

        # Check that the data loaders are not empty
        self.assertGreater(len(train_loader_labeled), 0, "Training loader is empty")
        self.assertGreater(len(test_loader), 0, "Test loader is empty")

        # Check that the number of samples in train_loader matches the labeled fraction
        expected_train_size = int(60000 * labeled_fraction)  # MNIST has 60,000 training samples
        actual_train_size = len(train_loader_labeled.dataset)
        self.assertEqual(actual_train_size, expected_train_size, f"Expected {expected_train_size} training samples, but got {actual_train_size}")

        # Check if the batch size is correct
        for batch in train_loader_labeled:
            self.assertEqual(batch[0].shape[0], batch_size, "Batch size mismatch")

        # Verify that the normalization of MNIST is applied correctly (mean, std)
        subset = train_loader_labeled.dataset
        mean = subset.dataset.data[subset.indices].float().mean() / 255.0  # MNIST is in [0, 255], we divide by 255 for normalization
        std = subset.dataset.data[subset.indices].float().std() / 255.0
        self.assertAlmostEqual(mean, 0.1307, delta=0.01)
        self.assertAlmostEqual(std, 0.3081, delta=0.01)


if __name__ == "__main__":
    unittest.main()