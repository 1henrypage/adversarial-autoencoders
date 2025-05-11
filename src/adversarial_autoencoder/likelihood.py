

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import KFold

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity

def cross_validate_sigma(samples, sigma_range, n_folds=5):
    """
    Cross-validates sigma (kernel bandwidth) using a validation dataset.
    """
    # Convert from torch to numpy if needed
    if hasattr(samples, 'detach'):
        samples = samples.detach().cpu().numpy()

    grid = GridSearchCV(
        KernelDensity(kernel='gaussian'),
        {'bandwidth': sigma_range},
        cv=n_folds,
        verbose=3
    )
    grid.fit(samples)
    return grid.best_params_['bandwidth']

def estimate_log_likelihood(samples, test_data, sigma):
    """
    Estimates the average log-likelihood of the test set under the Parzen window estimator.
    """
    # Convert from torch to numpy if needed
    if hasattr(samples, 'detach'):
        samples = samples.detach().cpu().numpy()
    if hasattr(test_data, 'detach'):
        test_data = test_data.detach().cpu().numpy()

    kde = KernelDensity(kernel='gaussian', bandwidth=sigma)
    kde.fit(samples)
    log_probs = kde.score_samples(test_data)
    return np.mean(log_probs), np.std(log_probs) / np.sqrt(len(test_data))


# Example usage:
# G_samples: samples generated from generative model G (n_samples x n_features)
# X_val: validation set (for cross-validating sigma)
# X_test: test set (to compute log-likelihood)

# Assume these arrays are available (replace with actual data)
# G_samples = np.array(...)  # generated samples
# X_val = np.array(...)      # validation data
# X_test = np.array(...)     # test data

# Parameters
# sigma_range = np.logspace(-1, 1, 20)  # Example range from 0.1 to 10

# sigma = cross_validate_sigma(G_samples, X_val, sigma_range)
# log_likelihood_mean, log_likelihood_se = estimate_log_likelihood(G_samples, X_test, sigma)

# print(f"Log-likelihood estimate: {log_likelihood_mean:.2f} ± {log_likelihood_se:.2f}")
