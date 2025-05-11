

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import KFold

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity

def log_likelihood_parzen(samples, data, sigma, batch_size=100):
    """
    Estimate the log-likelihood of `data` under a Parzen window estimator
    fitted to `samples`, using Gaussian kernels with bandwidth `sigma`.
    """

    if hasattr(samples, 'detach'):
        samples = samples.detach().cpu().numpy()
    if hasattr(data, 'detach'):
        data = data.detach().cpu().numpy()

    kde = KernelDensity(kernel='gaussian', bandwidth=sigma)
    kde.fit(samples)

    log_probs = []
    n = data.shape[0]
    for i in range(0, n, batch_size):
        batch = data[i:i+batch_size]
        log_p = kde.score_samples(batch)  # log-likelihoods
        log_probs.append(log_p)

    log_probs = np.concatenate(log_probs)
    return log_probs.mean()

def cross_validate_sigma(samples, validation_dataset, sigma_range, batch_size=100):
    """
    Cross-validate the kernel bandwidth (sigma) using the validation set.

    Args:
        samples (np.ndarray): Synthetic data from the model (num_samples x dim).
        validation_dataset (np.ndarray): Real validation data (num_valid x dim).
        sigma_range (list or np.ndarray): Candidate sigmas to evaluate.
        batch_size (int): Batch size for log-likelihood computation.

    Returns:
        float: Best sigma (highest log-likelihood).
    """
    best_ll = -np.inf
    best_sigma = None

    for sigma in sigma_range:
        print(f"Evaluating sigma = {sigma}")
        ll = log_likelihood_parzen(samples, validation_dataset, sigma, batch_size)
        print(f"Sigma: {sigma:.5f}, Log-Likelihood: {ll:.5f}")
        if ll > best_ll:
            best_ll = ll
            best_sigma = sigma

    print(f"Best sigma: {best_sigma}")

    return best_sigma


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


