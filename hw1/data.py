import numpy as np


def generate_dataset(n_samples, n_features, smax=10, smin=1,
                     sample_std=10, noise_std=0.01):
    """
    Generates a dataset for a linear regression optimization task:

    .. math::
        \min_{x\in\mathbb{R}} ||Ax - b||^2

    A and b.
    A will be an (m,n) matrix with singular values between sigma_max to
    sigma_min.
    b will be chosen as b = A*xs+n, where xs is some random point (the
    optimal solution) and n is uniform Gaussian noise with a given std.

    :param n_samples: Number of rows in A (n)
    :param n_features: Number of columns in A (also features in x, aka d)
    :param smax: Largest singular value
    :param smin: Smallest singular value
    :param sample_std: Stdev. of distribution to sample xs from.
    :param noise_std: Stdev. of Gaussian noise added to generate b.
    :return: A tuple containing A (n,d), b (n,), and xs (d,).
    """

    n, d = n_samples, n_features

    # Decompose a random matrix T with SVD
    T = np.random.randn(n, d)
    U, s, Vh = np.linalg.svd(T)

    # New singular values
    s = np.flipud(np.linspace(smin, smax, num=len(s)))

    # Create matrix A based on SVD
    S = np.zeros_like(T)
    S[:len(s), :len(s)] = np.diag(s)
    A = U @ S @ Vh

    # Sample a random solution xs with (large variance)
    xs = np.random.randn(d, 1) * 10

    # Create bias vector
    b = A @ xs + np.random.randn(n, 1) * noise_std

    return A, b.reshape(-1), xs.reshape(-1)
