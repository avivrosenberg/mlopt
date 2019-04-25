import numpy as np


def generate_linear_regression(n, d, smax=10, smin=1, sol_mu=100, sol_std=10,
                               noise_std=0.01, **kw):
    """
    Generates a dataset for a linear regression optimization task:

    .. math::
        \min_{x\in\mathbb{R}} ||Ax - b||^2

    A and b.
    A will be an (n,d) matrix with singular values between smax to smin.
    b will be chosen as b = A*xs+n, where xs is some random point (the
    optimal solution) and n is uniform Gaussian noise with a given std.

    :param n: Number of rows in A
    :param d: Number of columns in A (also features in x)
    :param smax: Largest singular value
    :param smin: Smallest singular value
    :param sol_mu: Mean of distribution to sample xs from.
    :param sol_std: Stdev. of distribution to sample xs from.
    :param noise_std: Stdev. of Gaussian noise added to generate b.
    :return: A tuple containing A (n,d), b (n,), and xs (d,).
    """

    # Decompose a random matrix T with SVD
    T = np.random.randn(n, d)
    U, s, Vh = np.linalg.svd(T)

    # New singular values
    s = np.flipud(np.linspace(smin, smax, num=len(s)))

    # Create samples matrix A based on SVD
    S = np.zeros_like(T)
    S[:len(s), :len(s)] = np.diag(s)
    A = U @ S @ Vh

    # Sample a random solution xs
    xs = np.random.multivariate_normal(
        np.zeros(d) + sol_mu, np.eye(d) * sol_std
    )

    # Create targets vector
    b = A.dot(xs) + np.random.randn(n) * noise_std

    return A, b, xs
