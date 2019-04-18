import numpy as np
import numpy.linalg as la

import pytest

import hw1.data as hw1data


def test_generate_dataset():
    n, d, smax, smin = 1024, 64, 25, 5
    noise_std = 0.001

    A, b, xs = hw1data.generate_dataset(n_samples=n, n_features=d,
                                        smax=smax, smin=smin,
                                        noise_std=noise_std)
    # Test dimensions
    assert A.shape == (n, d)
    assert b.shape == (n,)
    assert xs.shape == (d,)

    # Solve least squares with A, b
    xs_sol, err, rank, svals = la.lstsq(A, b.reshape(-1))

    # Test singular values
    assert svals[0] == pytest.approx(smax)
    assert svals[-1] == pytest.approx(smin)
    assert rank == d

    # Test solution
    assert la.norm(xs_sol - xs) < 1e-3
