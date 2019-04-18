import numpy as np
import numpy.linalg as la

import pytest
from collections import namedtuple

import hw1.data as hw1data


def test_generate_dataset():
    TestCase = namedtuple('TestCase', ['n', 'd', 'smax', 'smin'])
    test_cases = [
        TestCase(n=1024, d=64, smax=25, smin=5),
        TestCase(n=1024, d=64, smax=5, smin=1),
        TestCase(n=128,  d=16, smax=17, smin=0.1),
        TestCase(n=4096, d=512, smax=100, smin=99),
    ]
    for testcase in test_cases:
        n, d, smax, smin = testcase
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
        assert la.norm(xs_sol - xs) < 1e-1
