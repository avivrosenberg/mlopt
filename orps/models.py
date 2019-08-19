import math
from functools import partial

import numpy as np
from sklearn.utils import check_array
from sklearn.base import BaseEstimator, DensityMixin
from sklearn.utils.validation import check_is_fitted

import optim.optimizers as optimizers
import optim.projections as projections

import optim.stepsize_gen as stepsize_gen


class OnlineRebalancingPortfolio(BaseEstimator, DensityMixin):
    def __init__(self):
        pass

    @staticmethod
    def loss_fn(r: np.ndarray, x: np.ndarray):
        return -np.log(np.dot(x, r))

    @staticmethod
    def grad_fn(r: np.ndarray, x: np.ndarray):
        return - r * (1. / np.dot(x, r))

    def fit(self, R: np.ndarray, save_iterates=True, save_loss_fns=True, **kw):
        """
        Fits a rebalancing constant portfolio, which is a point within the
        unit simplex to the stock price data in X.
        :param R: Array of shape (n,d) where n is the number of time points
        (e.g. days) and d is the number of different assets. Each entry is
        the asset price-ratio between the current and previous time.
        :param save_iterates: Whether to save each iterate.
        :param save_loss_fns: Whether to save each round's loss function
        (as a callable accepting a single parameter).
        :return: self
        """
        # R is the reward (price ratio of day t+1 to day t) per asset
        assert R.ndim == 2
        T, d = R.shape
        R = R.astype(np.float32)

        # rt is the current return vector
        rt = R[0]

        # pt is the current iterate
        pt = np.full((d,), 1. / d, dtype=np.float32)

        # Save portfolios
        P = np.zeros_like(R) if save_iterates else None

        # Save loss functions
        loss_fns = [] if save_loss_fns else None

        # Hyperparameters
        D = math.sqrt(2)
        G = np.max(np.linalg.norm(R, axis=1))
        eta = D / (G * math.sqrt(T))

        opt = optimizers.GradientDescent(
            x0=pt, max_iter=T, yield_x0=True,
            stepsize_gen=stepsize_gen.const(eta),
            grad_fn=lambda x: self.grad_fn(rt, x),
            project_fn=projections.SimplexProjection(),
        )

        # Iterate over optimizer
        for t, pt in enumerate(opt, start=0):
            rt = R[t]

            if save_iterates:
                P[t] = pt

            if save_loss_fns:
                loss_fns.append(partial(self.loss_fn, np.copy(rt)))

        # save fit results
        self.p_ = pt
        self.P_ = P
        self.loss_fns_ = loss_fns
        return self

    def wealth(self, R: np.ndarray, P: np.ndarray = None):
        """
        Calculate the wealth of the algorithm at each trading round.
        :param R: Array of shape (n,d) where n is the number of time points
        (e.g. days) and d is the number of different assets. Each entry is
        the asset price-ratio between the current and previous time.
        :param P: Array of shape (n,d) or (d,) representing the Portfolio at
        each round or a fixed portfolio.
        None (default) specifies that the iterates from fit() will be
        used (Requires that the iterates were save when fitting).
        :return: Array of shape (n,), containing wealth of the algorithm at
        each trading round.
        """
        assert R.ndim == 2
        if P is None:
            check_is_fitted(self, ['P_'])
            P = self.P_
            assert R.shape == P.shape
        else:
            assert (P.ndim == 2 and R.shape == P.shape) or \
                   (P.ndim == 1 and P.shape[0] == R.shape[1])

        return np.cumprod(np.sum(R * P, axis=1), axis=0)

    def score(self, X, y=None):
        """
        TODO: Implement regret
        :param X: Asset price data
        :param y: Portfolio (distribution of assets). If None, the fitted
        portfolio will be used.
        :return:
        """
        check_is_fitted(self, ['p_', 'losses_', 'wealth_'])

        X = check_array(X, ensure_2d=True, ensure_min_features=2,
                        ensure_min_samples=2, dtype=np.float32)
        if y is None:
            y = self.p_

        R = (X[1:, :] / X[:-1, :])

        pass

    def predict(self, X):
        # Input validation
        check_is_fitted(self, ['p_', 'losses_', 'wealth_'])
        X = check_array(X)

        # TODO
        return None
