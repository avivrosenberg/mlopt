import abc
import math
from functools import partial

import numpy as np
from sklearn.base import BaseEstimator, DensityMixin
from sklearn.utils.validation import check_is_fitted

import optim.optimizers as optimizers
import optim.projections as projections

import optim.stepsize_gen as stepsize_gen


class OnlineRebalancingPortfolio(BaseEstimator, DensityMixin, abc.ABC):
    """
    Fits a rebalancing portfolio to asset-returns data.
    This solved the online convex optimization problem, using a loss
    function at time t of f_t(x) = -log(np.dot(r_t, x)) where r_t is the
    returns vector for assets at time t.

    Attributes:
        p_: The fitted portfolio (a distribution over assets).
        P_: Iterates of the portfolio.
        loss_fns_: Loss functions produced each iterate.
    """

    def __init__(self):
        pass

    @staticmethod
    def loss_fn(r: np.ndarray, x: np.ndarray):
        return -np.log(np.dot(x, r))

    @staticmethod
    def grad_fn(r: np.ndarray, x: np.ndarray):
        return - r * (1. / np.dot(x, r))

    @abc.abstractmethod
    def fit(self, R: np.ndarray, save_iterates=True, save_loss_fns=True, **kw):
        """
        Fits a rebalancing constant portfolio, which is a point within the
        unit simplex to the stock price data in X.
        :param R: Array of shape (T,d) where T is the number of time points
        (e.g. days) and d is the number of different assets. Each entry is
        the asset price-ratio between the current and previous time.
        :param save_iterates: Whether to save each iterate.
        :param save_loss_fns: Whether to save each round's loss function
        (as a callable accepting a single parameter).
        :return: self
        """
        pass

    def wealth(self, R: np.ndarray, P: np.ndarray = None):
        """
        Calculate the wealth of the algorithm at each trading round.
        :param R: Array of shape (T,d) where T is the number of time points
        (e.g. days) and d is the number of different assets. Each entry is
        the asset price-ratio between the current and previous time.
        :param P: Array of shape (T,d) or (d,) representing the Portfolio at
        each round or a fixed portfolio.
        None (default) specifies that the iterates from fit() will be
        used (Requires that the iterates were save when fitting).
        :return: Array of shape (T,), containing wealth of the algorithm at
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

    def regret(self, p_fixed: np.ndarray, average=False):
        """
        Calculates the regret of the fitted model as a function of
        the number of rounds, compared to a fixed portfolio.
        This requires that the model was fitted with save_loss_fns and
        save_iterates both true.

        :param p_fixed: Portfolio (distribution of assets) that will be used to
        as the "best in hindsight" (for all rounds).
        :return: Array of shape (T,), containing regret of the algorithm at
        each trading round compared to the fixed portfolio.
        """
        check_is_fitted(self, ['p_', 'P_', 'loss_fns_'])
        assert p_fixed.ndim == 1 and p_fixed.shape[0] == self.P_.shape[1]

        T = len(self.loss_fns_)
        regret = np.zeros((T,), dtype=np.float32)

        for t, loss_fn in enumerate(self.loss_fns_):
            pt = self.P_[t]
            ft = loss_fn(pt)
            regret[t] = regret[t - 1] + ft - loss_fn(p_fixed)

        if average:
            regret /= np.arange(start=1, stop=T + 1)

        return regret


class OGDOnlineRebalancingPortfolio(OnlineRebalancingPortfolio):
    """
    Implements online gradient descent (OGD) for the ORPS problem.
    """

    def __init__(self):
        super().__init__()

    def fit(self, R: np.ndarray, save_iterates=True, save_loss_fns=True, **kw):
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
        # G = np.max(np.linalg.norm(R, axis=1))
        G = np.max(np.linalg.norm(R, axis=1) / np.sum(R, axis=1))
        eta = D / (G * math.sqrt(T))
        print(f'eta(OGD)={eta}')

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
                loss_fns.append(partial(self.loss_fn,
                                        np.array(rt, copy=False)))

        # save fit results
        self.p_ = pt
        self.P_ = P
        self.loss_fns_ = loss_fns
        return self


class RFTLOnlineRebalancingPortfolio(OnlineRebalancingPortfolio):
    """
    Implements regularized follow-the-leader (RFTL) for the ORPS problem.
    """

    def __init__(self):
        super().__init__()

    def fit(self, R: np.ndarray, save_iterates=True, save_loss_fns=True, **kw):
        # R is the reward (price ratio of day t+1 to day t) per asset
        assert R.ndim == 2
        T, d = R.shape
        R = R.astype(np.float32)

        # pt is the current iterate
        pt = np.full((d,), 1. / d, dtype=np.float32)

        # Save portfolios
        P = np.zeros_like(R) if save_iterates else None

        # Save loss functions
        loss_fns = [] if save_loss_fns else None

        # Hyperparameters
        D = math.sqrt(math.log(d))
        G = np.max(np.max(np.abs(R), axis=1) / np.sum(R, axis=1))
        eta = D / (G * math.sqrt(2 * T))
        print(f'eta(RFTL)={eta}')

        for t in range(T):
            rt = R[t]

            if save_iterates:
                P[t] = pt

            if save_loss_fns:
                loss_fns.append(partial(self.loss_fn,
                                        np.array(rt, copy=False)))

            gt = self.grad_fn(rt, pt)

            pt_exp_gt = pt * np.exp(-eta * gt)
            pt = pt_exp_gt / np.sum(pt_exp_gt)

        # save fit results
        self.p_ = pt
        self.P_ = P
        self.loss_fns_ = loss_fns
        return self

