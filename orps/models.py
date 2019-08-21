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

    Attributes (after fitting model):
        p_: The fitted portfolio (a distribution over assets).
        P_: Iterates of the portfolio.
        loss_fns_: Loss functions produced each iterate.
        eta_: Learning rate used
    """

    def __init__(self):
        self.__rt__ = None

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
        :param R: Array of shape (T,d) where T is the number of time points
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

        # Iterate over optimizer
        self.__rt__ = R[0]  # To allow _pt_generator access to rt
        for t, pt in enumerate(self._pt_generator(R)):
            self.__rt__ = R[t]

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

    @abc.abstractmethod
    def _pt_generator(self, R: np.ndarray):
        """
        Generator returning pt, the portfolio at iteration t.
        :param R: The asset returns data matrix of shape (T, d).
        :return: A generator over iterates.
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
        :param average: Whether to calculate the average regret (divide by
        t, the number of the round) or not.
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

    def _pt_generator(self, R: np.ndarray):
        T, d = R.shape

        # Hyperparameters
        D = math.sqrt(2)
        G = np.max(np.linalg.norm(R, axis=1) / np.sum(R, axis=1))
        self.eta_ = D / (G * math.sqrt(T))

        p0 = np.full((d,), 1. / d, dtype=np.float32)

        opt = optimizers.GradientDescent(
            x0=p0, max_iter=T, yield_x0=True,
            stepsize_gen=stepsize_gen.const(self.eta_),
            grad_fn=lambda x: self.grad_fn(self.__rt__, x),
            project_fn=projections.SimplexProjection(),
        )

        return opt


class RFTLOnlineRebalancingPortfolio(OnlineRebalancingPortfolio):
    """
    Implements regularized follow-the-leader (RFTL) for the ORPS problem.
    """

    def __init__(self):
        super().__init__()

    def _pt_generator(self, R: np.ndarray):
        T, d = R.shape

        # Hyperparameters
        D = math.sqrt(math.log(d))
        G = np.max(np.max(np.abs(R), axis=1) / np.sum(R, axis=1))
        self.eta_ = D / (G * math.sqrt(2 * T))

        pt = np.full((d,), 1. / d, dtype=np.float32)

        for t in range(T):
            rt = R[t]
            gt = self.grad_fn(rt, pt)
            pt_exp_gt = pt * np.exp(-self.eta_ * gt)
            pt = pt_exp_gt / np.sum(pt_exp_gt)

            yield pt


class NewtonStepOnlineRebalancingPortfolio(OnlineRebalancingPortfolio):
    def __init__(self):
        super().__init__()

    def _pt_generator(self, R: np.ndarray):
        T, d = R.shape

        # Hyperparams
        D = math.sqrt(2)
        G = np.max(np.linalg.norm(R, axis=1) / np.sum(R, axis=1))
        alpha = math.inf
        gamma = 0.5 * min(1 / 4 / G / D, alpha)
        eps = 1 / (gamma ** 2) / (D ** 2)

        A = eps * np.eye(d)
        Ainv = 1 / eps * np.eye(d)

        pt = np.full((d,), 1. / d, dtype=np.float32)
        for t in range(T):
            rt = R[t]
            gt = self.grad_fn(rt, pt)

            # Update the gradient
            gtgt = np.outer(gt, gt)
            A = A + gtgt
            Ainv = Ainv - np.dot(Ainv, np.dot(gtgt, Ainv)) / \
                   (1 + np.dot(gt, np.dot(Ainv, gt)))

            yt = pt - (1 / gamma) * np.dot(Ainv, gt)

            proj = projections.MetricInducedSimplexProjection(A, eta_min=0.05)
            pt = proj(yt)

            yield pt


class BestFixedRebalancingPortfolio(BaseEstimator, DensityMixin):
    r"""
    Finds the best fixed (in hindsight) rebalancing portfolio s
    (distribution over assets). Note that this model is NOT an online
    optimization algorithm since it optimizes over an entire asset dataset
    each step, not sample by sample.

    Solves the optimization problem:
        \arg\min_{x in S} \sum_{t=1}^{T} f_t(x)
    where
        f_t(x) = -\log(r_t^T x)

    Attributes:
        p_: The fitted portfolio
        P_: The fitted portfolio at each time point (only if continuous=True)
    """

    def __init__(self, eta_min=0., max_iter=None, continuous=False):
        """
        :param eta_min: Stop optimization if step size is smaller than this.
        :param max_iter: Maximum number of iterations to fit for. Set to
        None to use the max_iter=d, the dimension of the input.
        :param continuous: Whether to fit continuously to each time point or
        only to the entire dataset (final time).
        """
        super().__init__()
        self.eta_min = eta_min
        self.max_iter = max_iter
        self.continuous = continuous

    @staticmethod
    def grad_fn(R, x):
        T, d = R.shape

        # Grad of -log(r x) is
        #   r * (-1 / (r^T x)) = r * a
        # Where a is a scalar
        a = np.dot(R, x)
        a = - 1 / a
        a = np.reshape(a, (T, -1))

        return np.sum(R * a, axis=0)

    def fit(self, R: np.ndarray):
        """
        Fits a rebalancing constant portfolio, which is a point within the
        unit simplex to the stock price data in R.
        :param R: Array of shape (T,d) where T is the number of time points
        (e.g. days) and d is the number of different assets. Each entry is
        the asset price-ratio between the current and previous time.
        :return: self
        """
        if not self.continuous:
            self.p_ = self._fit(R)
        else:
            self.P_ = np.empty_like(R)
            for t in range(R.shape[0]):
                self.P_[t] = self._fit(R[0:t+1])
            self.p_ = self.P_[-1]

    def _fit(self, R: np.ndarray):
        # R is the reward (price ratio of day t+1 to day t) per asset
        assert R.ndim == 2
        T, d = R.shape
        R = R.astype(np.float32)

        I = np.eye(d, dtype=np.float32)

        # pt is the current iterate
        pt = I[0]
        max_iter = d if self.max_iter is None else self.max_iter
        for t in range(1, max_iter + 1):
            eta = 2 / (1 + t)
            if eta < self.eta_min:
                break

            # rt is the current return vector
            # Gradient of the function we're optimizing
            gt = self.grad_fn(R, pt)

            # Solve arg min_{v in V(S)} <v, gt>, where V(S) are the extreme
            # points of the Simplex and <.,.> is an inner product.
            # Solving this is equivalent to taking the standard basis vector
            # which selects the minimal element from gt.
            imin = np.argmin(gt)
            vt = I[imin]

            pt = pt + eta * (vt - pt)

        return pt

    def wealth(self, R: np.ndarray):
        """
        Calculate the wealth of the algorithm at each trading round.
        :param R: Array of shape (T,d) where T is the number of time points
        (e.g. days) and d is the number of different assets. Each entry is
        the asset price-ratio between the current and previous time.
        :return: Array of shape (T,), containing wealth of the algorithm at
        each trading round.
        """
        assert R.ndim == 2
        check_is_fitted(self, ['p_'])
        p = self.p_
        assert p.shape[0] == R.shape[1]

        return np.cumprod(np.sum(R * p, axis=1), axis=0)


class BestSingleAssetRebalancingPortfolio(BestFixedRebalancingPortfolio):
    """
    Finds the best fixed (in hindsight) rebalancing portfolio which only
    invests in a single asset.
    Note that this model is NOT an online optimization algorithm since it
    optimizes over an entire asset dataset each step, not sample by sample.

    Attributes:
        pt_: The fitted portfolio
    """

    def fit(self, R: np.ndarray):
        """
        Fits a rebalancing constant portfolio consisting of a single asset
        only.
        :param R: Array of shape (T,d) where T is the number of time points
        (e.g. days) and d is the number of different assets. Each entry is
        the asset price-ratio between the current and previous time.
        :return: self
        """
        # R is the reward (price ratio of day t+1 to day t) per asset
        assert R.ndim == 2
        T, d = R.shape
        R = R.astype(np.float32)

        idx_best_single = np.argmax(np.prod(R, axis=0))

        p_best_single = np.zeros((d,), dtype=np.float32)
        p_best_single[idx_best_single] = 1.

        self.p_ = p_best_single
        return self
