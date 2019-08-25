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
        G_: Approximated max gradient
        D_: Approximated diameter of optimization region
    """

    def __init__(self, save_iterates=True):
        """
        :param save_iterates: Whether to save each iterate.
        """
        self.__rt__ = None
        self.save_iterates = save_iterates


    @staticmethod
    def loss_fn(r: np.ndarray, x: np.ndarray):
        return -np.log(np.dot(x, r))

    @staticmethod
    def grad_fn(r: np.ndarray, x: np.ndarray):
        return - r * (1. / np.dot(x, r))

    def fit(self, R: np.ndarray, **kw):
        """
        Fits a rebalancing constant portfolio, which is a point within the
        unit simplex to the stock price data in X.
        :param R: Array of shape (T,d) where T is the number of time points
        (e.g. days) and d is the number of different assets. Each entry is
        the asset price-ratio between the current and previous time.
        :return: self
        """
        # R is the reward (price ratio of day t+1 to day t) per asset
        assert R.ndim == 2
        R = R.astype(np.float32)

        # Save portfolios
        P = np.zeros_like(R) if self.save_iterates else None

        # Iterate over optimizer
        self.__rt__ = R[0]  # To allow _pt_generator access to rt
        for t, pt in enumerate(self._pt_generator(R)):
            self.__rt__ = R[t]

            if self.save_iterates:
                P[t] = pt

        # save fit results
        self.p_ = pt
        self.P_ = P
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

    def regret(self, R: np.ndarray, Pstar: np.ndarray, average=False):
        """
        Calculates the regret of the algorithm vs the best fixed portfolio
        in hindsight.
        :param R: Array of shape (T,d) containing the asset returns data
        that the model was fitted with.
        :param Pstar: Array of shape (T,d), each row t containing the best
        fixed portfolio in hindsight up to time t.
        :param average: Whether to calculate average regret.
        :return: An array of shape (T,) with the regret at each time.
        """
        check_is_fitted(self, ['p_', 'P_'])
        assert R.shape == Pstar.shape

        T, d = R.shape

        # Cumulative loss
        F = np.cumsum(-np.log(np.sum(R * self.P_, axis=1)))

        # Regret term
        Fstar = np.zeros_like(F)
        for t in range(T):
            # pstar is the best fixed portfolio up to time t
            pstar = Pstar[t]
            # Calculate the sum of the loss up to time t
            Fstar[t] = np.sum(-np.log(np.sum(R[0:t+1] * pstar, axis=1)))

        regret = F - Fstar

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
        self.D_ = math.sqrt(2)
        self.G_ = np.max(np.linalg.norm(R, axis=1) / np.sum(R, axis=1))
        self.eta_ = self.D_ / (self.G_ * math.sqrt(T))

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
        self.D_ = math.sqrt(math.log(d))
        self.G_ = np.max(np.max(np.abs(R), axis=1) / np.sum(R, axis=1))
        self.eta_ = self.D_ / (self.G_ * math.sqrt(2 * T))

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
        self.D_ = math.sqrt(2)
        self.G_ = np.max(np.linalg.norm(R, axis=1) / np.sum(R, axis=1))
        alpha = math.inf
        gamma = 0.5 * min(1 / 4 / self.G_ / self.D_, alpha)
        eps = 1 / (gamma ** 2) / (self.D_ ** 2)

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
    Finds the best fixed (in hindsight) rebalancing portfolio
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
