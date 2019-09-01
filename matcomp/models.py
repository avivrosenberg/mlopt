import abc
import copy
import inspect
import math
import os
import sys

import numpy as np
import tqdm
import scipy
import scipy.sparse.linalg

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

import optim.optimizers as opt
import optim.stepsize_gen
from optim import projections


class MatrixCompletion(abc.ABC, BaseEstimator, RegressorMixin):
    def __init__(self, n_users=1000, n_movies=1000,
                 max_iter=750, train_mse_tol=.05,
                 verbose=True, **kw):
        """
        Base matrix completion model.
        :param n_users: Number of users.
        :param n_movies: Number of movies.
        :param max_iter: Max number of iterations for training fit.
        :param train_mse_tol: Stop training if MSE is less than this.
        :param verbose: Whether to show training progress.
        """
        super().__init__()
        self.n_users = n_users
        self.n_movies = n_movies
        self.max_iter = max_iter
        self.train_mse_tol = train_mse_tol
        self.verbose = verbose

    def loss_fn(self, Xt, X, y):
        """
        Computes matrix completion loss function (minimization objective).
        :param Xt: A matrix of shape (n,m) where n is the number of
        users and m is the number of movies, representing the current
        iterate.
        :param X: An (N,2) tensor of user_id, movie_id pairs.
        :param y: An (N,) tensor of ratings,
        :return: The loss at point Xt.
        """
        loss = .5 * np.linalg.norm(Xt[X[:, 0], X[:, 1]] - y) ** 2
        return loss

    def grad_fn(self, Xt, X, y):
        """
        Computes the gradient of the matrix completion loss function at a
        specific point Xt.
        :param Xt: A matrix of shape (n,m) where n is the number of
        users and m is the number of movies, representing the current
        iterate.
        :param X: An (N,2) tensor of user_id, movie_id pairs.
        :param y: An (N,) tensor of ratings,
        :return: A tensor, the same shape as Xt, representing the gradient
        of the loss function at Xt.
        """
        # Note: float32 for speed
        grad_Xt = np.zeros((self.n_users, self.n_movies), dtype=np.float32)
        grad_Xt[X[:, 0], X[:, 1]] = (Xt[X[:, 0], X[:, 1]] - y)
        return grad_Xt

    def fit(self, X, y, Xtest=None, ytest=None):
        """
        Fits the model by finding a matrix M that fits the data.
        :param X: An (N,2) tensor of user_id, movie_id pairs.
        :param y: An (N,) tensor of ratings,
        :param Xtest: Optional test-set for evaluation every step.
        :param ytest: Optional test-set labels for evaluation every step.
        :return: self.
        """
        has_test = Xtest is not None and ytest is not None

        X, y = check_X_y(X, y)
        train_losses = np.full(self.max_iter, np.nan)
        test_mse_ = np.full(self.max_iter, np.nan)
        test_losses = None
        if has_test:
            Xtest, ytest = check_X_y(Xtest, ytest)
            test_losses = np.full(self.max_iter, np.nan)

        # Progress bar
        def pbar_desc(t_mse, v_mse=None):
            if v_mse is None:
                return f'[{self.name}] t_mse={t_mse:.3f}'
            else:
                return f'[{self.name}] t_mse={t_mse:.3f} v_mse={v_mse:.3f}'

        pbar_file = sys.stdout if self.verbose else open(os.devnull, 'w')

        # Training loop
        with tqdm.tqdm(total=self.max_iter, file=pbar_file) as pbar:
            t = 0
            # Run the implementation-specific fit function
            for Xt in self._fit(X, y):
                train_loss = self.loss_fn(Xt, X, y)
                train_losses[t] = train_loss
                train_mse = (2 / X.shape[0]) * train_loss
                test_mse = None

                if has_test:
                    test_loss = self.loss_fn(Xt, Xtest, ytest)
                    test_losses[t] = test_loss
                    test_mse = (2 / Xtest.shape[0]) * test_loss
                    test_mse_[t] = test_mse

                pbar.set_description(pbar_desc(train_mse, test_mse))
                pbar.update()

                t += 1
                if self.max_iter and t >= self.max_iter:
                    break
                if train_mse < self.train_mse_tol:
                    break

        # Save training results
        self.M_ = Xt
        self.t_final_ = t
        self.train_losses_ = train_losses
        self.test_losses_ = test_losses
        self.test_mse_ = test_mse_

        return self

    def predict(self, X):
        # Input validation
        check_is_fitted(self, ['M_'])
        X = check_array(X)

        return self.M_[X[:, 0], X[:, 1]]

    def get_params(self, deep=True):
        subclass_params = BaseEstimator.get_params(self, deep)
        # HACK: get base class params using sklearn's get_params...
        # Otherwise it's impossible to change baseclass params in cross
        # validation.
        c = copy.copy(self)
        c.__class__ = MatrixCompletion
        baseclass_params = BaseEstimator.get_params(c, deep)

        subclass_params.update(baseclass_params)
        return subclass_params

    @property
    @abc.abstractmethod
    def name(self):
        pass

    @abc.abstractmethod
    def _fit(self, X, y):
        """
        Implementation-specific fit function.
        This should be a generator which generates iterates, i.e. predictions
        of the completed matrix.
        :param X: An (N,2) tensor of user_id, movie_id pairs.
        :param y: An (N,) tensor of ratings,
        :return: Generates a sequence of matrices of
        shape(self.n_users, self.n_movies).
        """
        pass


class RankProjectionMatrixCompletion(MatrixCompletion):
    """
    Computes a matrix completion using an approximate projection to a
    low-rank matrix at each step.
    """

    def __init__(self, rank=5, eta=0.5, proj_n_iter=5, **kwargs):
        """
        :param rank: Desired maximal rank of result.
        :param eta: Learning rate.
        :param proj_n_iter: Number of iterations when computing approximate
        projection.
        :param kwargs: Extra args for the MatrixCompletion base class,
        see its init.
        """
        super().__init__(**kwargs)
        self.rank = rank
        self.eta = eta
        self.proj_n_iter = proj_n_iter

    @property
    def name(self):
        return 'rp'

    def _project_fn(self, Xt):
        """
        Projects a ratings matrix to a lower-rank representation using an
        approximate truncated SVD.
        :param Xt: Ratings matrix of shape (n_users, n_movies).
        :return: Low-rank version of Xt, same shape.
        """
        tsvd = TruncatedSVD(self.rank,
                            algorithm="randomized",
                            n_iter=self.proj_n_iter)
        # Fit approximate low-rank SVD decomposition
        Xt_reduced = tsvd.fit_transform(Xt)
        # Transform back to original shape: this is a low-rank projection
        return tsvd.inverse_transform(Xt_reduced)

    def _fit(self, X, y):
        # MS is the matrix of ratings representing the dataset.
        MS = np.zeros((self.n_users, self.n_movies), dtype=np.float32)
        MS[X[:, 0], X[:, 1]] = y

        # Initial iterate: low-rank projection of dataset matrix
        X0 = self._project_fn(MS)

        # Define optimizer
        stepsize_gen = optim.stepsize_gen.const(self.eta)
        optimizer = opt.GradientDescent(
            X0,
            grad_fn=lambda Xt: self.grad_fn(Xt, X, y),
            stepsize_gen=stepsize_gen,
            project_fn=self._project_fn,
        )

        # Run optimizer, yield iterates
        yield X0
        for Xt in optimizer:
            yield Xt


class FactorizedFormMatrixCompletion(MatrixCompletion):
    """
    Computes matrix-completion using a factorized-form optimization.
    I.e., instead of directly optimizing the target matrix X, we optimize
    two low-rank matrices such that X = U V^T.
    """

    def __init__(self, rank=5, **kwargs):
        """
        :param rank: Desired maximal rank of result. This will be the
        maximal rank of the two optimized matrices.
        :param kwargs: Extra args for the MatrixCompletion base class,
        see its init.
        """
        super().__init__(**kwargs)
        self.rank = rank

    @property
    def name(self):
        return 'ff'

    def _fit(self, X, y):
        # MS is the matrix of ratings representing the dataset of shape (n, m).
        MS = np.zeros((self.n_users, self.n_movies), dtype=np.float32)
        MS[X[:, 0], X[:, 1]] = y

        # Fit approximate low-rank SVD decomposition to get initial starting
        # matrices U, V.
        tsvd = TruncatedSVD(self.rank, algorithm="randomized", )
        Ut = tsvd.fit_transform(MS)  # U shape is (n,r)
        Vt = np.transpose(tsvd.components_)  # V shape is (m,r)

        # g(U,V) = f(U V^T)
        # d/dU g(U,V) = d/dU f(U V^T) = [ grad f(U V^T) ] V
        # d/dV g(U,V) = d/dV f(U V^T) = [ grad f(U V^T) ]^T U

        def grad_fn_U(_):
            # Note: using pre-calculated grad_f for speed, see below
            return np.matmul(grad_f, Vt)

        def grad_fn_V(_):
            return np.matmul(grad_f.T, Ut)

        def stepsize_U():
            while True:
                lambda_max_VV = np.linalg.eigvalsh(np.matmul(Vt.T, Vt))[-1]
                yield 1 / lambda_max_VV

        def stepsize_V():
            while True:
                lambda_max_UU = np.linalg.eigvalsh(np.matmul(Ut.T, Ut))[-1]
                yield 1 / lambda_max_UU

        optimizer_U = opt.GradientDescent(
            Ut, grad_fn=grad_fn_U, stepsize_gen=stepsize_U(),
        )
        optimizer_V = opt.GradientDescent(
            Vt, grad_fn=grad_fn_V, stepsize_gen=stepsize_V(),
        )

        iter_U, iter_V = iter(optimizer_U), iter(optimizer_V)
        while True:
            Xt = np.matmul(Ut, Vt.T)
            grad_f = self.grad_fn(Xt, X, y)  # calc grad_f for both updates
            yield Xt
            Ut = next(iter_U)
            Vt = next(iter_V)


class ConvexRelaxationMatrixCompletion(MatrixCompletion):
    """
    Computes matrix-completion using a convex-relaxation method.
    Instead of optimizing for a low-rank matrix, an upper bound on the
    nuclear norm of the matrix is used instead.
    """

    def __init__(self, tau=1500, power_method_iters=30, **kwargs):
        """
        :param tau: Desired maximal nuclear norm value of result.
        :param power_method_iters: Amount of iterations to run the power method
        :param kwargs: Extra args for the MatrixCompletion base class,
        see its init.
        """
        super().__init__(**kwargs)
        self.tau = tau
        self.t = 1
        self.power_method_iters = power_method_iters

    @property
    def name(self):
        return 'cr'

    def _fit(self, X, y):
        # MS is the matrix of ratings representing the dataset of shape (n, m).
        MS = np.zeros((self.n_users, self.n_movies), dtype=np.float32)
        MS[X[:, 0], X[:, 1]] = y
        ms_shape = MS.shape

        if self.n_users > self.n_movies:
            MS = MS.transpose()

        # Compute sigma max over the entire dataset to be used throughout
        # the training process
        tsvd = TruncatedSVD(n_components=1, algorithm="randomized")
        tsvd.fit(X=MS)
        sigma_max = tsvd.singular_values_[0]

        # Release unnecessary memory
        del tsvd, MS

        # Set the constant skeleton for the A matrix
        K = (self.n_users + self.n_movies)
        A = np.zeros((K, K)).astype(np.float32)
        A[0:self.n_users, 0:self.n_users] = sigma_max * np.eye(self.n_users)
        A[-self.n_movies:, -self.n_movies:] = sigma_max * np.eye(self.n_movies)

        # Initialize Xt into an arbitrary point
        Xt = np.eye(*ms_shape, dtype=np.float32)
        Xt *= self.tau / 2 / ms_shape[0]  # nuclear norm will be tau/2
        yield Xt

        # Yield iterates
        while True:
            eta_t = 2 / (self.t + 1)
            self.t += 1

            grad_Xt = self.grad_fn(Xt=Xt, X=X, y=y)

            A[0:self.n_users:, self.n_users:] = -grad_Xt
            A[self.n_users:, 0:self.n_users] = -grad_Xt.transpose()

            # wt = self._compute_largest_eigenvec(A=A)
            _, wt = scipy.sparse.linalg.eigs(A=A, k=1)
            wt = wt.astype(np.float32)

            u = wt[0:self.n_users] / np.linalg.norm(x=wt[0:self.n_users])
            v = wt[self.n_users:] / np.linalg.norm(x=wt[self.n_users:])

            Vt = self.tau * np.matmul(u, v.T)

            Xt += eta_t * (Vt - Xt)

            yield Xt

    def _compute_largest_eigenvec(self, A):
        """
        A utility method for computing the eigenvector which corresponds to the
        largest eigenvalue of A, using the power-iterations method

        :param A: The matrix whose eigenvector we wish to compute
        :return: w - The corresponding eigenvector
        """

        wt = np.random.randn(A.shape[0])

        for _ in range(self.power_method_iters):
            w = np.matmul(A, wt)
            wt = w / np.linalg.norm(x=w)

        return wt


class SVRGCGMatrixCompletion(MatrixCompletion):
    """
    Implements Algorithm 1 from
        Garber, D., & Kaplan, A. (2018).
        Fast Stochastic Algorithms for Low-rank and Nonsmooth Matrix Problems.
        arXiv preprint arXiv:1809.10477.
    """

    def __init__(self, tau=1500, eps=1., reg_lambda=2., mu=None,
                 sigma_n=1., sigma_q=5., svd_rank=1, svd_n_iter=None,
                 yield_every=10, **kw):
        """
        :param tau: Nuclear norm bound
        :param eps: Approximation error for optimization
        :param reg_lambda: Regularization factor
        :param mu: Smoothness parameter of approximate L1 regularizer
        :param sigma_n: Variance of noise added to observations
        :param sigma_q: Variance of noise for stochastic gradients
        :param yield_every: Yield (produce) a new iterate every this-number
        of iterations of the inner loop. Set to None to only yield on outer
        loop (epoch). Set to 1 to yield every inner-loop iterate.
        :param kw: Extra args for base class
        """
        super().__init__(**kw)
        # TODO: Validation
        self.tau = tau
        self.eps = eps
        self.reg_lambda = reg_lambda
        self.mu = mu if mu is not None else eps / (self.n_movies ** 2)
        self.sigma_n = sigma_n
        self.sigma_q = sigma_q
        self.svd_rank = svd_rank
        self.svd_n_iter = svd_n_iter
        self.yield_every = yield_every

    @property
    def name(self):
        return 'svrgcg'

    def regularization_huber(self, Xt):
        Xt_abs = np.abs(Xt)
        idx_lt_mu = Xt_abs <= self.mu
        idx_gt_mu = ~idx_lt_mu

        # Calculate Huber function, a smooth approximation of L1-norm
        Ht = np.zeros_like(Xt)
        Ht[idx_lt_mu] = Xt[idx_lt_mu] ** 2 / (2 * self.mu)
        Ht[idx_gt_mu] = Xt_abs[idx_gt_mu] - self.mu / 2

        return np.sum(Ht)

    def regularization_huber_grad(self, Xt):
        Xt_abs = np.abs(Xt)
        idx_lt_mu = Xt_abs <= self.mu
        idx_gt_mu = ~idx_lt_mu

        Ht = np.zeros_like(Xt)
        Ht[idx_lt_mu] = Xt[idx_lt_mu] / self.mu
        Ht[idx_gt_mu] = np.sign(Xt[idx_gt_mu])

        return Ht

    def full_loss_fn(self, Xt, M0):
        G = 0.5 * np.linalg.norm(Xt - M0) ** 2
        R = self.reg_lambda * self.regularization_huber(Xt)
        return G + R

    def _fit(self, X, y):
        # M0 is the matrix of ratings representing the dataset of shape (n, m).
        M0 = np.zeros((self.n_users, self.n_movies), dtype=np.float32)
        M0 += np.random.randn(*M0.shape) * self.sigma_n
        M0[X[:, 0], X[:, 1]] = y
        m, d = M0.shape

        nuclear_norm_projection = projections.NuclearNormProjection(
            self.tau, self.svd_rank, self.svd_n_iter, always_project=True)

        # Initial iterate
        Xs = np.eye(m, d, dtype=np.float32)
        Xs *= self.tau / 2 / m  # nuclear norm will be tau/2
        yield Xs

        # Parameters
        beta = 1 + self.reg_lambda / self.mu
        c0 = abs(self.full_loss_fn(Xs, M0) -
                 self.full_loss_fn(M0, M0))  # Initial loss (upper bound)
        S = math.ceil(math.log2(c0 / self.eps) + 2)  # Outer loop max
        T = math.ceil(8 * math.log(8) / 3 * beta + 1)  # Inner loop max

        k_s_gen = optim.stepsize_gen.custom(
            lambda s: math.ceil(32 * self.sigma_n / c0 * 2 ** (s - 1)),
            idx_range=range(1, 2 ** 63 - 1)
        )

        # Loop over epochs
        for s in range(S):
            k_s = next(k_s_gen)
            eta_t_gen = optim.stepsize_gen.const(
                self.mu / (2 * self.reg_lambda + 2 * self.mu)
            )

            # Compute snapshot gradient: Average of k_s stochastic gradients,
            # where each stochastic gradient is Xs - M_i and M_i is a random
            # matrix from a normal distribution with sigma_q variance
            M_i = M0 + self.sigma_q * np.random.randn(k_s, m, d)
            G_i = np.repeat(Xs[np.newaxis, ...], k_s, axis=0)
            Gsnap = np.mean(G_i - M_i, axis=0)
            del G_i, M_i

            # Inner loop to generate iterates
            Xst = Xs
            for t in range(T):
                eta_t = next(eta_t_gen)

                # Estimate G(X) gradient. Note that we don't need to
                # sample k_t stochastic grads here and average them because
                # the random parts (M_i) cancel out in this case.
                Gst_hat = Xst - Xs + Gsnap

                # Calculate gradient of the smooth regularizer R(X)
                Gst_R = self.regularization_huber_grad(Xst)

                # Step in the direction of the negative total gradient
                At = Xst - (1 / 2 / beta / eta_t) * (Gst_hat + Gst_R)

                # Project onto nuclear norm ball
                Vt = nuclear_norm_projection(At)

                # Update current iterate
                Xst = (1 - eta_t) * Xst + eta_t * Vt
                del Gst_hat, Gst_R, At, Vt

                if self.yield_every and (t % self.yield_every == 0):
                    yield Xst

            Xs = Xst
            if not self.yield_every or (t % self.yield_every != 0):
                yield Xs


###
# Collect parameter names from all model classes
ALL_PARAMS = {}

model_classes = inspect.getmembers(
    sys.modules[__name__],
    lambda m: inspect.isclass(m) and m.__module__ == __name__
)

for class_name, model_class in model_classes:
    if not inspect.isabstract(model_class):
        ALL_PARAMS.update(model_class().get_params())
ALL_PARAMS.pop('n_users')
ALL_PARAMS.pop('n_movies')
