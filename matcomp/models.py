import abc
import copy
import inspect
import os
import sys

import numpy as np
import tqdm
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

import optim.optimizers as opt
import optim.stepsize_gen


class MatrixCompletion(abc.ABC, BaseEstimator, RegressorMixin):
    def __init__(self, n_users=1000, n_movies=1000,
                 max_iter=10 ** 4, tol=1.,
                 verbose=True, **kw):
        """
        Base matrix completion model.
        :param n_users: Number of users.
        :param n_movies: Number of movies.
        :param max_iter: Max number of iterations for training fit.
        :param tol: Stop training if loss is less that this.
        :param verbose: Whether to show training progress.
        """
        super().__init__()
        self.n_users = n_users
        self.n_movies = n_movies
        self.max_iter = max_iter
        self.tol = tol
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
        grad_Xt = np.zeros((self.n_users, self.n_movies), dtype=np.float)
        grad_Xt[X[:, 0], X[:, 1]] = (Xt[X[:, 0], X[:, 1]] - y)
        return grad_Xt

    def fit(self, X, y):
        """
        Fits the model by finding a matrix M that fits the data.
        :param X: An (N,2) tensor of user_id, movie_id pairs.
        :param y: An (N,) tensor of ratings,
        :return: self.
        """
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        # Training: run the implementation-specific fit function
        Xt_final, t_final, losses = self._fit(X, y)

        # Save training results
        self.M_ = Xt_final
        self.t_final_ = t_final
        self.losses_ = losses

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
        pass


class RankProjectionMatrixCompletion(MatrixCompletion):
    def __init__(self, rank=20, eta=0.5, proj_n_iter=5, **kwargs):
        """
        Computes a matrix completion using an approximate projection to a
        low-rank matrix at each step.
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
        """
        Fits the model by finding a matrix M that fits the data.
        :param X: An (N,2) tensor of user_id, movie_id pairs.
        :param y: An (N,) tensor of ratings,
        :return: 3-tuple containing final iterate Xt, array of losses and
        final iteration t.
        """

        # MS is the matrix of ratings representing the dataset.
        MS = np.zeros((self.n_users, self.n_movies), dtype=np.float)
        MS[X[:, 0], X[:, 1]] = y

        # Initial iterate: low-rank projection of dataset matrix
        X0 = self._project_fn(MS)
        losses = np.full(self.max_iter + 1, np.nan)
        losses[0] = self.loss_fn(X0, X, y)

        # Define optimizer
        stepsize_gen = optim.stepsize_gen.const(self.eta)
        optimizer = opt.GradientDescent(
            X0,
            max_iter=self.max_iter + 1,
            grad_fn=lambda Xt: self.grad_fn(Xt, X, y),
            stepsize_gen=stepsize_gen,
            project_fn=self._project_fn,
        )

        # Run optimizer
        pbar_desc = lambda m: \
            f'[{self.name}({self.rank},{self.eta:.1f})] mse={m:.3f}'
        pbar_file = sys.stdout if self.verbose else open(os.devnull, 'w')
        with tqdm.tqdm(desc=pbar_desc(0),
                       total=self.max_iter, file=pbar_file) as pbar:

            for t, Xt in enumerate(optimizer, start=1):
                loss = self.loss_fn(Xt, X, y)
                mse = (2 / X.shape[0]) * loss

                pbar.set_description(pbar_desc(mse))
                pbar.update()

                losses[t] = loss
                if losses[t] < self.tol:
                    break

        return Xt, t, losses


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
