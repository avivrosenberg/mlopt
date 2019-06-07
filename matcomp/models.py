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

                pbar.set_description(pbar_desc(train_mse, test_mse))
                pbar.update()

                t += 1
                if t >= self.max_iter:
                    break
                if train_loss < self.tol:
                    break

        # Save training results
        self.M_ = Xt
        self.t_final_ = t
        self.train_losses_ = train_losses
        self.test_losses_ = test_losses

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

        # MS is the matrix of ratings representing the dataset.
        MS = np.zeros((self.n_users, self.n_movies), dtype=np.float)
        MS[X[:, 0], X[:, 1]] = y

        # Initial iterate: low-rank projection of dataset matrix
        X0 = self._project_fn(MS)

        # Define optimizer
        stepsize_gen = optim.stepsize_gen.const(self.eta)
        optimizer = opt.GradientDescent(
            X0,
            max_iter=self.max_iter + 1,
            grad_fn=lambda Xt: self.grad_fn(Xt, X, y),
            stepsize_gen=stepsize_gen,
            project_fn=self._project_fn,
        )

        # Run optimizer, yield iterates
        yield X0
        for Xt in optimizer:
            yield Xt


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
