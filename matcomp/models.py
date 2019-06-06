import os
import sys

import tqdm
import numpy as np
from sklearn.decomposition import TruncatedSVD

import optim.stepsize_gen
import optim.optimizers as opt

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class RankProjectionMatrixCompletion(BaseEstimator, ClassifierMixin):
    def __init__(self, n_users, n_movies,
                 rank=20, eta=0.5, max_iter=10 ** 4, tol=1.,
                 verbose=True):
        """

        :param n_users:
        :param n_movies:
        :param rank:
        :param eta:
        :param max_iter:
        :param tol:
        :param verbose:
        """
        super().__init__()
        self.n_users = n_users
        self.n_movies = n_movies
        self.rank = rank
        self.eta = eta
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = True
        self._name = 'rpmc'

    def fit(self, X, y):
        """
        Fits the model by finding a matrix M that fits the data.
        :param X: An (N,2) tensor of user_id, movie_id pairs
        :param y: An (N,) tensor of ratings
        :return: self.
        """
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        # MS is the matrix of rating representing the dataset.
        # Movie and user ids are not contiguous
        MS = np.zeros((self.n_users, self.n_movies), dtype=np.float)
        MS[X[:, 0], X[:, 1]] = y

        def loss_fn(Xt):
            """
            Computes matrix completion loss function (minimization objective).
            :param Xt: A matrix of shape (n,m) where n is the number of
            users and m is the number of movies, representing the current
            iterate.
            :return: The loss at point Xt.
            """
            loss = .5 * np.linalg.norm(Xt[X[:, 0], X[:, 1]] - y) ** 2
            return loss

        def grad_fn(Xt, k=1):
            """
            Computes the gradient of the matrix completion loss function at a
            specific point Xt.
            :param Xt: A matrix of shape (n,m) where n is the number of
            users and m is the number of movies, representing the current
            iterate.
            :return: A tensor, the same shape as Xt, representing the gradient
            of the loss function at Xt.
            """
            grad_Xt = np.zeros((self.n_users, self.n_movies), dtype=np.float)
            grad_Xt[X[:, 0], X[:, 1]] = (Xt[X[:, 0], X[:, 1]] - y)
            return grad_Xt

        def project_fn(Xt):
            tsvd = TruncatedSVD(self.rank, algorithm="randomized")
            # Fit approximate low-rank SVD decomposition
            Xt_reduced = tsvd.fit_transform(Xt)
            # Transform back to original shape: this is a low-rank projection
            return tsvd.inverse_transform(Xt_reduced)

        # Initial point: low-rank projection of dataset
        X0 = project_fn(MS)
        losses = np.full(self.max_iter + 1, np.nan)
        losses[0] = loss_fn(X0)

        # Define optimizer
        i=2
        stepsize_gen = optim.stepsize_gen.const(self.eta)
        optimizer = opt.GradientDescent(X0,
                                        max_iter=self.max_iter,
                                        grad_fn=grad_fn,
                                        stepsize_gen=stepsize_gen,
                                        project_fn=project_fn, )

        # Run optimizer
        pbar_desc = lambda t: f'[{self._name}] loss={losses[t]:.3f}'
        pbar_file = sys.stdout if self.verbose else open(os.devnull, 'w')
        with tqdm.tqdm(desc=pbar_desc(0),
                       total=self.max_iter, file=pbar_file) as pbar:

            for t, Xt in enumerate(optimizer, start=1):
                losses[t] = loss_fn(Xt)

                pbar.set_description(pbar_desc(t))
                pbar.update()

                if losses[t] < self.tol:
                    break

        # Save training results
        self.M_ = Xt
        self.losses_ = losses
        self.t_final_ = t

        return self

    def predict(self, X):
        # Input validation
        check_is_fitted(self, ['M_'])
        X = check_array(X)

        return self.M_[X[:, 0], X[:, 1]]
