import abc
import math
import os
from typing import Tuple

import numpy as np
import pandas as pd

from util.util import download_data, cachedproperty

URL_ML_100K = "http://files.grouplens.org/datasets/movielens/ml-100k.zip"
URL_ML_1M = "http://files.grouplens.org/datasets/movielens/ml-1m.zip"
DOWNLOAD_DIR = "./data"


class MatrixCompletionDataset(abc.ABC):
    @property
    @abc.abstractmethod
    def n(self):
        """
        :return: Number of items in first dimension (e.g. users)
        """
        pass

    @property
    @abc.abstractmethod
    def m(self):
        """
        :return: Number of items in second dimension (e.g. movies)
        """
        pass

    @property
    @abc.abstractmethod
    def N(self):
        """
        :return: Total number of samples in this dataset (non zero entries
        in the matrix).
        """
        pass

    @abc.abstractmethod
    def samples(self):
        """
        :return: (X, y) where X is a matrix of shape (N,2) containing indices
        and y is a vector of shape (N,) containing the values at those
        indices (e.g. movie ratings). N is the number of samples.
        """
        pass

    @abc.abstractmethod
    def samples_matrix(self):
        """
        :return: A matrix of shape (n, d) containing indices
        and y is a vector of shape (N,) containing the values at those
        indices (e.g. movie ratings). N is the number of samples.
        """
        pass


class SyntheticDataset(MatrixCompletionDataset):
    """
    Generates a synthetic dataset for matrix completion of the form
        M = Y Y^T + N
    where Y is a matrix of shape (d,r) and r is the desired rank of the result,
    and N is a (d,d) matrix of Gaussian white noise.
    The entries in Y are zero with probability p0, otherwise they're uniform
    U[1,...,k].
    """

    def __init__(self, d: int, r: int, p0: float = None, k=10, sigma2_n=1.,
                 dtype=np.float32):
        """
        :param d: Size of matrix (will be dxd).
        :param r: Rank of matrix.
        :param p0: Probability of a zero entry.
        If None, will be set to 1 - 1/sqrt(d).
        :param k: Maximal value of a non-zero entry (inclusive).
        :param sigma2_n: Variance of the added noise.
        """
        super().__init__()
        assert d > 0
        assert r >= 1
        assert p0 is None or 0 < p0 < 1
        assert k >= 2
        assert sigma2_n >= 0.
        self.d = d
        self.r = r
        self.p0 = p0 if p0 else 1 - 1 / math.sqrt(d)
        self.k = k
        self.sigma2_n = sigma2_n

        # Generate the data
        Pmask = np.random.rand(self.d, self.r)
        Y = np.random.randint(1, self.k + 1, size=Pmask.shape).astype(dtype)
        Y[Pmask < self.p0] = 0.

        N = np.random.randn(self.d, self.d) * self.sigma2_n
        YYT = np.matmul(Y, Y.T)

        # Create data matrix
        self.M = YYT + N

        # Create sample tensors
        i, j = np.nonzero(YYT)
        self.X = np.vstack((i, j)).T
        self.y = self.M[i, j]

    @property
    def n(self):
        return self.d

    @property
    def m(self):
        return self.d

    @property
    def N(self):
        return len(self.y)

    def samples(self):
        return self.X, self.y

    def samples_matrix(self):
        return self.M


class MovieLensDataset(MatrixCompletionDataset):
    def __init__(self, dataset_url, download_dir=DOWNLOAD_DIR):
        _, data_dir = download_data(download_dir, dataset_url)
        self.data_dir = data_dir
        self._users, self._movies, self._ratings = None, None, None
        self._user_occupations, self._movie_genres = None, None

    def samples(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the rating data as as sample tensors.
        :return: A tuple (X, y) where X is a (N,2) tensor of user_id,movie_id
        pairs and y is a (N,) tensor of ratings.
        """
        X = self.ratings[["user_id", "movie_id"]].to_numpy()
        y = self.ratings[["rating"]].to_numpy().reshape(-1)
        return X, y

    def samples_matrix(self) -> np.ndarray:
        """
        Returns the rating data as as single matrix.
        :return: Matrix of shape (n,m) where n is the number of users and m is
        the number of movies. Each entry in the matrix corresponds to a user
        rating.
        """
        X, y = self.samples()
        M = np.zeros((self.n_users, self.n_movies), dtype=np.float)
        M[X[:, 0], X[:, 1]] = y
        return M

    @cachedproperty
    @abc.abstractmethod
    def users(self) -> pd.DataFrame:
        """
        :return: Dataframe containing user features.
        """
        pass

    @cachedproperty
    @abc.abstractmethod
    def movies(self) -> pd.DataFrame:
        """
        :return: Dataframe containing movie features.
        """
        pass

    @cachedproperty
    @abc.abstractmethod
    def ratings(self) -> pd.DataFrame:
        """
        :return: Dataframe containing tuples of (user_id, movie_id, rating)
        """
        pass

    @cachedproperty
    def movie_genres(self) -> dict:
        return {
            'unknown': 0, 'Adventure': 2, 'Animation': 3, 'Children''s': 4,
            'Comedy': 5, 'Crime': 6, 'Documentary': 7, 'Drama': 8,
            'Fantasy': 9, 'Film-Noir': 10, 'Horror': 11, 'Musical': 12,
            'Mystery': 13, 'Romance': 14, 'Sci-Fi': 15, 'Thriller': 16,
            'War': 17, 'Western': 18,
        }

    @cachedproperty
    def user_occupations(self) -> dict:
        return {
            "other": 0, "academic/educator": 1, "artist": 2,
            "clerical/admin": 3, "college/grad student": 4,
            "customer service": 5, "doctor/health care/healthcare": 6,
            "executive/managerial": 7, "farmer": 8, "homemaker": 9,
            "K-12 student": 10, "lawyer": 11, "programmer": 12, "retired": 13,
            "sales/marketing/salesman": 14, "scientist": 15,
            "self-employed": 16, "technician/engineer": 17,
            "tradesman/craftsman": 18, "unemployed": 19, "writer": 20,
            "administrator": 21, "entertainment": 22, "librarian": 23,
            "none": 24,
        }

    @cachedproperty
    def n_users(self):
        """
        :return: Number of users in the dataset.
        """
        # Maximal id, not number of unique ids, since we use the user_ids as
        # an index into a the rating matrix. Add one since it's zero-based.
        return self.ratings['user_id'].max() + 1

    @cachedproperty
    def n_movies(self):
        """
        :return: Number of movies in the dataset.
        """
        return self.ratings['movie_id'].max() + 1

    @property
    def n(self):
        return self.n_users

    @property
    def m(self):
        return self.n_movies

    @property
    def N(self):
        return len(self.ratings)

    def __repr__(self):
        return f'{self.__class__.__name__}(n_users={self.n_users}, ' \
               f'n_movies={self.n_movies}, data_dir={self.data_dir})'


class MovieLens1M(MovieLensDataset):
    def __init__(self):
        super().__init__(URL_ML_1M)
        self.csv_args = dict(sep="::", header=None, engine='python')
        self.users_file = os.path.join(self.data_dir, 'users.dat')
        self.movies_file = os.path.join(self.data_dir, 'movies.dat')
        self.ratings_file = os.path.join(self.data_dir, 'ratings.dat')

    @cachedproperty
    def users(self):
        df_users = pd.read_csv(self.users_file, **self.csv_args,
                               names=["user_id", "gender", "age_group",
                                      "occupation", "zipcode"], )
        df_users['user_id'] -= 1
        return df_users

    @cachedproperty
    def movies(self):
        df_movies = pd.read_csv(self.movies_file, **self.csv_args,
                                names=["movie_id", "title", "genres"])
        df_movies['movie_id'] -= 1
        return df_movies

    @cachedproperty
    def ratings(self):
        df_ratings = pd.read_csv(self.ratings_file, **self.csv_args,
                                 usecols=[0, 1, 2],
                                 names=["user_id", "movie_id", "rating"], )
        df_ratings['user_id'] -= 1
        df_ratings['movie_id'] -= 1
        return df_ratings


class MovieLens100K(MovieLensDataset):
    def __init__(self):
        super().__init__(URL_ML_100K)
        self.csv_args = dict(header=None, engine='c')
        self.users_file = os.path.join(self.data_dir, 'u.user')
        self.movies_file = os.path.join(self.data_dir, 'u.item')
        self.ratings_file = os.path.join(self.data_dir, 'u.data')

    @cachedproperty
    def users(self):
        def convert_occupation(o_str):
            o_num = self.user_occupations.get(o_str)
            if o_num is not None:
                return o_num
            for k, v in self.user_occupations.items():
                if o_str in k:
                    self.user_occupations[o_str] = v
                    return v
            return 0

        df_users = pd.read_csv(self.users_file, **self.csv_args, sep='|',
                               names=["user_id", "age_group", "gender",
                                      "occupation", "zipcode"],
                               converters={3: convert_occupation})
        df_users['user_id'] -= 1
        return df_users

    @cachedproperty
    def movies(self):
        df_movies = pd.read_csv(self.movies_file, **self.csv_args, sep='|',
                                encoding="iso-8859-1",
                                usecols=[0, 1, ],
                                names=["movie_id", "title", ])
        df_movies['movie_id'] -= 1
        return df_movies

    @cachedproperty
    def ratings(self):
        df_ratings = pd.read_csv(self.ratings_file, **self.csv_args, sep='\t',
                                 usecols=[0, 1, 2],
                                 names=["user_id", "movie_id", "rating"], )
        df_ratings['user_id'] -= 1
        df_ratings['movie_id'] -= 1
        return df_ratings
