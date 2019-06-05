import abc
import os
from typing import Tuple

import numpy as np
import pandas as pd

from util.util import download_data, cachedproperty

URL_ML_100K = "http://files.grouplens.org/datasets/movielens/ml-100k.zip"
URL_ML_1M = "http://files.grouplens.org/datasets/movielens/ml-1m.zip"
DOWNLOAD_DIR = "./data"


class MovieLensDataset(abc.ABC):
    def __init__(self, dataset_url, download_dir=DOWNLOAD_DIR):
        _, data_dir = download_data(download_dir, dataset_url)
        self.data_dir = data_dir
        self._users, self._movies, self._ratings = None, None, None
        self._user_occupations, self._movie_genres = None, None

    def rating_samples(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the rating data as as sample tensors.
        :return: A tuple (X, y) where X is a (N,2) tensor of user_id,movie_id
        pairs and y is a (N,) tensor of ratings.
        """
        X = self.ratings[["user_id", "movie_id"]].to_numpy()
        y = self.ratings[["rating"]].to_numpy().reshape(-1)
        return X, y

    def ratings_matrix(self) -> np.ndarray:
        """
        Returns the rating data as as single matrix.
        :return: Matrix of shape (n,m) where n is the number of users and m is
        the number of movies. Each entry in the matrix corresponds to a user
        rating.
        """
        X, y = self.rating_samples()

        # Movie and user ids are not contiguous
        n, m = np.max(X + 1, axis=0)

        M = np.zeros((n, m), dtype=np.float)
        M[X[:, 0], X[:, 1]] = y
        return M

    @cachedproperty
    @abc.abstractmethod
    def users(self) -> pd.DataFrame:
        pass

    @cachedproperty
    @abc.abstractmethod
    def movies(self) -> pd.DataFrame:
        pass

    @cachedproperty
    @abc.abstractmethod
    def ratings(self) -> pd.DataFrame:
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

    @property
    def n_users(self):
        return len(self.users)

    @property
    def n_movies(self):
        return len(self.movies)


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
