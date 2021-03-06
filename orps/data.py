import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from util.util import download_data, cachedproperty

URL_EHAZAN_PFDATA = "http://www.cs.princeton.edu/~ehazan/pfdata/" \
                    "data_490_1000.mat"
DOWNLOAD_DIR = "./data"


class EHazanPFDataset(object):
    def __init__(self, dataset_url=URL_EHAZAN_PFDATA,
                 download_dir=DOWNLOAD_DIR):

        self.ds_file, _ = download_data(download_dir, dataset_url)
        mat = scipy.io.loadmat(self.ds_file)

        A = mat['A']  # 490x1000, contains prices of 490 stocks on 1000 days
        B = mat['B']  # 491x1, contains filenames e.g. 'AAPL.csv'

        Bflat = np.hstack(B.flat)  # flatten array of numpy arrays of string
        asset_names = map(lambda s: s.split('.')[0], Bflat)

        # We'll treat each asset as a feature (column) and each day as a
        # sample (row).
        self._df = pd.DataFrame(A.T, columns=asset_names, dtype=np.float32)

    def __len__(self):
        return len(self._df)

    def __getitem__(self, item):
        if isinstance(item, (str, tuple)):
            # Indexing with a string or tuple of strings will return
            # features (columns).
            return self._df.loc[:, item].to_numpy()
        elif isinstance(item, (int, slice)):
            # Indexing or slicing numerically will return a rows (samples).
            return self._df.iloc[item, :].to_numpy()
        else:
            raise ValueError(f"Unknown index type: {type(item)}")

    def asset_names(self):
        return tuple(self._df.columns)

    def to_numpy(self):
        return self._df.to_numpy(copy=True)

    def plot_single(self, i, label=None, ax: plt.Axes = None):
        data = self._df.iloc[:, i].to_numpy()
        t = np.arange(len(data))

        if not label:
            label = self.asset_names()[i]

        if ax is None:
            fig, ax = plt.subplots(1, 1)
            ax.set_xlabel('t (days)')
            ax.set_ylabel('price (USD)')

        ax.plot(t, data, label=label)
        ax.legend()
        return ax

    def plot_random(self, n, ax: plt.Axes = None):
        T, d = self._df.shape
        for i in np.random.choice(d, n):
            ax = self.plot_single(i, ax=ax, label=None)

        return ax
