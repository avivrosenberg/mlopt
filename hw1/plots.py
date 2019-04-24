import os

import numpy as np
import matplotlib.pyplot as plt

import hw1.config as hw1cfg
from hw1.config import ExperimentResults, ExperimentConfig


def plot_from_file(results_filename, out_dir):
    """
    Plots experiment results from file.
    :param results_filename: Path of file containing serialized results.
    :param out_dir: output folder for plots.
    :return: List of filenames written.
    """
    results = hw1cfg.load_results(results_filename)
    return plot_experiments(results, out_dir)


def plot_experiments(results, out_dir):
    """
    Plots experiment results.
    :param results: A list/tuple of ExperimentResults
    :param out_dir: output folder for plots.
    :return: List of filenames written.
    """
    return [plot_experiment(r, out_dir) for r in results]


def plot_experiment(results: ExperimentResults, out_dir):
    """
    Plots a single experiment's results.
    :param results: An ExperimentResults.
    :param out_dir: output folder for plots.
    :return: List of filenames written.
    """
    fig, ax = plt.subplots(1, 1)
    fig: plt.Figure = fig
    ax: plt.Axes = ax
    cfg: ExperimentConfig = results.config

    for run_name, run_data in results.results_map.items():
        t_axis = np.arange(1, run_data.shape[1]+1)
        mean = run_data[0]
        sterr = run_data[1]

        ax.plot(t_axis, mean, label=run_name, linestyle='--', linewidth=0.7)
        ax.fill_between(t_axis, mean-sterr, mean+sterr, alpha=0.3)

    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$\vert f(\mathbf{x_t}) - f(\mathbf{x^{*}})\vert$')
    kappa = str(cfg.smax/cfg.smin) if cfg.smin > 0 else r'\infty'
    ax.set_title(rf'{cfg.name}, ($\kappa(A)={kappa}$)')
    ax.grid()
    ax.legend(loc='upper right')
    ax.set_xlim(auto=True)
    ax.set_yscale('log')
    ax.set_xscale('log')

    filename = os.path.join(out_dir, f'{cfg.name}')
    fmt = 'pdf'
    fig.savefig(f'{filename}.{fmt}', format=fmt)
