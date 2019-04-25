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


def plot_experiments(results, out_dir, **kw):
    """
    Plots experiment results.
    :param results: A list/tuple of ExperimentResults
    :param out_dir: output folder for plots.
    :return: List of filenames written.
    """
    return [plot_experiment(r, out_dir, **kw) for r in results]


def plot_experiment(results: ExperimentResults, out_dir,
                    no_plot_title=False, **kw):
    """
    Plots a single experiment's results.
    :param results: An ExperimentResults.
    :param out_dir: output folder for plots.
    :param no_plot_title: Whether to skip adding a title.
    :return: List of filenames written.
    """
    fig, ax = plt.subplots(1, 1)
    fig: plt.Figure = fig
    ax: plt.Axes = ax
    cfg: ExperimentConfig = results.config

    # Plot the optimizer losses
    for run_name, run_data in results.results_map.items():
        t_axis = np.arange(1, run_data.shape[1]+1)
        mean = run_data[0]
        sterr = run_data[1]

        ax.plot(t_axis, mean, label=run_name, linestyle='--', linewidth=0.7)
        ax.fill_between(t_axis, mean-sterr, mean+sterr, alpha=0.3)

    # Add eplison line
    t_axis = np.arange(1, results.config.n_iter+1)
    eps = np.full_like(t_axis, results.config.eps, dtype=np.float)
    ax.plot(t_axis, eps, 'k:', label=rf'$\epsilon=${results.config.eps}')

    # Title
    if not no_plot_title:
        kappa = str(cfg.smax/cfg.smin) if cfg.smin > 0 else r'\infty'
        ax.set_title(rf'{cfg.name}, ($\kappa(A)={kappa}$)')

    # Axes
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$\vert f(\mathbf{x_t}) - f(\mathbf{x^{*}})\vert$')
    ax.set_xlim(auto=True)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.grid()
    ax.legend(loc='upper right')

    filename = os.path.join(out_dir, f'{str.replace(cfg.name, " ", "_")}')
    fmt = 'pdf'
    fig.set_size_inches(8*0.8, 6*0.8)
    fig.savefig(f'{filename}.{fmt}', format=fmt,
                bbox_inches='tight', pad_inches=0.1)
