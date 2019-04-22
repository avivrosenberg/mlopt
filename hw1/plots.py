import os

import numpy as np
import matplotlib.pyplot as plt

import hw1.config as hw1cfg
from hw1.config import ExperimentResults


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

    exp_name = results.config.name
    for run_name, run_data in results.results_map.items():
        t_axis = np.arange(run_data.shape[1])
        mean = run_data[0]
        sterr = run_data[1]

        #ax.errorbar(t_axis, mean, yerr=sterr, label=run_name)
        ax.plot(t_axis, mean, label=run_name)
        ax.fill_between(t_axis, mean-sterr, mean+sterr, alpha=0.3)

    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$f(\mathbf{x_t})$')
    ax.set_title(exp_name)
    ax.set_yscale('log')
    ax.grid()
    ax.legend()

    filename = os.path.join(out_dir, f'{exp_name}.pdf')
    fig.savefig(filename, format='pdf')
