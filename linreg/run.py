import abc
import math
import multiprocessing as mp
import sys

import numpy as np
import tqdm

from util.util import import_name
from linreg.config import ExperimentConfig, ExperimentResults
from typing import Dict


class ExperimentRunner(abc.ABC):
    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg

    @abc.abstractmethod
    def run_experiment(self) -> Dict[str, np.ndarray]:
        """
        An experiment means run one or more algorithms once with the
        same data.
        :returns: A map from a name of an optimization algorithm to an ndarray
        of loss values, i.e. the difference between the function value at a
        current iterate and the function value at the solution point.
        """
        pass


def run_configurations(configurations, parallel=False):
    """
    Runs a multiple experiments based on a list of configurations.
    :param configurations: list of configurations to run.
    :param parallel: Whether to run each configuration concurrently in a
    separate process.
    :return: A list of ExperimentResults per configuration.
    """
    if parallel:
        config_results = \
            mp.Pool().map(run_single_configuration, configurations)
    else:
        config_results = \
            [run_single_configuration(cfg) for cfg in configurations]

    return config_results


def run_single_configuration(cfg: ExperimentConfig):
    # Get the runner for this configuration
    try:
        runner_class = import_name(cfg.runner)
        runner = runner_class(cfg)
    except BaseException as e:
        raise ValueError(f'Please specify a valid runner in your '
                         f'ExperimentConfig. Got {cfg.runner}.')

    # run_data will hold a matrix of run results, per optimizer
    run_data = {}
    for k in tqdm.tqdm(range(cfg.n_repeats), file=sys.stdout, desc=cfg.name):
        single_exp_results = runner.run_experiment()
        for opt_name, losses in single_exp_results.items():
            opt_results = run_data.get(opt_name)
            if opt_results is None:
                opt_results = np.empty((cfg.n_repeats, cfg.n_iter))
                run_data[opt_name] = opt_results
            opt_results[k, :] = losses

    # Generate data for plotting: just mean and std err, per optimizer
    plot_data = {}
    for opt_name, losses in run_data.items():
        # Remove columns where all values are NaN (means no experiment
        # repeat got to that iteration)
        losses = losses[:, ~np.all(np.isnan(losses), axis=0)]
        means = np.nanmean(losses, axis=0)
        sterr = np.nanstd(losses, axis=0) / math.sqrt(losses.shape[0])
        plot_data[opt_name] = np.array([means, sterr])

    return ExperimentResults(config=cfg, results_map=plot_data)

