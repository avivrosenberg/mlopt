import math
import os
import json
import pickle
import collections

from typing import List

EXPERIMENT_PARAMS = dict(
    name='Configuration name',
    runner='Fully qualified name of python class that will '
           'run a single experiment with this configuration. '
           'The class must extend linreg.run.ExperimentRunner.',
    n='Number of observations in dataset (rows of A)',
    d='Number of features per observation (columns of A)',
    smax='Largest singular value', smin='Smallest singular value',
    sol_mu='Mean of normal distribution from which x* is sampled',
    sol_std='St.dev. of normal distribution from which x* is sampled',
    n_iter='Maximal number of iterations (optimizer steps) to run without '
           'convergence',
    eps='Difference between current and ideal loss considered as convergence',
    n_repeats='Times to repeat the experiment with newly generated data'
)

ExperimentConfig = collections.namedtuple(
    'ExperimentConfig',
    EXPERIMENT_PARAMS.keys(),
    defaults=(None, 2**11, 2**5, 5, 0.5, 100, 0.001, 100000, 0.01, 10),
)

ExperimentResults = collections.namedtuple(
    'ExperimentResults', ['config', 'results_map']
)

DEFAULT_CONFIGURATIONS = [
    # Positive definite with high and low condition number
    ExperimentConfig(name='PD HC', smax=8, smin=1),
    ExperimentConfig(name='PD LC', smax=4, smin=1),
    # Positive semi-definite with high and low maximal singular value
    ExperimentConfig(name='PSD HS', smax=8, smin=0),
    ExperimentConfig(name='PSD LS', smax=4, smin=0),
]

DEFAULTS_FILENAME = 'cfg/defaults.json'


def dump_configs(configs: List[ExperimentConfig], filename: str):
    """
    Writes configuration to file (json).
    :param configs: A list/tuple of ExperimentConfig objects.
    :param filename: Filename to write to.
    """
    config_dicts = [cfg._asdict() for cfg in configs]

    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(config_dicts, file, indent=4)


def load_configs(filename: str) -> List[ExperimentConfig]:
    """
    Reads a list of ExperimentConfigs from file.
    :param filename: The file to read from.
    """
    with open(filename, 'r', encoding='utf-8') as file:
        config_dicts = json.load(file)
        return [ExperimentConfig(**d) for d in config_dicts]


def dump_results(results: List[ExperimentResults], filename: str):
    """
    Store results to a file (binary).
    :param results: List/tuple of ExperimentResults.
    :param filename: Path of filename to write.
    """
    with open(filename, 'wb') as file:
        pickle.dump(results, file)


def load_results(filename: str) -> List[ExperimentResults]:
    """
    Load ExperimentResults from a file.
    :param filename: Path to file.
    :return: The results loaded from the file.
    """
    with open(filename, 'rb') as file:
        return pickle.load(file)


def calc_problem_params(cfg: ExperimentConfig):
    alpha = cfg.smin ** 2
    beta = cfg.smax ** 2

    xs_norm = cfg.sol_mu * math.sqrt(cfg.d)
    b_norm = cfg.smax * xs_norm

    f_xs = 0
    f_x1 = 0.5 * (b_norm ** 2)

    R = xs_norm
    D = 2 * R
    G = (cfg.smax ** 2) * R + cfg.smax * b_norm

    PGD_nonsmooth = D**2 * G**2 / cfg.eps**2
    PGD_smooth_PSD = beta * xs_norm**2 / cfg.eps
    PGD_smooth_PD =  beta/alpha * math.log(f_x1/cfg.eps)
    AGM = math.sqrt(beta * xs_norm**2 / cfg.eps)
    AGM_SC = math.sqrt(beta/alpha) * math.log(xs_norm / alpha / cfg.eps)

    return locals()


dump_configs(DEFAULT_CONFIGURATIONS,
             os.path.join(os.path.dirname(__file__), DEFAULTS_FILENAME))
