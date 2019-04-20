import os
import json
import pickle
import collections

from typing import List

ExperimentConfig = collections.namedtuple(
    'ExperimentConfig',
    ['name', 'n', 'd', 'smax', 'smin', 'sol_mu', 'sol_std',
     'n_iter', 'n_repeats'],
    defaults=[1024, 4, 5, 0.5, 100, 10, 100_00, 10]
)

ExperimentResults = collections.namedtuple(
    'ExperimentResults', ['config', 'results_map']
)

DEFAULT_CONFIGURATIONS = [
        # Positive definite with High, medium and low condition number
        ExperimentConfig(name='PD HC', smax=5, smin=0.1),
        ExperimentConfig(name='PD MC', smax=5, smin=0.5),
        ExperimentConfig(name='PD LC', smax=5, smin=1.0),
        # Positive semi-definite
        ExperimentConfig(name='PSD', smax=5, smin=0),
    ]

DEFAULTS_FILENAME = 'cfg/defaults.json'


def dump_configs(configs: List[ExperimentConfig], filename: str):
    """
    Writes configuration to file (json).
    :param configs: A list/tuple of ExperimentConfig objects.
    :param filename: Filename to write to.
    """
    config_dicts = [cfg._asdict() for cfg in configs]

    with open(filename, 'w') as file:
        json.dump(config_dicts, file, indent=4)


def load_configs(filename: str) -> List[ExperimentConfig]:
    """
    Reads a list of ExperimentConfigs from file.
    :param filename: The file to read from.
    """
    with open(filename, 'r') as file:
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


dump_configs(DEFAULT_CONFIGURATIONS,
             os.path.join(os.path.dirname(__file__), DEFAULTS_FILENAME))
