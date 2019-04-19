import math
import sys
import collections
import multiprocessing as mp

import numpy as np
import numpy.linalg as la
import tqdm

import hw1.data as hw1data
import hw1.optimizers as hw1opt

ExperimentConfig = collections.namedtuple(
    'ExperimentConfig',
    ['name', 'n', 'd', 'smax', 'smin', 'sol_mu', 'sol_std',
     'n_iter', 'n_repeats'],
    defaults=[1024, 4, 5, 0.5, 100, 10, 100_0, 2]
)

ExperimentResults = collections.namedtuple(
    'ExperimentResults', ['config', 'results_map']
)


def run_all_experiments():
    configurations = [
        # Positive definite with High, medium and low condition number
        ExperimentConfig(name='PD HC', smax=5, smin=0.1),
        ExperimentConfig(name='PD MC', smax=5, smin=0.5),
        ExperimentConfig(name='PD LC', smax=5, smin=1.0),
        # Positive semi-definite
        ExperimentConfig(name='PSD', smax=5, smin=0),
    ]

    # Run configuration on multiple processes
    mppool = mp.Pool()
    config_results = mppool.map(run_configuration, configurations)

    return config_results


def run_configuration(cfg: ExperimentConfig):
    # run_data will hold a matrix of run results, per optimizer
    run_data = {}
    for k in range(cfg.n_repeats):
        single_exp_results = single_experiment(cfg)
        for opt_name, losses in single_exp_results.items():
            opt_results = run_data.get(opt_name)
            if opt_results is None:
                opt_results = np.empty((cfg.n_repeats, cfg.n_iter))
                run_data[opt_name] = opt_results
            opt_results[k, :] = losses

    # Generate data for plotting: just mean and std err, per optimizer
    plot_data = {}
    for opt_name, losses in run_data.items():
        means = np.mean(losses, axis=0)
        sterr = np.std(losses, axis=0) / math.sqrt(losses.shape[1])
        plot_data[opt_name] = np.array([means, sterr])

    return ExperimentResults(config=cfg, results_map=plot_data)


def single_experiment(cfg: ExperimentConfig):
    """
    A single experiment means run each algorithm once with the same data
    """
    # Generate data for the experiment
    # n, d, smax, smin = 1024, 3, 5, 0.5
    # sol_mu, sol_std = 100, 10

    A, b, xs = hw1data.generate_linear_regression(**cfg._asdict())
    alpha = cfg.smin ** 2
    beta = cfg.smax ** 2
    R = cfg.sol_mu + 3 * cfg.sol_std
    G = (cfg.smax ** 2) * R + cfg.smax * la.norm(b)
    D = 2 * R

    # Loss function for all optimizers (what we minimize)
    def loss_fn(x):
        return 0.5 * la.norm(A.dot(x) - b) ** 2

    # Gradient "Oracle" for all optimizers
    def grad_fn(x, k=1):
        return A.T.dot(A.dot(x) - b)

    # Step size generators, per optimizer
    def stepsize_nonsmooth():
        t = 0
        while True:
            t += 1
            yield D / G / math.sqrt(t)

    def stepsize_smooth():
        while True:
            yield 1 / beta

    # Optimizers
    x0 = np.zeros(cfg.d)
    optimizers = {
        'PGD Non-smooth':
            hw1opt.GradientDescent(x0, stepsize_nonsmooth(), grad_fn),
        'PGD Smooth':
            hw1opt.GradientDescent(x0, stepsize_smooth(), grad_fn),
    }

    results = {}
    for name, opt in optimizers.items():
        losses = np.zeros(cfg.n_iter)

        for t in tqdm.tqdm(range(cfg.n_iter), file=sys.stdout):
            xt = next(opt)
            losses[t] = loss_fn(xt)

        results[name] = losses

    return results


def single_run(optimizer, loss_fn, n_iter=100_000):
    losses = np.zeros(n_iter)

    for t in tqdm.tqdm(range(n_iter), file=sys.stdout):
        xt = next(optimizer)
        losses[t] = loss_fn(xt)

    print(f'final loss={losses[-1]:.5f}')
    return losses


if __name__ == '__main__':
    results = run_all_experiments()
    print(results)
