import math
import multiprocessing as mp
import sys

import numpy as np
import numpy.linalg as la
import tqdm

import hw1.config as hw1cfg
import hw1.data as hw1data
import hw1.optimizers as hw1opt
from hw1.config import ExperimentConfig, ExperimentResults


def run_configurations(configurations, parallel=False):
    # Run configuration on multiple processes
    if parallel:
        config_results = mp.Pool().map(run_single_configuration,
                                       configurations)
    else:
        config_results = [run_single_configuration(cfg) for cfg in
                          configurations]

    return config_results


def run_single_configuration(cfg: ExperimentConfig):
    # run_data will hold a matrix of run results, per optimizer
    run_data = {}
    for k in tqdm.tqdm(range(cfg.n_repeats), file=sys.stdout, desc=cfg.name):
        single_exp_results = run_single_experiment(cfg)
        for opt_name, losses in single_exp_results.items():
            opt_results = run_data.get(opt_name)
            if opt_results is None:
                opt_results = np.empty((cfg.n_repeats, cfg.n_iter))
                run_data[opt_name] = opt_results
            opt_results[k, :] = losses

    # Generate data for plotting: just mean and std err, per optimizer
    plot_data = {}
    for opt_name, losses in run_data.items():
        # losses = losses[:, ~np.all(np.isnan(losses), axis=0)]
        means = np.mean(losses, axis=0)
        sterr = np.std(losses, axis=0) / math.sqrt(losses.shape[0])
        plot_data[opt_name] = np.array([means, sterr])

    return ExperimentResults(config=cfg, results_map=plot_data)


def run_single_experiment(cfg: ExperimentConfig):
    """
    A single experiment means run each algorithm once with the same data
    """

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

    def stepsize_agm():
        eta = 1
        while True:
            yield eta
            eta = 0.5 * (-eta ** 2 + math.sqrt(eta ** 4 + 4 * eta ** 2))

    # Optimizers for experiment
    x0 = np.zeros(cfg.d)
    optimizers = {
        'PGD Non-smooth':
            hw1opt.GradientDescent(x0, stepsize_gen=stepsize_nonsmooth(),
                                   grad_fn=grad_fn, max_iter=cfg.n_iter),
        'PGD Smooth':
            hw1opt.GradientDescent(x0, stepsize_gen=stepsize_smooth(),
                                   grad_fn=grad_fn, max_iter=cfg.n_iter),
        'AGM':
            hw1opt.NesterovAGM(0, beta, x0, stepsize_gen=stepsize_agm(),
                               grad_fn=grad_fn, max_iter=cfg.n_iter),
    }
    if alpha > 0:
        optimizers['AGM Strongly Convex'] = \
            hw1opt.NesterovAGM(alpha, beta, x0, stepsize_gen=stepsize_agm(),
                               grad_fn=grad_fn, max_iter=cfg.n_iter)

    loss_x0 = loss_fn(x0)
    loss_xs = loss_fn(xs)
    results = {}

    for name, optimizer in optimizers.items():
        losses = np.full(cfg.n_iter + 1, np.nan)
        losses[0] = loss_x0

        # Run single optimizer
        for t, xt in enumerate(optimizer, start=1):
            loss_xt = loss_fn(xt)
            losses[t] = math.fabs(loss_xt - loss_xs)
            if losses[t] < cfg.eps:
                break

        results[name] = losses[0:-1]

    return results


if __name__ == '__main__':
    results = run_configurations(hw1cfg.DEFAULT_CONFIGURATIONS)
