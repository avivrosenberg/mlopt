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
        # Remove columns where all values are NaN (means no experiment
        # repeat got to that iteration)
        losses = losses[:, ~np.all(np.isnan(losses), axis=0)]
        means = np.nanmean(losses, axis=0)
        sterr = np.nanstd(losses, axis=0) / math.sqrt(losses.shape[0])
        plot_data[opt_name] = np.array([means, sterr])

    return ExperimentResults(config=cfg, results_map=plot_data)


def run_single_experiment(cfg: ExperimentConfig):
    """
    A single experiment means run each algorithm once with the same data
    :param cfg: Configuration parameters of this experiment.
    :returns: A map from a name of an optimization algorithm to a list of
    loss values, i.e. the difference between the function value at a current
    iterate and the function value at the solution point.
    """

    # Generate a single dataset all the optimizers will work with in this
    # experiment
    A, b, xs = hw1data.generate_linear_regression(**cfg._asdict())
    x0 = np.zeros(cfg.d)

    # Calculate problem parameters based on the data configuration
    alpha = cfg.smin ** 2
    beta = cfg.smax ** 2
    xs_norm = cfg.sol_mu * math.sqrt(cfg.d)
    b_norm = cfg.smax * xs_norm
    f_xs = 0
    f_x1 = 0.5 * (b_norm ** 2)
    R = xs_norm
    D = 2 * R
    G = (cfg.smax ** 2) * R + cfg.smax * b_norm

    # Loss function for all optimizers (what we minimize)
    def loss_fn(x):
        return 0.5 * la.norm(A.dot(x) - b) ** 2

    # Gradient "Oracle" for all optimizers
    def grad_fn(x, k=1):
        return A.T.dot(A.dot(x) - b)

    # Step size generators, per optimizer
    stepsize_nonsmooth = \
        hw1opt.GradientDescent.optimal_stepsize_generator_nonsmooth(D, G)
    stepsize_smooth = \
        hw1opt.GradientDescent.optimal_stepsize_generator_smooth(beta)
    stepsize_agm = hw1opt.NesterovAGM.optimal_stepsize_generator()

    # Create optimizers for experiment
    optimizers = {
        'PGD Non-smooth':
            hw1opt.GradientDescent(x0, stepsize_gen=stepsize_nonsmooth,
                                   grad_fn=grad_fn, max_iter=cfg.n_iter),
        'PGD Smooth':
            hw1opt.GradientDescent(x0, stepsize_gen=stepsize_smooth,
                                   grad_fn=grad_fn, max_iter=cfg.n_iter),
        'AGM':
            hw1opt.NesterovAGM(0, beta, x0, stepsize_gen=stepsize_agm,
                               grad_fn=grad_fn, max_iter=cfg.n_iter),
    }
    if alpha > 0:
        optimizers['AGM Strongly Convex'] = \
            hw1opt.NesterovAGM(alpha, beta, x0, stepsize_gen=stepsize_agm,
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
