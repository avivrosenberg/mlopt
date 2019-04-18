import math
import sys
import collections

import numpy as np
import numpy.linalg as la
import tqdm

import hw1.data as hw1data
import hw1.optimizers as hw1opt


ExperimentConfig = collections.namedtuple(
    'ExperimentConfig',
    field_names=['name', 'n', 'd', 'smax', 'smin', 'sol_mu', 'sol_std',
                 'n_iter', 'n_repeat', 'fullrank'],
    defaults=[1024, 4, 5, 0.5, 100, 10, 100_000, 20, True]
)


def run_all_experiments():
    configurations = [
        ExperimentConfig(name='Default PD', fullrank=True),
        # ExperimentConfig(name='Default PSD', fullrank=False),
    ]

    cfg_plot_data = {}
    cfg_repeats = 20
    for cfg in configurations:
        # run_data will hold a matrix of run results, per optimizer
        run_data = {}
        for k in range(cfg_repeats):
            for opt_name, losses in single_experiment(cfg).items():
                opt_results = run_data.get(opt_name)
                if opt_results is None:
                    opt_results = np.empty((cfg_repeats, cfg.n_iter))
                    run_data[opt_name] = opt_results
                opt_results[k, :] = losses

        # Generate data for plotting: just mean and std err.
        plot_data = {}
        for opt_name, losses in run_data.items():
            means = np.mean(losses, axis=0)
            sterr = np.std(losses, axis=0) / math.sqrt(losses.shape[1])
            plot_data[opt_name] = np.array([means, sterr])

        cfg_plot_data[cfg.name] = plot_data

    return cfg_plot_data


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
        print(f'Running {name}')
        losses = single_run(opt, loss_fn, cfg.n_iter)
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
    run_all_experiments()
