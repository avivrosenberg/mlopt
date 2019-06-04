import math

import numpy as np
import numpy.linalg as la

import linreg.data as data
import linreg.optimizers as opt
from linreg.config import ExperimentConfig


def hw1_experiment(cfg: ExperimentConfig):
    """
    A single experiment means run each algorithm once with the same data
    :param cfg: Configuration parameters of this experiment.
    :returns: A map from a name of an optimization algorithm to a list of
    loss values, i.e. the difference between the function value at a current
    iterate and the function value at the solution point.
    """

    # Generate a single dataset all the optimizers will work with in this
    # experiment
    A, b, xs = data.generate_linear_regression(**cfg._asdict())
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
        opt.GradientDescent.optimal_stepsize_generator_nonsmooth(D, G)
    stepsize_smooth = \
        opt.GradientDescent.optimal_stepsize_generator_smooth(beta)
    stepsize_agm = opt.NesterovAGM.optimal_stepsize_generator()

    # Create optimizers for experiment
    optimizers = {
        'PGD Non-smooth':
            opt.GradientDescent(x0, stepsize_gen=stepsize_nonsmooth,
                                grad_fn=grad_fn, max_iter=cfg.n_iter),
        'PGD Smooth':
            opt.GradientDescent(x0, stepsize_gen=stepsize_smooth,
                                grad_fn=grad_fn, max_iter=cfg.n_iter),
        'AGM':
            opt.NesterovAGM(0, beta, x0, stepsize_gen=stepsize_agm,
                            grad_fn=grad_fn, max_iter=cfg.n_iter),
    }
    if alpha > 0:
        optimizers['AGM Strongly Convex'] = \
            opt.NesterovAGM(alpha, beta, x0, stepsize_gen=stepsize_agm,
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

