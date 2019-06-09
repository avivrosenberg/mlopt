import math
from typing import Dict

import numpy as np
import numpy.linalg as la

import linreg.data as data
import linreg.run as run

import optim.optimizers as opt
import optim.stepsize_gen as stepsize_gen

from linreg.config import ExperimentConfig


class Runner(run.ExperimentRunner):

    def run_experiment(self) -> Dict[str, np.ndarray]:
        cfg: ExperimentConfig = self.cfg

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

        # Gradient Stochastic Oracle for SGD and SVRG
        def grad_fn(x, nbatch=None, seed=None):
            if seed is not None:
                np.random.seed(seed)
            i_full = range(A.shape[0])
            if nbatch is not None:
                i = np.random.choice(i_full, size=nbatch, replace=False)
            else:
                i = i_full
            return A[i].T.dot(A[i].dot(x) - b[i])

        # Create optimizers for experiment
        sgd_batch_sizes = [1, 4, 16, 64]
        optimizers = {}
        for batch_size in sgd_batch_sizes:
            optimizers[f'SGD{batch_size}'] = opt.GradientDescent(
                x0,
                stepsize_gen=stepsize_gen.sgd_sc(alpha),
                steps_per_iter=int(self.cfg.n / batch_size),
                grad_fn=lambda x: grad_fn(x, batch_size),
                max_iter=cfg.n_iter
            )
        optimizers['SVRG'] = opt.SVRG(
            x0,
            # not sure why smaller stepsize is needed...
            stepsize_gen=stepsize_gen.const(1/(10*beta)/10),
            steps_per_iter=int(20*beta/alpha),
            grad_fn=grad_fn,
            max_iter=cfg.n_iter
        )

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
