# mlopt

This repo contains implementations of various optimization algorithms for
machine-learning applications which were implemented as part of a course about
optimization.


## General optimization algorithms

The `optim/optimizers.py` module provides implementations of stochastic gradient descent (SGD),
Nesterov accelerated gradient (AGM), and stochastic variance-reduced gradient
(SVRG).


## Projection algorithms

The `optim/projections.py` module provides algorithms for projection of points
onto the (scaled) Simplex (using either a regular euclidean norm or a custom
norm induced by a positive definite matrix) and also an algorithm for projecting
matrices onto the set of matrices with a bounded nuclear-norm.

## Linear regression

The `linreg` package contains functions to generate linear-regression datasets,
run experiments using the above optimization methods and plot results.

## Matrix completion

The `matcomp/data.py` module contains functions to create datasets for the
matrix-completion tasks, including synthetic data and data from the MovieLens
100k/1M datasets.

The `matcomp/models.py` module implements a few algorithms for solving the
matrix completion task. For example, optimizing over a projection onto the space of low-rank
matrices, factorizing into a product of two low-rank matrices and optimizing
over them, and optimizing over over a projection onto a bounded nuclear-norm.


## Online rebalancing portfolio selection (ORPS)

The `orps/data.py` module contains functions to create a small dataset for the ORPS
task.

The `orps/models.py` module contains implementation of multiple algorithms to
solve the ORPS problem, as well as calculation of the regret and wealth per trading round.
The implemented algorithms are online gradient descent (OGD), regularized
follow-the-leader (RFTL), Newton-step, best-fixed portfolio and best
single-asset portfolio.
