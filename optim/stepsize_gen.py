"""
Generator functions for optimal step-sizes of various optimization algorithms
under different conditions.
"""
import math
from typing import Callable, Iterable


def pgd_nonsmooth(D, G):
    """
    :param D: Diameter of solution set.
    :param G: Upper bound of gradient.
    :return: A generator for the optimal stepsize of projected
    gradient descent (PGD) for non-smooth optimization.
    """
    assert D > 0. and G > 0.
    t = 0
    while True:
        t += 1
        yield D / G / math.sqrt(t)


def pgd_smooth(beta):
    """
    :param beta: Smoothness coefficient.
    :return: A generator for the optimal stepsize of PGD for beta-smooth
    functions.
    """
    assert beta > 0.
    while True:
        yield 1 / beta


def nesterov_agm():
    """
    :return: A generator for the optimal step size of Nesterov's accelerated
    gradient method (AGM).
    """
    eta = 1
    while True:
        yield eta
        eta = 0.5 * (-eta ** 2 + math.sqrt(eta ** 4 + 4 * eta ** 2))


def sgd_sc(alpha):
    """
    :param alpha: Strong-convexity coefficient.
    :return: A generator for the optimal step-size of stochastic
    gradient descent (SGD) for an alpha-strongly-convex function.
    """
    assert alpha > 0.
    t = 0
    while True:
        t += 1
        yield 2. / (alpha * (t + 1))


def const(eta):
    assert eta > 0.
    while True:
        yield eta


def custom(fn: Callable, idx_range: Iterable = None):
    if not idx_range:
        idx_range = range(2 ** 63 - 1)

    for idx in idx_range:
        yield fn(idx)


def cond_grad():
    """
    :return: A generator for the optimal step-size of the conditional gradient
    method for matrix completion.
    """
    t = 0
    while True:
        t += 1
        yield 2. / (t + 1)
