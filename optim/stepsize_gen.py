"""
Generator functions for optimal step-sizes of various optimization algorithms
under different conditions.
"""
import math


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
        eta = 0.5 * (-eta**2 + math.sqrt(eta**4 + 4*eta**2))
