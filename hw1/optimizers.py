import numpy as np


class Optimizer(object):
    """
    Represents an optimization algorithm.
    """
    def __init__(self, x0, stepsize_gen, grad_fn, project_fn=None):
        """
        Initializes the algorithm.
        :param x0:
        :param stepsize_gen:
        :param grad_fn:
        :param project_fn:
        """
        self.xt = x0
        self.stepsize_gen = stepsize_gen
        self.grad_fn = grad_fn
        self.project_fn = project_fn

    def __next__(self):
        return self.step()

    def __iter__(self):
        return self

    def step(self):
        raise NotImplementedError()


class GradientDescent(Optimizer):
    def __init__(self, x0, stepsize_gen, grad_fn, project_fn=None):
        super().__init__(x0, stepsize_gen, grad_fn, project_fn)

    def step(self):
        eta = next(self.stepsize_gen)
        grad = self.grad_fn(self.xt)

        xnew = self.xt - eta * grad
        if self.project_fn is not None:
            xnew = self.project_fn(xnew)

        self.xt = xnew
        return xnew
