import math


class Optimizer(object):
    """
    Represents an optimization algorithm.
    """
    def __init__(self, x0, stepsize_gen, grad_fn,
                 max_iter=math.inf, project_fn=None):
        """
        Initializes the algorithm.
        :param x0:
        :param stepsize_gen:
        :param grad_fn:
        :param max_iter:
        :param project_fn:
        """
        self.xt = x0
        self.stepsize_gen = stepsize_gen
        self.grad_fn = grad_fn
        self.max_iter = max_iter
        self.project_fn = project_fn

        self.t = 0

    def __next__(self):
        self.xt = self.step()

        self.t += 1
        if self.t > self.max_iter:
            raise StopIteration()

        return self.xt

    def __iter__(self):
        return self

    def step(self):
        raise NotImplementedError()


class GradientDescent(Optimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def step(self):
        eta = next(self.stepsize_gen)
        grad = self.grad_fn(self.xt)

        xnew = self.xt - eta * grad
        if self.project_fn is not None:
            xnew = self.project_fn(xnew)

        return xnew


class NesterovAGM(Optimizer):
    def __init__(self, beta, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.yt = self.xt

    def step(self):
        eta = next(self.stepsize_gen)

        ztp1 = (1-eta) * self.yt + eta * self.xt
        grad = self.grad_fn(ztp1)

        xtp1 = self.xt - 1.0/(self.beta * eta) * grad
        if self.project_fn is not None:
            xtp1 = self.project_fn(xtp1)

        ytp1 = (1-eta) * self.yt + eta * xtp1

        self.xt = xtp1
        self.yt = ytp1
        return xtp1



