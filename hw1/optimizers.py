import math


class Optimizer(object):
    """
    Represents an optimization algorithm.
    """

    def __init__(self, x0, stepsize_gen, grad_fn,
                 max_iter=math.inf, project_fn=None):
        """
        Initializes the algorithm.
        :param x0: Starting point.
        :param stepsize_gen: A generator returning a new step size each time.
        :param grad_fn: A function that given a point, computes the gradient of
        the minimization target at that point.
        :param max_iter: Max number of iterations to run when iterating over
        this instance.
        :param project_fn: A function that given a point x, projects it onto
        some set, returning a new point xp.
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
        if self.t >= self.max_iter:
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

    @staticmethod
    def optimal_stepsize_generator_nonsmooth(D, G):
        """
        :param D: Diameter of solution set.
        :param G: Upper bound of gradient.
        :return: A generator for the optimal stepsize of nonsmooth PGD.
        """
        t = 0
        while True:
            t += 1
            yield D / G / math.sqrt(t)

    @staticmethod
    def optimal_stepsize_generator_smooth(beta):
        """
        :param beta: Smoothness coefficient.
        :return: A generator for the optimal stepsize of smooth PGD.
        """
        while True:
            yield 1 / beta


class NesterovAGM(Optimizer):
    def __init__(self, alpha=0., beta=1., *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert 0 <= alpha <= beta

        self.alpha = alpha
        self.beta = beta
        self.yt = self.xt

        if alpha > 0:
            self.step = self.step_strongly_convex
        else:
            self.step = self.step_non_strongly_convex

    def step_strongly_convex(self):
        sub_max_iter = math.ceil(math.sqrt(128 * self.beta / 9 / self.alpha))
        sub_stepsize_gen = NesterovAGM.optimal_stepsize_generator()

        sub_opt = NesterovAGM(alpha=0, beta=self.beta,
                              x0=self.xt,
                              stepsize_gen=sub_stepsize_gen,
                              grad_fn=self.grad_fn,
                              project_fn=self.project_fn,
                              max_iter=sub_max_iter)

        for xtp1 in sub_opt:
            pass

        if self.project_fn is not None:
            xtp1 = self.project_fn(xtp1)

        self.xt = xtp1
        return xtp1

    def step_non_strongly_convex(self):
        eta = next(self.stepsize_gen)

        ztp1 = (1 - eta) * self.yt + eta * self.xt
        grad = self.grad_fn(ztp1)

        xtp1 = self.xt - 1.0 / (self.beta * eta) * grad
        if self.project_fn is not None:
            xtp1 = self.project_fn(xtp1)

        ytp1 = (1 - eta) * self.yt + eta * xtp1

        self.xt = xtp1
        self.yt = ytp1
        return xtp1

    @staticmethod
    def optimal_stepsize_generator():
        """
        :return: A generator for the optimal step size of AGM
        """
        eta = 1
        while True:
            yield eta
            eta = 0.5 * (-eta**2 + math.sqrt(eta**4 + 4*eta**2))
