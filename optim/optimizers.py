import math

import optim.stepsize_gen as stepsize_gen


class Optimizer(object):
    """
    Represents an optimization algorithm.
    """

    def __init__(self, x0, stepsize_gen, grad_fn,
                 max_iter=math.inf, steps_per_iter=1, project_fn=None):
        """
        Initializes the algorithm.
        :param x0: Starting point.
        :param stepsize_gen: A generator returning a new step size each time.
        :param grad_fn: A function that given a point, computes the gradient of
        the minimization target at that point.
        :param max_iter: Max number of iterations to run when iterating over
        this instance.
        :param steps_per_iter: Number of optimization steps, i.e. calls to
        step(), to run per iteration when iterating over this instance.
        :param project_fn: A function that given a point x, projects it onto
        some set, returning a new point xp.
        """
        self.xt = x0
        self.stepsize_gen = stepsize_gen
        self.grad_fn = grad_fn
        self.max_iter = max_iter
        self.steps_per_iter = steps_per_iter
        self.project_fn = (lambda x: x) if project_fn is None else project_fn

        self.current_iter = 0

    def __next__(self):
        for k in range(self.steps_per_iter):
            self.xt = self.step()

        self.current_iter += 1
        if self.current_iter >= self.max_iter:
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
        xnew = self.project_fn(xnew)

        return xnew


class NesterovAGM(Optimizer):
    def __init__(self, alpha=0., beta=1., *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert 0 <= alpha <= beta

        self.alpha = alpha
        self.beta = beta
        self.yt = self.xt

    def step(self):
        if self.alpha > 0:
            return self.step_strongly_convex()

        return self.step_non_strongly_convex()

    def step_strongly_convex(self):
        sub_max_iter = math.ceil(math.sqrt(128 * self.beta / 9 / self.alpha))
        sub_stepsize_gen = stepsize_gen.nesterov_agm()

        sub_opt = NesterovAGM(alpha=0, beta=self.beta,
                              x0=self.xt,
                              stepsize_gen=sub_stepsize_gen,
                              grad_fn=self.grad_fn,
                              project_fn=self.project_fn,
                              max_iter=sub_max_iter)

        for xtp1 in sub_opt:
            pass

        xtp1 = self.project_fn(xtp1)

        self.xt = xtp1
        return xtp1

    def step_non_strongly_convex(self):
        eta = next(self.stepsize_gen)

        ztp1 = (1 - eta) * self.yt + eta * self.xt
        grad = self.grad_fn(ztp1)

        xtp1 = self.xt - 1.0 / (self.beta * eta) * grad
        xtp1 = self.project_fn(xtp1)

        ytp1 = (1 - eta) * self.yt + eta * xtp1

        self.xt = xtp1
        self.yt = ytp1
        return xtp1