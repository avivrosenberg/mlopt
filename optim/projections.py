import numpy as np


class SimplexProjection(object):
    """
    Projects onto Simplex.
    Callable, so can be used as a projection function.

    Code taken from:
        Mathieu Blondel, Akinori Fujino, and Naonori Ueda.
        ICPR 2014.
        http://www.mblondel.org/publications/mblondel-icpr2014.pdf
    """
    def __init__(self, method='sort', z=1):
        """
        :param method: must be either 'sort', 'pivot' or 'bisection'.
        See methods in paper.
        :param z: Projection sum (i.e. the sum of the elements in
        the projected vector), default (z=1) is the probabilistic simplex.
        """
        assert z > 0
        assert method in ('sort', 'pivot', 'bisection')

        self.z = z
        self.method = method

    def __call__(self, x, **kwargs):
        proj_fn = self.__getattribute__(f'projection_simplex_{self.method}')
        return proj_fn(x, **kwargs)

    @staticmethod
    def projection_simplex_sort(v, z=1):
        n_features = v.shape[0]
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u) - z
        ind = np.arange(n_features) + 1
        cond = u - cssv / ind > 0
        rho = ind[cond][-1]
        theta = cssv[cond][-1] / float(rho)
        w = np.maximum(v - theta, 0)
        return w

    @staticmethod
    def projection_simplex_pivot(v, z=1, random_state=None):
        rs = np.random.RandomState(random_state)
        n_features = len(v)
        U = np.arange(n_features)
        s = 0
        rho = 0
        while len(U) > 0:
            G = []
            L = []
            k = U[rs.randint(0, len(U))]
            ds = v[k]
            for j in U:
                if v[j] >= v[k]:
                    if j != k:
                        ds += v[j]
                        G.append(j)
                elif v[j] < v[k]:
                    L.append(j)
            drho = len(G) + 1
            if s + ds - (rho + drho) * v[k] < z:
                s += ds
                rho += drho
                U = L
            else:
                U = G
        theta = (s - z) / float(rho)
        return np.maximum(v - theta, 0)

    @staticmethod
    def projection_simplex_bisection(v, z=1, tau=0.0001, max_iter=1000):
        lower = 0
        upper = np.max(v)
        current = np.inf

        for it in range(max_iter):
            if np.abs(current) / z < tau and current < 0:
                break

            theta = (upper + lower) / 2.0
            w = np.maximum(v - theta, 0)
            current = np.sum(w) - z
            if current <= 0:
                upper = theta
            else:
                lower = theta
        return w
