import numpy as np
import scipy.sparse.linalg


class MetricInducedSimplexProjection(object):
    """
    Projects points onto a unit simplex, but uses a metric induced by a
    given matrix instead of a regular euclidean norm.

    Solves the optimization problem:
        arg min_{z in S} (z-x)^T A (z-x)
    where x is the point to project and A is a positive definite matrix.

    This implementation uses the conditional-gradient (Frank-Wolfe) method to
    solve the optimization problem.
    """

    def __init__(self, A: np.ndarray = None, eta_min=0.):
        """
        :param A: The metric-inducing matrix. Should be positive definite.
        If set to None, an identity matrix will be used which corresponds to
        using a regular euclidean norm for the projection.
        :param eta_min: Stop optimization if step size is smaller than this.
        """
        self.A = A
        self.eta_min = eta_min

    def __call__(self, y, **kw):
        d, = y.shape

        # The "extreme points" of the unit simplex are the standard basis
        # vectors in d-dimensions
        I = np.eye(d, dtype=np.float32)
        A = I if self.A is None else self.A

        pt = I[0]
        for t in range(1, d + 1):
            eta = 2 / (1 + t)
            if eta < self.eta_min:
                break

            # Gradient of the function we're optimizing
            gt = 2 * np.dot(A, pt - y)

            # Solve arg min_{v in V(S)} <v, gt>, where V(S) are the extreme
            # points of the Simplex and <.,.> is an inner product.
            # Solving this is equivalent to taking the standard basis vector
            # which selects the minimal element from gt.
            imin = np.argmin(gt)
            vt = I[imin]

            pt = pt + eta * (vt - pt)

        return pt


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

    def projection_simplex_sort(self, v):
        n_features = v.shape[0]
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u) - self.z
        ind = np.arange(n_features) + 1
        cond = u - cssv / ind > 0
        rho = ind[cond][-1]
        theta = cssv[cond][-1] / float(rho)
        w = np.maximum(v - theta, 0)
        return w

    def projection_simplex_pivot(self, v, random_state=None):
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
            if s + ds - (rho + drho) * v[k] < self.z:
                s += ds
                rho += drho
                U = L
            else:
                U = G
        theta = (s - self.z) / float(rho)
        return np.maximum(v - theta, 0)

    def projection_simplex_bisection(self, v, tau=0.0001, max_iter=1000):
        lower = 0
        upper = np.max(v)
        current = np.inf

        for it in range(max_iter):
            if np.abs(current) / self.z < tau and current < 0:
                break

            theta = (upper + lower) / 2.0
            w = np.maximum(v - theta, 0)
            current = np.sum(w) - self.z
            if current <= 0:
                upper = theta
            else:
                lower = theta
        return w


class NuclearNormProjection(object):
    """
    Projects a matrix on to the nuclear norm ball with radius tau.
    """

    def __init__(self, tau, rank=None, maxiter=None, always_project=False):
        """
        :param tau: Maximal nuclear norm
        :param rank: Maximal rank of SVD for projection
        :param maxiter: Maximal iterations for SVD computation
        :param always_project: Whether to project a matrix onto the nuclear
        norm ball if it's nuclear norm is less than tau.
        """
        self.tau = tau
        self.rank = rank
        self.maxiter = maxiter
        self.simplex_proj = SimplexProjection(method='sort', z=tau)
        self.always_project = always_project

    def __call__(self, X):
        rank = min(X.shape) - 1 if self.rank is None else self.rank

        U, s, Vt = scipy.sparse.linalg.svds(
            X, k=rank, maxiter=self.maxiter, which='LM',
            return_singular_vectors=True,
        )

        # If the nuclear norm is within the desired radius, no need to project
        # unless 'always_project' is true.
        if not self.always_project and np.sum(s) <= self.tau and np.all(s >= 0):
            return X

        # Project s onto tau-scaled simplex
        sproj = self.simplex_proj(s)

        # Reconstruct with reverse SVD
        Xproj = np.dot(U, np.dot(np.diag(sproj), Vt))

        return Xproj
