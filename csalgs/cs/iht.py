# encoding: utf-8
"""
[1] T. Blumensath and M. E. Davies, Normalized Iterative Hard Thresholding:
    Guaranteed Stability and Performance, IEEE Journal of Selected Topics in
    Signal Processing, vol. 4, no. 2, pp. 298--309, Apr. 2010.
"""
import numpy as np
from numpy.linalg import norm


__all__ = ['constant_stepsize', 'adaptive_stepsize', 'iht_estimator']


def _hard_threshold(x, nnz, retsupp=False):
    """Computes the best approximation of `x` with at most `nnz` nonzero
    entries

    :param x: @todo
    :param nnz: @todo
    :param retsupp: @todo
    :returns: @todo

    """
    argpart = np.argpartition(np.abs(x), -nnz)
    supp, rest = argpart[-nnz:], argpart[:-nnz]
    x_new = x.copy()
    x_new[rest] = 0
    return (x_new, supp) if retsupp else x_new


def constant_stepsize(mu):
    """Returns a stepsize function with constant stepsize

    :param mu: @todo
    :returns: @todo

    """
    return (lambda *args, **kwargs: mu)


def _same_supports(supp1, supp2):
    """@todo: Docstring for _same_supports.

    :param supp1: @todo
    :param supp2: @todo
    :returns: @todo

    """
    return np.all(np.sort(supp1) == np.sort(supp2))


def adaptive_stepsize(rescale_const=.5, kappa=3.):
    """@todo: Docstring for adaptive_stepsize.

    :param rescale_const: @todo
    :param kappa: @todo
    :returns: @todo

    """
    assert kappa * rescale_const > 1

    def accepting_threshold(x_new, x_old, A):
        diff = x_new - x_old
        assert 1e50 > norm(diff) > 1e-10
        return (1 - rescale_const) * norm(diff)**2 / norm(A @ diff)**2

    def stepsize(x, A, g, supp):
        mu = norm(g[supp])**2 / norm(A[:, supp] @ g[supp])**2
        nnz = len(supp)

        while True:
            x_new, supp_new = _hard_threshold(x + mu * g, nnz, retsupp=True)
            omega = accepting_threshold(x_new, x, A)
            if _same_supports(supp, supp_new) or (mu < omega):
                return mu
            mu /= kappa * rescale_const

    return stepsize


def iht_estimator(A, y, nnz, stepsize=adaptive_stepsize(), x_init=None):
    """@todo: Docstring for csIHT.

    :param A: Sensing matrix
    :param y: Observation
    :param nnz: Number of non-zero elements
    :param stepsize: Stepsize function
    :param rgen:
    :param x_init: @todo
    :returns: @todo

    """
    x_hat = x_init if x_init is not None else np.zeros(A.shape[1])
    _, supp = _hard_threshold(A.T @ y, nnz, retsupp=True)
    while True:
        g = A.T @ (y - A @ x_hat)
        mu = stepsize(x_hat, A, g, supp)
        x_hat, supp = _hard_threshold(x_hat + mu * g, nnz, retsupp=True)
        yield x_hat
