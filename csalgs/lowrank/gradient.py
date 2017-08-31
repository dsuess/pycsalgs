# encoding: utf-8

import itertools as it

import numpy as np
import numpy.linalg as la
from scipy.sparse.linalg import svds

__all__ = ['adaptive_stepsize', 'iht_estimator', 'cgm_estimator']


def _vec(A):
    newshape = A.shape[:-2]
    newshape = newshape + (A.shape[-2] * A.shape[-1],)
    return A.reshape(newshape)


def hard_threshold(mat, rank, retproj=False):
    """PU, PV ... projectors on left/right eigenspaces"""
    U_full, s, Vstar_full = la.svd(mat)
    U = U_full[:, :rank]
    V = Vstar_full.T.conj()[:, :rank]
    PU = U @ U.T.conj()
    PV = V @ V.T.conj()
    mat_projected = U @ np.diag(s[:rank]) @ V.conj().T
    return (mat_projected, (PU, PV)) if retproj else mat_projected


def adaptive_stepsize(projection='row'):
    """@todo: Docstring for adaptive_stepsize.

    :param projection: Possible values: 'row', 'col', 'rowcol', None
    :returns: @todo

    """
    assert projection in {'col', 'row', 'rowcol', None}

    def stepsize(A, g, projectors):
        PU, PV = projectors

        if projection == 'col':
            g = PU @ g
        elif projection == 'row':
            g = g @ PV
        elif projection == 'rowcol':
            g = PU @ g @ PV

        return la.norm(g)**2 / la.norm(_vec(A) @ _vec(g))**2

    return stepsize


def iht_estimator(A, y, rank, stepsize=adaptive_stepsize(), x_init=None):
    x_hat = np.zeros(A.shape[1:]) if x_init is None else x_init
    _, projectors = hard_threshold(np.tensordot(y, A, axes=(-1, 0)), rank,
                                   retproj=True)

    while True:
        g = np.tensordot(y - (_vec(A) @ _vec(x_hat)), A, axes=(-1, 0))
        mu = stepsize(A, g, projectors)
        x_hat, projectors = hard_threshold(x_hat + mu * g, rank, retproj=True)
        yield x_hat


def _expval(A, x):
    return np.dot(A.reshape((len(A), -1)), x.ravel())


def _cgm_iterator(A, y, alpha, svds=svds, ret_gap=False):
    x = np.zeros(A.shape[1:3], dtype=A.dtype)
    for iteration in it.count():
        z = _expval(A, x)
        u, _, v = svds(np.tensordot(z - y, A, axes=(0, 0)), 1)
        h = - alpha * u * v
        eta = 2 / (iteration + 2)
        x = (1 - eta) * x + eta * h

        duality_gap = np.dot(z - _expval(A, h), z - y)
        yield x, duality_gap


def cgm_estimator(A, y, alpha, relerr=1e-1, maxiter=int(1e6)):
    """@todo: Docstring for cgm_estimator.

    """
    solution = _cgm_iterator(A, y, alpha, ret_gap=True)
    for x, gap in it.islice(solution, maxiter):
        if gap < relerr:
            return x

    raise ValueError("Did not find solution with error < {} in {} iterations"
                     .format(relerr, maxiter))
