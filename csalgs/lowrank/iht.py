# encoding: utf-8

import numpy as np
import numpy.linalg as la


__all__ = ['adaptive_stepsize', 'iht_estimator']


def _vec(A):
    newshape = A.shape[:-2]
    newshape = newshape + (A.shape[-2] * A.shape[-1],)
    return A.reshape(newshape)


def _hard_threshold(mat, rank, retproj=False):
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
    _, projectors = _hard_threshold(np.tensordot(y, A, axes=(-1, 0)), rank,
                                    retproj=True)

    while True:
        g = np.tensordot(y - (_vec(A) @ _vec(x_hat)), A, axes=(-1, 0))
        mu = stepsize(A, g, projectors)
        x_hat, projectors = _hard_threshold(x_hat + mu * g, rank, retproj=True)
        yield x_hat
