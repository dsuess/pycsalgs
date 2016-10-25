#encoding: utf-8


import functools as ft
import itertools as it

import numpy as np
import theano as th
import theano.tensor as T
from scipy.linalg.blas import dgemm

import mpnum as mp
from mpnum.mparray import _local_dot, _ltens_to_array
from warnings import warn

__all__ = ['altmin_step', 'altmin_estimator']


def _linear_least_squares(A, y):
    """
    Return the least-squares solution to a linear matrix equation.
    Solves the equation `A x = y` by computing a vector `x` that
    minimizes the Euclidean 2-norm `|| b - a x ||^2`.  The equation may
    be under-, well-, or over- determined (i.e., the number of
    linearly independent rows of `A` can be less than, equal to, or
    greater than its number of linearly independent columns).  If `A`
    is square and of full rank, then `x` (but for round-off error) is
    the "exact" solution of the equation.

    :param A: (m, d) array like
    :param y: (m,) array_like
    :returns x: (d,) ndarray, least square solution

    """
    if type(A) != np.ndarray or not A.flags['C_CONTIGUOUS']:
        warn('Matrix a is not a C-contiguous numpy array. ' +
             'The solver will create a copy, which will result' +
             ' in increased memory usage.')

    A = np.asarray(A, order='c')
    i = dgemm(alpha=1.0, a=A.T, b=A.T, trans_b=True)
    x = np.linalg.solve(i, dgemm(alpha=1.0, a=A.T, b=y)).flatten()

    return x


def _get_optimmat_row(Ai, X, pos):
    iterator = zip(Ai.lt, X.lt)

    b_l = _ltens_to_array(_local_dot(a, x, axes=(1, 1)) for a, x in it.islice(iterator, pos))[0] \
        if pos > 0 else np.ones(1)
    a_c = next(iterator)[0][0, :, 0]
    b_r = _ltens_to_array(_local_dot(a, x, axes=(1, 1)) for a, x in iterator)[..., 0] \
        if pos < len(X) - 1 else np.ones(1)

    return b_l[:, None, None] * a_c[None, :, None] * b_r[None, None, :]


def _get_optimmat(A, X, pos):
    return np.array([_get_optimmat_row(a, X, pos) for a in A])


def altmin_step(A, y, X):
    for pos in range(len(X) - 1):
        B = _get_optimmat(A, X, pos)
        shape = B.shape[1:]
        ltens = _linear_least_squares(B.reshape((B.shape[0], -1)), y)
        X.lt.update(pos, ltens.reshape(shape))
        X.normalize(left=pos + 1)

    for pos in range(len(X) - 1, 0, -1):
        B = _get_optimmat(A, X, pos)
        shape = B.shape[1:]
        ltens = _linear_least_squares(B.reshape((B.shape[0], -1)), y)
        X.lt.update(pos, ltens.reshape(shape))
        X.normalize(right=pos)

    return X


def altmin_estimator(A, y, rank, X_init=None):
    X_sharp = mp.sumup(A, weights=y).compression('svd', bdim=rank) \
        if X_init is None else X_init
    yield X_sharp.copy()

    while True:
        altmin_step(A, y, X_sharp)
        yield X_sharp.copy()
