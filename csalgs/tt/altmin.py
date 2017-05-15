#encoding: utf-8

import itertools as it
import collections

import numpy as np
from scipy.linalg.blas import dgemm

from mpnum.mparray import _local_dot, _ltens_to_array
from mpnum.special import sumup
from warnings import warn

try:
    from sklearn.utils.extmath import randomized_svd as svdfunc
except ImportError:
    warn("The randomized SVD is unavailable. Fallback to standard dense SVD." +
         " Consider installing scikit-learn for much faster initialization.")
    from mpnum._tools import truncated_svd as svdfunc

__all__ = ['AltminEstimator']


def _llsq_solver_fast(A, y):
    """ Return the least-squares solution to a linear matrix equation.
    Solves the equation `A x = y` by computing a vector `x` that
    minimizes the Euclidean 2-norm `|| b - a x ||^2`.  The equation may
    be under-, well-, or over- determined (i.e., the number of
    linearly independent rows of `A` can be less than, equal to, or
    greater than its number of linearly independent columns).  If `A`
    is square and of full rank, then `x` (but for round-off error) is
    the "exact" solution of the equation.

    However, if A is rank-deficient, this solver may fail. In that case, use
    :func:`_llsq_solver_pinv`.

    :param A: (m, d) array like
    :param y: (m,) array_like
    :returns x: (d,) ndarray, least square solution

    """
    if type(A) != np.ndarray or not A.flags['C_CONTIGUOUS']:
        warn("Matrix a is not a C-contiguous numpy array. " +
             "The solver will create a copy, which will result" +
             " in increased memory usage.")

    A = np.asarray(A, order='c')
    i = dgemm(alpha=1.0, a=A.T, b=A.T, trans_b=True)
    x = np.linalg.solve(i, dgemm(alpha=1.0, a=A.T, b=y)).flatten()

    return x


def _llsq_solver_pinv(A, y):
    """Same as :func:`llsq_solver_fast` but more robust, albeit slower.

    :param A: (m, d) array like
    :param y: (m,) array_like
    :returns x: (d,) ndarray, least square solution

    """
    B = np.linalg.pinv(A.T @  A)
    return B @ A.T @ y


def llsqsolve(A, y):
    """@todo: Docstring for llsqsolve.

    :param A: @todo
    :param y: @todo
    :returns: @todo

    """
    try:
        return _llsq_solver_fast(A, y)
    except np.linalg.LinAlgError:
        return _llsq_solver_pinv(A, y)


def _get_optimmat_row(Ai, X, pos):
    iterator = zip(Ai.lt, X.lt)

    b_l = _ltens_to_array(_local_dot(a, x, axes=(1, 1)) for a, x in it.islice(iterator, pos))[0] \
        if pos > 0 else np.ones(1)
    a_c = next(iterator)[0][0, :, 0]
    b_r = _ltens_to_array(_local_dot(a, x, axes=(1, 1)) for a, x in iterator)[..., 0] \
        if pos < len(X) - 1 else np.ones(1)

    return b_l[:, None, None] * a_c[None, :, None] * b_r[None, None, :]


class AltminEstimator(object):
    """Docstring for AltminEstimator. """

    def __init__(self, A, y, rank, X_init=None, llsqsolve=llsqsolve):
        """@todo: to be defined1.

        :param A: List of mpnum.MPArray containing the measurements. For now,
            only product measurements (i.e. of rank 1) are allowed
        :param y: List containing the measured values
        :param rank: Rank the reconstruction should have (either single integer
            or list of integers for each bond separately)
        :param X_init:

        """
        assert len(A) == len(y)
        assert all(all(bdim == 1 for bdim in a.bdims) for a in A)

        self._A = A
        self._y = y
        self._rank = tuple(rank) if isinstance(rank, collections.Iterable) \
                else (rank,) * (len(A[0]) - 1)
        self._rank = rank
        self._llsqsolve = llsqsolve

        if X_init is None:
            self._X_init = sumup(A, rank, weights=y, svdfunc=svdfunc)
        else:
            self._X_init = X_init

    def _get_optimmat(self, X, direction='right'):
        if direction is 'right':
            idx = range(len(X) - 1)
        elif direction is 'left':
            idx = range(len(X) - 1, 0, -1)
        else:
            raise ValueError(f"{direction} is not a valid direction")

        for pos in idx:
            yield pos, np.array([_get_optimmat_row(a, X, pos) for a in self._A])

    def _altmin_step(self, X):
        for pos, B in self._get_optimmat(X, direction='right'):
            shape = B.shape[1:]
            ltens = self._llsqsolve(B.reshape((B.shape[0], -1)), self._y)
            X.lt.update(pos, ltens.reshape(shape))
            X.normalize(left=pos + 1)

        for pos, B in self._get_optimmat(X, direction='left'):
            shape = B.shape[1:]
            ltens = self._llsqsolve(B.reshape((B.shape[0], -1)), self._y)
            X.lt.update(pos, ltens.reshape(shape))
            X.normalize(right=pos)

        return X

    def __iter__(self):
        X_sharp = self._X_init
        yield X_sharp.copy()

        while True:
            self._altmin_step(X_sharp)
            yield X_sharp.copy()
