#encoding: utf-8

import itertools as it
import collections

import numpy as np
from scipy.linalg.blas import dgemm

from mpnum.mparray import _local_dot, _ltens_to_array, normdist
from mpnum.special import sumup
from warnings import warn

from mpnum.utils.extmath import randomized_svd as svdfunc

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


def partial_inner_prod(mpa1, mpa2, direction):
    """

    :param mpa1: MPArray with one physical leg per site and bond dimension 1
    :param mpa2: MPArray with same physical shape as mpa1
    :returns: @TODO

    """
    assert all(rank == 1 for rank in mpa1.ranks)
    assert all(pleg == 1 for pleg in mpa1.ndims)
    assert all(pleg == 1 for pleg in mpa2.ndims)

    if direction is 'left':
        ltens1 = iter(mpa1.lt._ltens)
        ltens2 = iter(mpa2.lt._ltens)

        res = np.dot(next(ltens1)[0, :, 0].conj(), next(ltens2))
        yield res
        for l1, l2 in zip(ltens1, ltens2):
            res = np.dot(res, np.dot(l1[0, :, 0].conj(), l2))
            yield res

    elif direction is 'right':
        ltens1 = iter(mpa1.lt._ltens[::-1])
        ltens2 = iter(mpa2.lt._ltens[::-1])

        res = np.dot(next(ltens1)[0, :, 0].conj(), next(ltens2))
        yield res
        for l1, l2 in zip(ltens1, ltens2):
            res = np.dot(np.dot(l1[0, :, 0].conj(), l2), res)
            yield res


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
        assert all(all(rank == 1 for rank in a.ranks) for a in A)

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
        # get rid of the last entry since that is the full inner product
        partials = [list(partial_inner_prod(a, X, direction))[:-1]
                    for a in self._A]
        if direction is 'right':
            left_terms = np.ones((len(self._A), 1))
            for pos in range(len(X) - 1):
                right_terms = (partial.pop().ravel() for partial in partials)
                rows = [b_l[:, None, None] * a.lt._ltens[pos] * b_r[None, None, :]
                        for b_l, a, b_r in zip(left_terms, self._A, right_terms)]
                yield pos, np.asarray(rows)
                left_terms = [np.dot(b_l, np.dot(a.lt._ltens[pos][0, :, 0], X.lt._ltens[pos]))
                              for b_l, a in zip(left_terms, self._A)]
            return

        elif direction is 'left':
            right_terms = np.ones((len(self._A), 1))
            for pos in range(len(X) - 1, 0, -1):
                left_terms = (partial.pop().ravel() for partial in partials)
                rows = [b_l[:, None, None] * a.lt[pos] * b_r[None, None, :]
                        for b_l, a, b_r in zip(left_terms, self._A, right_terms)]
                yield pos, np.asarray(rows)
                right_terms = [np.dot(np.dot(a.lt._ltens[pos][0, :, 0], X.lt._ltens[pos]), b_r)
                              for b_r, a in zip(right_terms, self._A)]
            return
        else:
            raise ValueError(f"{direction} is not a valid direction")

    def _altmin_step(self, X):
        for pos, B in self._get_optimmat(X, direction='right'):
            shape = B.shape[1:]
            ltens = self._llsqsolve(B.reshape((B.shape[0], -1)), self._y)
            X.lt.update(pos, ltens.reshape(shape))
            X.canonicalize(left=pos + 1)

        for pos, B in self._get_optimmat(X, direction='left'):
            shape = B.shape[1:]
            ltens = self._llsqsolve(B.reshape((B.shape[0], -1)), self._y)
            X.lt.update(pos, ltens.reshape(shape))
            X.canonicalize(right=pos)

        return X

    def __iter__(self):
        X_sharp = self._X_init
        yield X_sharp.copy()

        while True:
            self._altmin_step(X_sharp)
            yield X_sharp.copy()

    def estimate(self, n_steps=25, thresh=None):
        X_old = self._X_init.copy()
        for _ in range(n_steps):
            X_new = self._altmin_step(X_old.copy())
            if (thresh is not None) and (normdist(X_old, X_new) < thresh):
                break
            X_old = X_new
        return X_new
