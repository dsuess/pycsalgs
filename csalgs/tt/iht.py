# encoding: utf-8
"""Providing tensor iterative hard thresholding algorithm"""
from __future__ import absolute_import, division, print_function

import functools as ft

import numpy as np

import mpnum as mp
import mpnum.special as mpsp
from mpnum._testing import compression_svd


__all__ = ['iht_estimator', 'iht_estimator_mpa']


def _get_stepsize_np(A, g, projectors):
    """Computes the adaptive stepsize for TIHT in the representation using
    dense numpy arrays.

    :param A: Design matrix as ndarray
    :param g: Gradient at x given by :math:`A^*(y - AX)` as ndarray
    :param projectors: List of projectors on the left and right eigenspaces of
        the matrications :math:`X^{[i]}`. Therefore, `projectors[0]` only acts
        on the leftmost index, `projectors[1]` only on the two leftmost indices
        and `projectors[-1]` on all but the rightmost index
    :returns: Stepsize given by :math:`\frac{\Vert Pg \Vert^2}{\Vert A Pg \Vert}`
        where :math:`P` denotes the action of all the eigenspace projectors
        in `projectors`.

    """
    # Apply all the projectors
    for P, _ in projectors:
        g = np.tensordot(P, g, axes=(1, 0))
        g = g.reshape((g.shape[0] * g.shape[1], ) + g.shape[2:])
    g = g.ravel()

    m = A.shape[0]
    numerator = np.linalg.norm(g)**2
    denominator = np.linalg.norm(np.dot(A.reshape((m, -1)), g))**2
    return numerator / denominator


def iht_estimator(A, observation, bdim, X_init=None, stepsize='adaptive',
                  batchsize=None, retinfo=None):
    """Performs TIHT on given observations and design matrix.

    :param A: Design matrix of shape (nr_obs, ) + nr_sites * (local_dim,)
    :param observation: Array containing observations
    :param bdim: Max. bonddim of the projection step
    :param X_init: Intialisation (default 0)
    :param stepsize: Stepsize of one iteration step. Either pass a constant
        number or 'adaptive' (default 'adaptive')
    :param batchsize: Size of the batch for the stochastic gradient descent.
        `None` corresponds to full gradient descent
    :param retinfo: If not None, it should be a list of local variable names
        that should be returned from the function for debugging purposes
    :returns: Infinte iterator containing the iteration steps

    """
    X = X_init if X_init is not None else np.zeros(A.shape[1:])
    random_selection = ft.partial(np.random.choice, len(A), size=batchsize,
                                  replace=False)
    sel = slice(None) if batchsize is None else random_selection()

    _, projectors = compression_svd(X[sel], bdim=bdim, retproj=True)

    while True:
        if batchsize is not None:
            sel = random_selection()
            A_, y_ = A[sel], observation[sel]
            m = batchsize
        else:
            A_, y_ = A, observation
            m = len(A)

        y_sharp = np.dot(A_.reshape((m, -1)), X.ravel())
        g = np.tensordot(A_, y_ - y_sharp, axes=(0, 0))

        if stepsize == 'adaptive':
            mu = _get_stepsize_np(A_, g, projectors) * m / len(A)
        elif stepsize == 'adaptive_noproj':
            mu = _get_stepsize_np(A_, g, [])
        else:
            mu = stepsize

        X, projectors = compression_svd(X + mu * g, bdim=bdim, retproj=True)

        if retinfo is not None:
            info = dict()
            for s in retinfo:
                info[s] = locals()[s]
            yield (X, info)
        else:
            yield X


# Any stepsize function should have the following signature:
#  :param A: Design matrix as MPArray
#  :param g: Gradient at x given by :math:`A^*(y - AX)` as MPArray
#  :param X: The current value of math:`X`. Brought to left-normal form, the
#      local tensors are exactly the projectors needed
#  :returns: Stepsize `mu` as float
def _get_stepsize_mpa(A, g):
    """Computes stepsize :math:`\mu = \frac{\Vert g \Vert^2}{\Vert A g \Vert^2}`
    for TIHT in the representation using MPArrays.

    For more details on the arguments see :func:`TihtStepsize.__call__`.

    """
    numerator = mp.norm(g)**2
    denominator = sum(mpsp.inner_prod_mps(a, g)**2 for a in A)
    return float(numerator / denominator)


def adaptive_stepsize(scale=1.0):
    def stepsize(A, g, X):
        return scale * _get_stepsize_mpa(A, g)

    return stepsize


def constant_stepsize(stepsize):
    return (lambda A, g, X: stepsize)


def fully_projected(stepfun, compargs={}):
    def stepsize(A, g, X):
        X.normalize(left=len(X) - 1)

        # get the left-eigenvectors
        Us, _ = X.split(len(X) - 2)
        Us = Us.reshape([s + (1,) for s in Us.pdims[:-1]] + [Us.pdims[-1]])
        proj = mp.dot(Us.conj(), Us, axes=(1, 1))
        g = mp.partialdot(proj, g, start_at=0)
        g.compress(**compargs)

        return stepfun(A, g, _)

    return stepsize


def _local_contraction(l):
    l = np.rollaxis(l, 1).reshape((l.shape[1], -1))
    return np.tensordot(l.conj(), l, axes=(1, 1))


def locally_projected(stepfun, compargs={}):
    def stepsize(A, g, X):
        # its actually not a projection, but it works surprisingly
        proj = mp.MPArray([_local_contraction(l)[None, ..., None]
                           for l in X.lt[:-1]])
        g = mp.partialdot(proj, g, start_at=0)
        g.compress(**compargs)

        return stepfun(A, g, None)
    return stepsize


def memory_stepsize(stepfun, const_steps=1, choose=lambda x: x[-1]):
    def stepsize(A, g, X):
        if stepsize.counter >= const_steps:
            mu = stepfun(A, g, X)
            stepsize.memory.append(mu)
            stepsize.counter = 1
        else:
            stepsize.counter += 1
        return choose(stepsize.memory)

    stepsize.counter = float('Inf')
    stepsize.memory = []
    return stepsize


def iht_estimator_mpa(A, observation, bdim, X_init=None, batchsize=None, retinfo=None,
                      stepsize=adaptive_stepsize(), compargs={'method': 'svd'}):
    """Performs TIHT in MPArray-representation on given observations and
    design matrix.

    :param A: Design matrix of shape (nr_obs, ) + nr_sites * (local_dim,)
        given as numpy array of MPArrays
    :param observations: Array containing observations
    :param bdim: Max. bonddim of the projection step
    :param X_init: Intialisation given as MPArray(default 0)
    :param stepsize: Stepsize of one iteration step. Either pass a constant
        number or 'adaptive' (default 'adaptive')
    :param batchsize: Size of the batch for the stochastic gradient descent.
        `None` corresponds to full gradient descent
    :returns: Infinte iterator containing the iteration steps

    """
    assert 'bdim' not in compargs
    # FIXME once iterator of MParray is fixed
    Atmp = np.empty(len(A), dtype=object)
    Atmp[:] = A
    A = Atmp

    pdim = A[0].pdims
    sites = len(A[0])

    X = X_init if X_init is not None else mp.zero(sites, pdim, 1)

    random_selection = ft.partial(np.random.choice, len(A), size=batchsize,
                                  replace=False)
    sel = slice(None) if batchsize is None else random_selection()
    # Need to add second variable since its not well defined for the first step
    X_for_stepsize = mp.sumup(A[sel], weights=observation[sel])
    X_for_stepsize.compress(**compargs)

    while True:
        if batchsize is not None:
            sel = random_selection()
            A_, y_ = A[sel], observation[sel]
            m = batchsize
        else:
            A_, y_ = A, observation
            m = len(observation)

        y_sharp = np.array([mpsp.inner_prod_mps(a, X) for a in A_])
        g = mp.sumup(A_, weights=y_ - y_sharp)
        g.compress(bdim=2 * bdim, **compargs)

        mu = stepsize(A, g, X_for_stepsize)

        X += float(mu) * g
        X.compress(**compargs, bdim=bdim)
        X_for_stepsize = X

        if retinfo is not None:
            info = dict()
            for s in retinfo:
                info[s] = locals()[s]
            yield (X, info)
        else:
            yield X
