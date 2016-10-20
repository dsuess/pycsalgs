# encoding: utf-8
"""
[1] K. Zhong, P. Jain, and I. S. Dhillon, Efficient matrix sensing using rank-1
    gaussian measurements, in International Conference on Algorithmic Learning
    Theory, 2015, pp. 3–18.
"""

import numpy as np


__all__ = ['altmin_estimator']


def _min_least_sq(A, y):
    """Computes the least square solution of the linear problem

            argmin_x || y - Ax ||

    using the formula

            x_sharp = inv(A^T A) A^T y

    In case the problem is underdetermined, the inverse can be replaced by
    the Moore-Penrose pseudo inverse

    :param A: @todo
    :param y: @todo
    :returns: @todo

    """
    return np.linalg.pinv(A.T @ A) @ A.T @ y


def altmin_estimator(A, y, rank):
    """@todo: Docstring for altmin_estimator.

    :param A: @todo
    :param y: @todo
    :returns: @todo

    """
    yA = np.tensordot(y, A, axes=(0, 0))
    l_singular, _, _ = np.linalg.svd(yA / len(y))
    U = l_singular[:, :rank]
    shape_U, shape_V = (A.shape[1], rank), (A.shape[2], rank)

    while True:
        AU = np.tensordot(A, U, axes=(1, 0))
        V = _min_least_sq(AU.reshape(len(y), -1), y).reshape(shape_V)
        V, _ = np.linalg.qr(V)

        AV = np.tensordot(A, V, axes=(2, 0))
        U = _min_least_sq(AV.reshape(len(y), -1), y).reshape(shape_U)

        yield U, V
        U, _ = np.linalg.qr(U)
