# encoding: utf-8
"""
[1] K. Zhong, P. Jain, and I. S. Dhillon, Efficient matrix sensing using rank-1
    gaussian measurements, in International Conference on Algorithmic Learning
    Theory, 2015, pp. 3–18.
"""

import cvxpy as cvx
import numpy as np


__all__ = ['nucmin_estimator', 'constrained_l2_estimator']


def _expval(A, x):
    """@todo: Docstring for _expval.

    :param A: @todo
    :param x: @todo
    :returns: @todo

    """
    # transpose A since cvxpy vectorizes in Fortran order
    A_map = A.transpose(0, 2, 1).reshape((len(A), -1))
    return A_map * cvx.vec(x)


def nucmin_estimator(A, y, eta=None, **kwargs):
    """@todo: Docstring for nucmin_estimator.

    :param A: @todo
    :param y: @todo
    :param **kwargs: @todo
    :returns: @todo

    """
    x_sharp = cvx.Variable(A.shape[1], A.shape[2])
    objective = cvx.Minimize(cvx.normNuc(x_sharp))

    if eta is None:
        constraints = [_expval(A, x_sharp) == y]
    else:
        constraints = [cvx.abs(_expval(A, x_sharp) - y) < eta]

    problem = cvx.Problem(objective, constraints)
    problem.solve(**kwargs)

    if problem.status not in ['optimal']:
        raise ValueError("Optimization did not converge: " + problem.status)
    return np.array(x_sharp.value)


def constrained_l2_estimator(A, Y, alpha, **kwargs):
    x_sharp = cvx.Variable(rows=A.shape[1], cols=A.shape[2])
    objective = cvx.Minimize(cvx.norm2(Y - _expval(A, x_sharp)))
    constraints = [cvx.norm(x_sharp, p='nuc') <= alpha]

    problem = cvx.Problem(objective, constraints)
    problem.solve(**kwargs)

    if problem.status not in ['optimal']:
        raise ValueError("Optimization did not converge: " + problem.status)
    return np.asarray(x_sharp.value)
