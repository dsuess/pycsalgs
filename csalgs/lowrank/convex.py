# encoding: utf-8
"""
[1] K. Zhong, P. Jain, and I. S. Dhillon, Efficient matrix sensing using rank-1
    gaussian measurements, in International Conference on Algorithmic Learning
    Theory, 2015, pp. 3–18.
"""

import cvxpy as cvx
import numpy as np


__all__ = ['nucmin_estimator']


def nucmin_estimator(A, y, eta=None, **kwargs):
    """@todo: Docstring for nucmin_estimator.

    :param A: @todo
    :param y: @todo
    :param **kwargs: @todo
    :returns: @todo

    """
    dim = A.shape[1]
    x_sharp = cvx.Variable(dim, dim)
    objective = cvx.Minimize(cvx.normNuc(x_sharp))

    if eta is None:
        constraints = [cvx.trace(a.T * x_sharp) == yi for a, yi in zip(A, y)]
    else:
        constraints = [cvx.abs(cvx.trace(a.T * x_sharp) - yi) < eta
                       for a, yi in zip(A, y)]

    problem = cvx.Problem(objective, constraints)
    problem.solve(**kwargs)

    if problem.status not in ['optimal']:
        raise ValueError("Optimization did not converge: " + problem.status)
    return np.array(x_sharp.value)
