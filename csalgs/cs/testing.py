# encoding: utf-8

import numpy as np


def random_sparse_vector(dim, nnz, rgen=np.random):
    """@todo: Docstring for random_sparse_vector.

    :param dim: @todo
    :param nnz: @todo
    :param rgen: @todo
    :returns: @todo

    """
    idx = rgen.choice(np.arange(dim), size=nnz, replace=False)
    result = np.zeros(dim)
    result[idx] = rgen.randn(nnz)
    return result


def sensingmat_gauss(measurements, dim, rgen=np.random):
    """Returns a m*n sensing matrix with independent, normal
    components. See the remark below for normalization

    :param measurments: @todo
    :param dim: @todo
    """
    return rgen.randn(measurements, dim) / np.sqrt(measurements)
