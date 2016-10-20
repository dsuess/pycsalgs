# encoding: utf-8

import numpy as np


def random_lowrank_matrix(dim, rank, hermitian=False, rgen=np.random):
    return sensingmat_rank1(1, dim, hermitian=hermitian, rgen=rgen)[0]


def sensingmat_gauss(measurements, dim, rgen=np.random):
    return rgen.randn(measurements, dim, dim) / np.sqrt(measurements)


def sensingmat_rank1(measurements, dim, hermitian=True, rgen=np.random):
    A = rgen.randn(measurements, dim)
    A /= np.linalg.norm(A, axis=1, keepdims=True)

    if hermitian:
        B = A
    else:
        B = rgen.randn(measurements, dim)
        B /= np.linalg.norm(A, axis=1, keepdims=True)

    return A[:, :, None] * B[:, None, :]
