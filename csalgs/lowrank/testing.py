# encoding: utf-8

import numpy as np


def random_lowrank_hmatrix(dim, rank, rgen=np.random):
    A = rgen.randn(dim, rank)
    return A @ A.T


def random_lowrank_matrix(dim, rank, rgen=np.random):
    A, B = rgen.randn(2, dim, rank)
    return A @ B.T


def sensingmat_gauss(measurements, dim, rgen=np.random):
    return rgen.randn(measurements, dim, dim) / np.sqrt(measurements)


def sensingmat_rank1(measurements, dim, independent=True, rgen=np.random):
    A = rgen.randn(measurements, dim)
    A /= np.linalg.norm(A, axis=1, keepdims=True)

    if independent:
        B = rgen.randn(measurements, dim)
        B /= np.linalg.norm(A, axis=1, keepdims=True)
    else:
        B = A

    return A[:, :, None] * B[:, None, :]
