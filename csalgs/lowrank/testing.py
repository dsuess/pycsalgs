# encoding: utf-8

import numpy as np


def random_lowrank_matrix(dim, rank, hermitian=True, rgen=np.random):
    A = rgen.randn(dim, rank)
    B = A if hermitian else rgen.randn(dim, rank)
    return A @ B.T


def random_lowrank_matrix_cnr(dim, rank, condition_scale=[1.], hermitian=True,
                              rgen=np.random):
    scale = rgen.choice(condition_scale, size=rank, replace=True).astype(np.float_)
    scale = np.concatenate((scale, np.ones(dim - rank)))
    A = random_lowrank_matrix(dim, rank, hermitian=hermitian, rgen=rgen)
    U, s, V = np.linalg.svd(A)
    return U @ np.diag(s * scale) @ V.T


def sensingmat_gauss(measurements, dim, rgen=np.random):
    return rgen.randn(measurements, dim, dim) / np.sqrt(measurements)


def sensingmat_rank1(measurements, dim, hermitian=True, rgen=np.random):
    A = rgen.randn(measurements, dim)
    if hermitian:
        B = A
    else:
        B = rgen.randn(measurements, dim)
    return A[:, :, None] * B[:, None, :] / np.sqrt(measurements)
