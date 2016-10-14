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
