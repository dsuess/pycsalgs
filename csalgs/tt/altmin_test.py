import numpy as np
import itertools as it
import mpnum as mp
from mpnum.mparray import _local_dot, _ltens_to_array
from numpy import testing as nptest
from .altmin import AltminEstimator
import pytest as pt


@pt.mark.parametrize('sites', [4])
@pt.mark.parametrize('dim', [2])
@pt.mark.parametrize('rank', [4])
def test_recover(sites, dim, rank):
    X = mp.random_mpa(sites, dim, rank, normalized=True)
    measurements = 5 * sites * rank**2 * dim
    A = [mp.random_mpa(sites, dim, 1) for _ in range(measurements)]
    Y = [mp.special.inner_prod_mps(a, X) for a in A]

    estimator = AltminEstimator(A, Y, 2 * rank)
    X_hat = next(it.islice(estimator, 10, 11))
    assert mp.normdist(X, X_hat) < 1e-3
