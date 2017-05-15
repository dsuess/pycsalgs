# coding: utf-8
import numpy as np
import ipyparallel
import pandas as pd
from os import environ
from tools.helpers import AsyncTaskWatcher


CLUSTER_ID = environ.get('CLUSTER_ID', None)
_CLIENTS = ipyparallel.Client(cluster_id=CLUSTER_ID)
_VIEW = _CLIENTS.load_balanced_view()
print("Kernels available: {}".format(len(_CLIENTS)))


SAMPLES = 1
SITES = [2, 3, 4, 5, 6, 7, 8, 9, 10]
CS = np.linspace(1.0, 6.0, 10)
RANK = 10
DIM = 4
RGEN = np.random.RandomState(1235)


with _CLIENTS[:].sync_imports():
    import numpy
    import mpnum
    from csalgs.tt.altmin import AltminEstimator
    from mpnum.special import inner_prod_mps


def experiment_generator(sites, dim, rank, C, dist_crit=1e-4, maxiter=25):

    def run(seed):
        rgen = numpy.random.RandomState(seed)
        X = mpnum.random_mpa(sites, dim, rank, randstate=rgen, normalized=True)
        nr_measurements = int(C * dim * sites * rank**2 * np.log2(rank + 1))
        A = [mpnum.random_mpa(len(X), X.pdims, 1, randstate=rgen,
                              normalized=True, dtype=X.dtype)
             for _ in range(nr_measurements)]
        y = [inner_prod_mps(a, X) for a in A]

        X_sharp = AltminEstimator(A, y, rank).estimate(maxiter, thresh=dist_crit)

        return {'X': X, 'X_sharp': X_sharp, 'dist': mpnum.normdist(X, X_sharp),
                'C': C, 'seed': seed, 'dim': dim, 'rank': rank, 'sites': sites}

    return run


seeds = RGEN.randint(2**31, size=SAMPLES)

watcher = AsyncTaskWatcher()
for sites in SITES:
    for C in CS:
        sample = experiment_generator(sites, DIM, RANK, C)
        task = _VIEW.map_async(sample, seeds)
        watcher.append(task)
watcher.block()


df = pd.DataFrame(columns=['X', 'X_sharp', 'dist', 'C', 'seed', 'dim', 'rank', 'sites'])
counter = 0
for task in watcher._tasks:
    for entry in task.get():
        new_row = pd.DataFrame(entry, index=[counter])
        counter += 1
        df = df.append(new_row)

df.to_pickle('phase_transition.pkl')
