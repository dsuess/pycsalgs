# coding: utf-8
import numpy as np
import ipyparallel
import pandas as pd
from os import environ, rename
import progressbar as pb
from time import time


CLUSTER_ID = environ.get('CLUSTER_ID', None)
_CLIENTS = ipyparallel.Client(cluster_id=CLUSTER_ID)
_VIEW = _CLIENTS.load_balanced_view()
print("Kernels available: {}".format(len(_CLIENTS)))


SAMPLES = 100
SITES = [2, 3, 4]
CS = np.linspace(1.0, 2, 5)
RANK = 5
DIM = 2
RGEN = np.random.RandomState(1235)
SAVE_EVERY_SECONDS = 5
FILENAME = 'phase_transition.pkl'


with _CLIENTS[:].sync_imports():
    import numpy
    import mpnum
    from csalgs.tt.altmin import AltminEstimator
    from mpnum.special import inner_prod_mps


def experiment_generator(sites, dim, rank, C, dist_crit=1e-4, maxiter=40):

    def run(seed):
        rgen = numpy.random.RandomState(seed)
        X = mpnum.random_mpa(sites, dim, rank, randstate=rgen, normalized=True)
        nr_measurements = int(C * dim * sites * rank**2 * numpy.log2(rank + 1))
        A = [mpnum.random_mpa(len(X), X.pdims, 1, randstate=rgen,
                              normalized=True, dtype=X.dtype)
             for _ in range(nr_measurements)]
        y = [inner_prod_mps(a, X) for a in A]

        X_sharp = AltminEstimator(A, y, rank).estimate(maxiter, thresh=dist_crit)

        return {'X': X, 'X_sharp': X_sharp, 'dist': mpnum.normdist(X, X_sharp),
                'C': C, 'seed': seed, 'dim': dim, 'rank': rank, 'sites': sites}

    return run


seeds = RGEN.randint(2**31, size=SAMPLES)

df = pd.DataFrame(columns=['X', 'X_sharp', 'dist', 'C', 'seed', 'dim', 'rank', 'sites'])
task_list = []
for sites in SITES:
    for C in CS:
        sample = experiment_generator(sites, DIM, RANK, C)
        task = _VIEW.map_async(sample, seeds)
        task_list.append(task)

counter = 0
last_saved = 0
widgets = [pb.SimpleProgress(), '  [', pb.Percentage() , ']     ',
        pb.DynamicMessage('saved@')]
bar = pb.ProgressBar(max_value=sum(len(task) for task in task_list),
                    widgets=widgets)
bar.start()
for task in task_list:
    for entry in task:
        new_row = pd.DataFrame(entry, index=[counter])
        counter += 1
        df = df.append(new_row)
        bar.update(value=counter)

        if time() - last_saved > SAVE_EVERY_SECONDS:
            df.to_pickle(FILENAME + '.tmp')
            rename(FILENAME + '.tmp', FILENAME)
            done_tasks = sum(task.progress for task in task_list)
            # Set to current time since saving might actually take a while
            bar.dynamic_messages = {'saved@': counter}
            last_saved = time()

df.to_pickle(FILENAME + '.tmp')
rename(FILENAME + '.tmp', FILENAME)
