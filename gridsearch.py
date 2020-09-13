# 
# This is an intermediate version of the code which
# uses the BayesianOptimization package.
#

from bayes_opt import BayesianOptimization, UtilityFunction
from sklearn.gaussian_process.kernels import Matern, RBF
from scipy.stats import spearmanr
import pandas as pd
import numpy as np

def opt(**kwargs):
    global df, dfo, mu, std

    dim         = dict.get(kwargs, 'dim', 2) # dim     = 2 # 2, 5, 8, 30
    init_points = dict.get(kwargs, 'init_points', 100)
    acq         = dict.get(kwargs, 'acq', 'ucb')
    n_iter      = dict.get(kwargs, 'n_iter', 256)
    kernel      = dict.get(kwargs, 'kernel', 'matern')
    nu          = dict.get(kwargs, 'nu', 2.5)
    kappa       = dict.get(kwargs, 'kappa', 2.5)
    xi          = dict.get(kwargs, 'xi', 1e-2)
    # ard       = dict.get(kwargs, 'ard', 1e-2)

    dfo     = pd.read_csv(f"data/df-{dim}.csv").set_index('mutation')
    df      = dfo.copy()
    mu, std = df.mean(axis = 0), df.std(axis=0)

    df -= df.mean(axis = 0)
    df /= std

    print("Spearman cor:", spearmanr(df.value, df.model)[0])

    X, y, mutant_ids = df.drop(columns='value'), df.value, []

    def closest(**values_dict):
        # euclid = (X - pd.Series(values_dict)).abs().sum(1)
        euclid = (X - pd.Series(values_dict)).pow(2).sum(1).pow(0.5)
        euclid = euclid.sort_values()[:1]
        top    = euclid.index[0]
        mutant_ids.append(top)
        return top
        
    def black_box_function(**args):
        top = closest(**args)
        val = y.loc[top]
        orig = mu.value + val*std.value
        print(top, "\t", orig)

        if orig >= 0.15:
            print("=========YES=========")

        return y.loc[top]
        
    min_max = list(zip(X.min().tolist(), X.max().tolist()))
    bounds  = dict(zip(X.columns.tolist(), min_max))

    optimizer = BayesianOptimization(
        f            = black_box_function,
        pbounds      = bounds, 
        verbose      = 0, # verbose = 1 prints when a maximum is observed, 0 is silent
        random_state = 0
    )

    if kernel == 'matern':
        kernel_ = Matern(length_scale = [1]*(dim+1), nu=nu)
    else:
        kernel_ = RBF()

    optimizer.maximize(**{
        'acq':         acq, 
        'init_points': init_points,
        'n_iter':      n_iter,
        'kernel':      kernel_, #Matern(length_scale = [1, 1, 1], nu=0.5), 
        'kappa':       kappa,
        'xi':          xi
        #'alpha':      1e-3,
    })

    series = list(map(lambda e: mu.value + e['target']*std.value, optimizer.res))
    save_df = pd.DataFrame(data={'mutant': mutant_ids, 'value': series})

    append = ''
    if kernel == 'matern':
        kernel += f"(nu{str(nu).replace('.',',')})"
    else:
        kernel = ''

    save_df.to_csv(f"""
        results/
         dim:{dim}
        -init_points:{init_points}
        -acq:{acq}
        -kappa:{str(kappa).replace('.',',')}
        -xi:{str(xi).replace('.',',')}
        -n_iter:{n_iter}
        -kernel:{kernel}
        .csv
    """.replace('\n','').replace(' ',''))

if __name__ == "__main__":
    # {'acq': 'ucb', 'params': [ { 'kappa': x } for x in [1e-2, 1e-1, 1e-0, 2,    5,    10]]},
    # {'acq': 'ei',  'params': [ { 'xi':    x } for x in [1e-4, 1e-3, 1e-2, 1e-1, 1e-0,  2]]},
    # {'acq': 'poi', 'params': [ { 'xi':    x } for x in [1e-4, 1e-3, 1e-2, 1e-1, 1e-0,  2]]}

    # for kappa in [1e-3, 1e-2, 1e-1, 1e-0, 2, 5, 10]:
    #     opt(dim=16, kappa=kappa, kernel='matern', nu=2.5, n_iter=32)

    for kappa in [1e-2]:
        opt(dim=16, kappa=kappa, kernel='matern', nu=2.5, n_iter=512)

    # opt(dim=16, kernel='matern') (kappa 2.5, finds K190Q)
    
