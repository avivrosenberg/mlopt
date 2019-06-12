import matplotlib

matplotlib.use('TKAgg')

import matplotlib.pyplot as plt
import os
import pandas as pd
import linreg_main
import matcomp_main
import numpy as np

from ast import literal_eval


DEFAULT_CFG_FILE = os.path.join('hw2', 'cfg', 'hw2-linreg.json')

plt.rcParams["figure.figsize"] = [20, 10]
axis_fontsize = 18
title_fontsize = 26

if __name__ == '__main__':
    print('=== MLOPT HW2: Aviv Rosenberg & Yonatan Elul')
    print('=== ========================================')

    print('=== Running matrix completion models...')

    names = {
        'rp': "Rank Projection",
        'ff': "Factorized Form:",
        'cr': "Convex Relaxation",
    }

    params = {
        'rp': None,
        'ff': None,
        'cr': None,
    }

    test_mse = {
        'rp': None,
        'ff': None,
        'cr': None,
    }

    for dataset_name in matcomp_main.DATASETS.keys():
        for model_name in matcomp_main.MODELS.keys():
            cv_results_file = os.path.join(matcomp_main.OUT_DIR_DEFAULT,
                                           f'cv-{model_name}.tsv')
            if os.path.isfile(cv_results_file):
                df = pd.read_csv(cv_results_file, sep='\t')
                df = df.loc[df['rank_test_score'] == 1]['params']
                best_params = literal_eval(df.values[0])
                print(f'=== Using best cross-validated params: {best_params}')
            else:
                best_params = {}
                print(f'=== No cross-validation results file, using default '
                      f'parameters')

            res = matcomp_main.run_training(
                model_name, dataset_name, matcomp_main.OUT_DIR_DEFAULT,
                no_test_set=False, test_ratio=1/3, random_seed=42,
                **best_params
            )

            # Save intermediate results, to be plotted at the end of the loop
            if res['model'].name == 'cr':
                params[res['model'].name] = res['model'].tau

            else:
                params[res['model'].name] = res['model'].rank

            test_mse[res['model'].name] = res['model'].test_mse_
            max_iter = res['model'].max_iter

        if dataset_name[-1] == 'k':
            ds_name = "MovieLens 100K"

        else:
            ds_name = "MovieLens 1M"

        iterations = np.arange(start=1, stop=(max_iter + 1))

        plt.figure()
        plt.plot(iterations, test_mse['rp'], 'b')
        plt.plot(iterations, test_mse['ff'], 'r')
        plt.plot(iterations, test_mse['cr'], 'k')
        plt.legend([
            names['rp'] + ' - Rank = ' + f"{params['rp']}",
            names['ff'] + ' - Rank = ' + f"{params['ff']}",
            names['cr'] + ' - $\\tau$ = ' + f"{params['cr']}",
        ])
        plt.title(ds_name)
        plt.xlabel("Iteration #")
        plt.ylabel("Test Set MSE")

    plt.show()

    print('=== Running linear regression models...')
    print('=== ===================================')
    plot_filenames = linreg_main.run_multi(DEFAULT_CFG_FILE, parallel=True)
    print(f'=== Saved plot files: {plot_filenames}')

