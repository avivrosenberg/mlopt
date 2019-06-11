import os
import pandas as pd
from ast import literal_eval

import linreg_main
import matcomp_main

DEFAULT_CFG_FILE = os.path.join('hw2', 'cfg', 'hw2-linreg.json')

if __name__ == '__main__':
    print('=== MLOPT HW2: Aviv Rosenberg & Yonatan Elul')
    print('=== ========================================')

    print('=== Running matrix completion models...')
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

            # TODO: create plots from the results
            # plot_filenames = ...

    print('=== Running linear regression models...')
    print('=== ===================================')
    plot_filenames = linreg_main.run_multi(DEFAULT_CFG_FILE, parallel=True)
    print(f'=== Saved plot files: {plot_filenames}')

