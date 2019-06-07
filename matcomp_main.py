import argparse
import os
import sys

import pandas as pd
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV

import matcomp.data as data
import matcomp.models as models

OUT_DIR_DEFAULT = os.path.join('out', 'matcomp')
DATASETS = {'ml100k': data.MovieLens100K, 'ml1m': data.MovieLens1M}
MODELS = {'rp': models.RankProjectionMatrixCompletion}


def create_parser():
    def is_dir(dirname):
        if not os.path.isdir(dirname):
            raise argparse.ArgumentTypeError(f'{dirname} is not a directory')
        else:
            return dirname

    def is_file(filename):
        if not os.path.isfile(filename):
            raise argparse.ArgumentTypeError(f'{filename} is not a file')
        else:
            return filename

    p = argparse.ArgumentParser(
        description='MLOPT matcomp: matrix completion solvers',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument('--out-dir', '-o', type=str,
                   help='Output folder for results and plots',
                   default=OUT_DIR_DEFAULT, required=False)
    p.add_argument('--no-plot-title', '-P', action='store_true',
                   required=False, help='Suppress titles in plots')
    p.add_argument('--model', '-m', required=True,
                   choices=['rp', 'ff', 'cr'], dest='model_name',
                   help='Model type to use: rp (rank-projection),'
                        'ff (factorized-form) or cr (convex relaxation)')
    p.add_argument('--dataset', '-d', required=True,
                   choices=['ml100k', 'ml1m'], dest='dataset_name',
                   help='Dataset to use: ml100k (MovieLens100K) or '
                        'ml1m (MovieLens1M)')
    p.add_argument('--test-ratio', type=float, default=1 / 3.,
                   required=False,
                   help='Ratio of test-set (held out) to entire dataset')
    p.add_argument('--random-seed', '-r', type=int, default=42,
                   required=False, help='Random seed for splits')

    sp = p.add_subparsers(dest='subcmd', help='Sub-commands')

    # Cross-validation
    sp_cv = sp.add_parser(
        'cross-validate',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help='Run cross-validation on one matrix-completion model.'
    )
    sp_cv.set_defaults(subcmd_fn=run_cv)
    sp_cv.add_argument('--jobs', type=int, default=4, required=False,
                       help='Number of parallel jobs to run')
    sp_cv.add_argument('--splits', type=int, default=4, required=False,
                       help='Number of splits for cross-validation')

    # Train
    sp_tr = sp.add_parser(
        'train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help='Train a matrix-completion model.'
    )
    sp_tr.set_defaults(subcmd_fn=run_training)
    sp_tr.add_argument('--calc-test-losses', required=False,
                       action='store_true',
                       help='calculate test-set loss at each step')

    # Add model parameters
    for param_name, default_val in models.ALL_PARAMS.items():
        for sp, nargs in [(sp_cv, '+'), (sp_tr, '?')]:
            arg_name = f"--{str.replace(param_name, '_', '-', -1)}"
            if type(default_val) == bool:
                sp.add_argument(arg_name, required=False, action='store_true')
            else:
                sp.add_argument(arg_name, required=False, default=default_val,
                                nargs=nargs, type=type(default_val))

    return p


def parse_cli(parser: argparse.ArgumentParser):
    parsed = parser.parse_args()
    if parsed.subcmd is None:
        parser.print_help()
        sys.exit()
    return parsed


def run_training(model_name, dataset_name, out_dir,
                 calc_test_losses, test_ratio, random_seed, **kw):
    model_params = {}
    for k, v in kw.items():
        if k in models.ALL_PARAMS:
            model_params[k] = v

    dataset: data.MovieLensDataset = DATASETS[dataset_name]()
    model: models.MatrixCompletion = MODELS[model_name](
        n_users=dataset.n_users, n_movies=dataset.n_movies,
        **model_params
    )

    print(f'=== Running training')
    print(f'=== Model: {model}')
    print(f'=== Dataset: {dataset}')

    X, y = dataset.rating_samples()
    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y, test_size=test_ratio, random_state=random_seed
    )

    if calc_test_losses:
        model.fit(Xtrain, ytrain, Xtest, ytest)
    else:
        model.fit(Xtrain, ytrain)

    ytrain_pred = model.predict(Xtrain)
    final_mse_train = mean_squared_error(ytrain, ytrain_pred)

    ytest_pred = model.predict(Xtest)
    final_mse_test = mean_squared_error(ytest, ytest_pred)

    print(f'=== final_mse_train={final_mse_train:.3f}, '
          f'final_mse_test={final_mse_test:.3f}')

    # TODO: serialize losses to file


def run_cv(model_name, dataset_name, out_dir,
           splits, test_ratio, random_seed, jobs, **kw):
    model_params = {}
    cv_params = {}

    # Distinguish between parameters of CV (lists) and model parameters.
    for k, v in kw.items():
        if k in models.ALL_PARAMS:
            if isinstance(v, list):
                if len(v) == 1:
                    model_params[k] = v[0]
                else:
                    cv_params[k] = v
            else:
                model_params[k] = v

    dataset: data.MovieLensDataset = DATASETS[dataset_name]()
    model: models.MatrixCompletion = MODELS[model_name](
        n_users=dataset.n_users, n_movies=dataset.n_movies,
        **model_params
    )

    # Best was rank=10, eta=0.5

    print(f'=== Running cross-validation')
    print(f'=== Model: {model}')
    print(f'=== Dataset: {dataset}')
    print(f'=== CV parameters: {cv_params}')

    X, y = dataset.rating_samples()
    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y, test_size=test_ratio, random_state=random_seed
    )

    cv_scorer = make_scorer(mean_squared_error, greater_is_better=False)
    cv = GridSearchCV(
        model, cv_params,
        scoring=cv_scorer, cv=splits, n_jobs=jobs, verbose=4,
        return_train_score=True,
    )

    cv.fit(Xtrain, ytrain)

    print(f'=== best_params={cv.best_params_}')

    os.makedirs(out_dir, exist_ok=True)
    outfile = os.path.join(out_dir, f'{model_name}.tsv')
    print(f'=== Writing results to {outfile}...')
    pd.DataFrame(cv.cv_results_).to_csv(outfile, sep='\t')


if __name__ == '__main__':
    parsed_args = parse_cli(create_parser())
    print(parsed_args)
    parsed_args.subcmd_fn(**vars(parsed_args))
