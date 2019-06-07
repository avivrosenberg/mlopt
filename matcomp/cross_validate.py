import matcomp.data as data
import matcomp.models as models

from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split

import pandas as pd


def cross_validate(model, param_grid, dataset, cv_splits=4):
    X, y = dataset.rating_samples()
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=1/3.)

    cv_scorer = make_scorer(mean_squared_error, greater_is_better=False)
    cv = GridSearchCV(
        model, param_grid,
        scoring=cv_scorer, cv=cv_splits, n_jobs=4, verbose=4,
    )

    print('params:')
    for param, value in cv.get_params().items():
        print(f'{param}: {value}')

    cv.fit(Xtrain, ytrain)

    print(f'best_params={cv.best_params_}')

    outfile = f'out/{model.name}.tsv'
    print(f'writing results to {outfile}...')
    pd.DataFrame(cv.cv_results_).to_csv(outfile, sep='\t')


def cross_validate_rpmc():
    dataset = data.MovieLens100K()

    model = models.RankProjectionMatrixCompletion(
        n_users=dataset.n_users, n_movies=dataset.n_movies,
        max_iter=1000, verbose=True,

    )

    param_grid = {
        'rank': [10, 15, 20, 25],
        'eta': [0.5, 1.0],
    }

    cross_validate(model, param_grid, dataset,)
