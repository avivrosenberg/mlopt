import os
import time
import datetime as dt

import numpy as np
import matplotlib.pyplot as plt

import orps.data
import orps.models


def orps_train():
    # Load data
    ds = orps.data.EHazanPFDataset()
    X = ds.to_numpy()

    # Create asset returns
    R = (X[1:, :] / X[:-1, :])

    # Add short-asset returns
    R = np.hstack((R, 1 / R))

    T, d = R.shape

    # Create models
    models = {
        'OGD': orps.models.OGDOnlineRebalancingPortfolio(),
        'RFTL': orps.models.RFTLOnlineRebalancingPortfolio(),
        'Newton': orps.models.NewtonStepOnlineRebalancingPortfolio(),
        'Best Fixed': orps.models.BestFixedRebalancingPortfolio(
            continuous=True, max_iter=25),
        'Best Single': orps.models.BestSingleAssetRebalancingPortfolio(),
    }

    # Fit models
    for name, model in models.items():
        print(f'Training {name}... ', end='')
        t = time.time()
        model.fit(R)
        print(f'done in {time.time() - t:.3f} sec.')

    # Calculate wealth
    print('Calculating wealth...', end='')
    t = time.time()
    wealth = {}
    for name, model in models.items():
        wealth[name] = model.wealth(R)
    print(f'done in {time.time() - t:.3f} sec.')

    # Calculate regret
    print('Calculating regret...', end='')
    t = time.time()
    regret = {}
    Pstar = models['Best Fixed'].P_
    for name, model in models.items():
        if name.startswith("Best"):
            continue
        regret[name] = model.regret(R, Pstar, average=True)
    print(f'done in {time.time() - t:.3f} sec.')

    return models, wealth, regret


def orps_plot(wealth: dict, regret: dict):
    T, = next(iter(wealth.values())).shape
    t_axis = np.arange(start=1, stop=T + 1)

    # Plot wealth
    fig_wealth = plt.figure()
    for name, w in wealth.items():
        linespec = '--' if name.startswith("Best") else '-'
        plt.plot(t_axis, w, linespec, label=name)
    plt.title('Wealth')
    plt.yscale('log')
    plt.xlabel('time (days)')
    plt.legend()

    # Plot regret
    fig_regret = plt.figure()
    for name, r in regret.items():
        plt.plot(t_axis, r, label=name)
    # plt.plot(t_axis, np.log(t_axis)/t_axis, '--', label=r'$\log(t)/t$')
    # plt.plot(t_axis, np.sqrt(t_axis)/t_axis, '--', label=r'$\sqrt{t}/t$')
    plt.title('Average Regret')
    plt.xlabel('time (days)')
    plt.legend()

    timestamp = dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    out_dir = os.path.join('out', 'orps', timestamp)
    os.makedirs(out_dir, exist_ok=True)
    for fig in (fig_wealth, fig_regret):
        fmt = 'pdf'
        filename = os.path.join(out_dir, f'{fig.number}')
        fig.set_size_inches(8 * 0.8, 6 * 0.8)
        fig.savefig(f'{filename}.{fmt}', format=fmt,
                    bbox_inches='tight', pad_inches=0.1)

    plt.show()


if __name__ == '__main__':
    print('=== MLOPT HW3: Aviv Rosenberg & Yonatan Elul')
    print('=== ========================================')

    print('=== Runining online rebalancing portfolio selection...')
    models, wealth, regret = orps_train()
    orps_plot(wealth, regret)
