import time

import numpy as np
import matplotlib.pyplot as plt

import orps.data
import orps.models


def run_training():
    ds = orps.data.EHazanPFDataset()

    rounds = 1
    X = ds.to_numpy()
    X = np.vstack((X,) * rounds)

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
        'Best Fixed': orps.models.BestFixedRebalancingPortfolio(),
        'Best Single': orps.models.BestSingleAssetRebalancingPortfolio(),
    }

    # Fit models
    for name, model in models.items():
        print(f'Training {name}... ', end='')
        t = time.time()
        model.fit(R)
        print(f'done in {time.time()-t:.3f} sec.')


    # Plot wealth
    t_axis = np.arange(start=1, stop=T + 1)
    plt.figure()
    for name, model in models.items():
        linespec = '--' if name.startswith("Best") else '-'
        plt.plot(t_axis, model.wealth(R), linespec, label=name)
    plt.title('Wealth')
    plt.yscale('log')
    plt.xlabel('time (days)')
    plt.legend()

    # Plot regret
    plt.figure()
    for name, model in models.items():
        if name.startswith("Best"):
            continue
        plt.plot(t_axis, model.regret(models['Best Fixed'].p_), label=name)
    # plt.plot(t_axis, np.log(t_axis), '--', label=r'$\log(t)$')
    # plt.plot(t_axis, np.sqrt(t_axis), '--', label=r'$\sqrt{t}$')
    plt.title('Regret')
    plt.xlabel('time (days)')
    plt.legend()

    plt.show()


if __name__ == '__main__':
    run_training()
