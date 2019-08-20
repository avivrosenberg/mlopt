import numpy as np
import matplotlib.pyplot as plt

from orps.data import EHazanPFDataset
from orps.models import OGDOnlineRebalancingPortfolio, \
    RFTLOnlineRebalancingPortfolio, NewtonStepOnlineRebalancingPortfolio


def run_training():
    ds = EHazanPFDataset()

    rounds = 1
    X = ds.to_numpy()
    X = np.vstack((X,) * rounds)

    R = (X[1:, :] / X[:-1, :])
    T, d = R.shape

    # Fit models
    m_ogd = OGDOnlineRebalancingPortfolio()
    m_ogd.fit(R)

    m_rftl = RFTLOnlineRebalancingPortfolio()
    m_rftl.fit(R)

    m_newton = NewtonStepOnlineRebalancingPortfolio()
    m_newton.fit(R)

    # Best single asset
    idx_best_single = np.argmax(np.prod(R, axis=0))
    p_best_single = np.zeros((d,), dtype=np.float32)
    p_best_single[idx_best_single] = 1.
    wealth_best_single = m_ogd.wealth(R, p_best_single)

    # Find best fixed portfolio
    def find_best_fixed(models, R):
        best_fixed = None
        best_wealth = 0
        for i, m in enumerate(models):
            for j, p in enumerate(m.P_):
                wealth = m.wealth(R, p)[-1]
                if wealth > best_wealth:
                    best_wealth = wealth
                    best_fixed = p
                    ind = (i, j)
        return best_fixed, ind

    p_best_fixed, i_best_fixed = find_best_fixed([m_ogd, m_rftl, m_newton], R)
    print(p_best_fixed)
    print(i_best_fixed)
    print(np.sum(p_best_fixed))
    wealth_best_fixed = m_ogd.wealth(R, p_best_fixed)

    t_axis = np.arange(start=1, stop=T + 1)

    # Plot wealth
    plt.figure()
    plt.plot(t_axis, m_ogd.wealth(R), label='OGD')
    plt.plot(t_axis, m_rftl.wealth(R), label='RFTL')
    plt.plot(t_axis, m_newton.wealth(R), label='Newton')
    plt.plot(t_axis, wealth_best_single, label='Best Single')
    plt.plot(t_axis, wealth_best_fixed, label='Best Fixed')
    plt.title('Wealth')
    plt.yscale('log')
    plt.xlabel('time (days)')
    plt.legend()

    # Plot regret
    plt.figure()
    plt.plot(t_axis, m_ogd.regret(p_best_fixed), label='OGD')
    plt.plot(t_axis, m_rftl.regret(p_best_fixed), label='RFTL')
    plt.plot(t_axis, m_newton.regret(p_best_fixed), label='Newton')
    # plt.plot(t_axis, np.log(t_axis), '--', label=r'$\log(t)$')
    # plt.plot(t_axis, np.sqrt(t_axis), '--', label=r'$\sqrt{t}$')
    plt.title('Regret')
    plt.xlabel('time (days)')
    plt.legend()

    print(f'eta(OGD)={m_ogd.eta_}')
    print(f'eta(RFTL)={m_rftl.eta_}')
    plt.show()


if __name__ == '__main__':
    run_training()
