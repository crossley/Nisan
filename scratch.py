import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.cluster import KMeans


def simulate_network(n_cells, w, I, time_params):

    tau = time_params['tau']
    T = time_params['T']
    t = time_params['t']
    n = time_params['n']

    A = 1
    t_peak = 2
    psp_amp = 1
    psp_decay = 100

    v = np.zeros((n_cells, n))
    u = np.zeros((n_cells, n))
    g = np.zeros((n_cells, n))
    spike = np.zeros((n_cells, n))

    # hold onto sum of all presynaptic outputs
    I_net = np.zeros((n_cells, n))

    # neuron parameters
    C = 50
    vr = -80
    vt = -25
    vpeak = 40
    k = 1
    a = 0.01
    b = -20
    c = -55
    d = 150

    # init v and u
    v[:, 0] = vr

    for i in range(1, n):

        dt = t[i] - t[i - 1]

        # iterate through postsynaptic neurons
        for jj in range(n_cells):

            # iterate through presynaptic neurons
            for kk in range(n_cells):
                if jj != kk:
                    I_net[jj, i - 1] += w[kk, jj] * g[kk, i - 1]

            dvdt = (k * (v[jj, i - 1] - vr) * (v[jj, i - 1] - vt) -
                    u[jj, i - 1] - I_net[jj, i - 1] + I[jj, i - 1]) / C
            dudt = a * (b * (v[jj, i - 1] - vr) - u[jj, i - 1])
            dgdt = (-g[jj, i - 1] + psp_amp * spike[jj, i - 1]) / psp_decay

            v[jj, i] = v[jj, i - 1] + dvdt * dt
            u[jj, i] = u[jj, i - 1] + dudt * dt
            g[jj, i] = g[jj, i - 1] + dgdt * dt

            if v[jj, i] >= vpeak:
                v[jj, i - 1] = vpeak
                v[jj, i] = c
                u[jj, i] = u[jj, i] + d
                spike[jj, i] = 1

    return t, n, v, g, spike


def plot_results(t, n, v, g, spike):
    # get spike times
    spike_times = []
    cmap = ['C0', 'C1', 'C2']
    for i in range(spike.shape[0]):
        spike_times.append(t[spike[i, :] == 1])

    # compute covaraince matrix on output g
    cormat_raw = np.corrcoef(g)

    # k-means cluster cov matrix
    X = cormat_raw
    kmeans = KMeans(n_clusters=3).fit(X)

    # sort g by k-means clusters and recompute cov matrix
    # NOTE: Seems like I ought to be able to directly sort the cov matrix but I
    # am not quite seeing how at the moment
    sort_inds = np.argsort(kmeans.labels_)
    g_sort = g[sort_inds, :]
    v_sort = v[sort_inds, :]
    spike_times_sort = [spike_times[i] for i in sort_inds]
    cormat_sort = np.corrcoef(g_sort)

    # plot raster and cov matrix
    fig, ax = plt.subplots(2, 2, squeeze=False)
    ax[0, 0].eventplot(spike_times, colors='k', lineoffsets=1, linelengths=0.1)
    ax[1, 0].eventplot(spike_times_sort,
                       colors='k',
                       lineoffsets=1,
                       linelengths=0.1)
    ax[0, 1].imshow(cormat_raw, origin='lower', aspect='equal')
    ax[1, 1].imshow(cormat_sort, origin='lower', aspect='equal')
    ax[0, 0].set_xlim(0, t.max())
    ax[1, 0].set_xlim(0, t.max())
    plt.show()

    # hold useful data in a data frame for convenience in later steps
    neuron = np.repeat(np.arange(0, n_cells, 1), n)
    cluster = np.repeat(kmeans.labels_, n)
    d = pd.DataFrame({
        'neuron': neuron,
        'cluster': cluster,
        'g': g.flatten(),
        'spike': spike.flatten(),
        't': np.tile(t, n_cells)
    })
    # dd = d.iloc[::100, :]
    # sns.lineplot(data=dd, x='t', y='g', hue='neuron')
    # plt.show()

    # # plot v and g coloured by cluster
    # fig, ax = plt.subplots(n_cells, 2, squeeze=False)
    # cmap = ['C0', 'C1', 'C2']
    # for i in range(n_cells):
    #     ax[i, 0].plot(t, v[i, :], color=cmap[kmeans.labels_[i]])
    #     ax[i, 1].plot(t, g[i, :], color=cmap[kmeans.labels_[i]])
    # plt.show()

    # figure 5
    fig, ax = plt.subplots(2, 2, squeeze=False)
    cmap = ['C0', 'C1', 'C2']
    ax[0, 0].eventplot(spike_times_sort,
                       colors=[cmap[x] for x in kmeans.labels_[sort_inds]],
                       lineoffsets=1,
                       linelengths=0.1)
    dd = d.groupby(['cluster', 't'])['g'].mean().reset_index()
    dd['cluster'] = dd['cluster'].astype('category')
    dd = dd.iloc[::100, :]
    sns.lineplot(data=dd,
                 x='t',
                 y='g',
                 hue='cluster',
                 legend=None,
                 ax=ax[1, 0])
    ax[0, 1].imshow(cormat_sort, origin='lower', aspect='equal')
    gg = g[:, ::100]
    ptm = np.dot(gg.T, gg)
    ax[1, 1].imshow(ptm, origin='lower', aspect='equal')
    ax[0, 0].set_xlim(0, t.max())
    ax[1, 0].set_xlim(0, t.max())
    plt.show()


tau = 0.1
T = 30000
t = np.arange(0, T, tau)
n = t.shape[0]
time_params = {'tau': tau, 'T': T, 't': t, 'n': n}

n_cells = 20

ibif = 325

# NOTE: fixed input
I = np.random.uniform(ibif, ibif + 1, (n_cells, 1))
I = np.tile(I, n)
# for i in range(I.shape[0]):
#     plt.plot(t, I[i, :])
# plt.show()

# NOTE: fluctuating input (changes every 10 ms ~ 10 ms / tau ms / sample)
# I = np.random.uniform(ibif, ibif + 1, (n_cells, int(10 / tau)))
# I = np.repeat(I, n // I.shape[1], axis=1)
# for i in range(I.shape[0]):
#     plt.plot(t, I[i, :])
# plt.show()

# NOTE: weakly interconnected
p = 0.15

# NOTE: strongly interconnected
p = 0.85

k = 1e-4
w = np.random.uniform(0, 1, (n_cells, n_cells))
w = (w < p).astype(int)
w = 0
eps = np.random.uniform(0.8, 1.2, (n_cells, n_cells))
w = (k / p) * eps

t, n, v, g, spike = simulate_network(n_cells, w, I, time_params)
plot_results(t, n, v, g, spike)
