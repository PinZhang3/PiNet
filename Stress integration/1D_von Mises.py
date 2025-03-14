import numpy as np
import matplotlib.pyplot as plt

def von_Mises(sig, sig_bar):
    return np.abs(sig) - sig_bar

sig_bar = 100.  # kPa
E = 10000.      # kPa
h = E
eps_p = 0.
eps = 0.
sig = 0.
N_step = 800
N_periodic = 800
e_tol = 1E-6

eps_all = []
lambda_all = []
sig_all = []
sig_bar_all = []

for i in range(N_step):
    deps = 0.01 / 100 * np.sign(np.cos(2 * np.pi * i / N_periodic))  # %
    deps_p = 0.
    delta_lambda = 0
    eps += deps
    sig += E * deps
    f = von_Mises(sig, sig_bar)
    df_dsig = np.sign(sig)
    if f > e_tol:
        delta_lambda = f/(E+h)
        deps_p = delta_lambda * df_dsig
        sig = sig - E * deps_p
        sig_bar = sig_bar + h * delta_lambda

    delta_lambda += delta_lambda
    eps_all.append(eps)
    lambda_all.append(delta_lambda)
    sig_all.append(sig)
    sig_bar_all.append(sig_bar)

rc = {"font.family": "serif", "mathtext.fontset": "stix", "font.size": 16}
plt.rcParams.update(rc)
plt.rcParams["font.serif"] = ["Times New Roman"]

fig_1 = plt.figure(1, figsize=(16, 5))
plt.subplots_adjust(left=0.05, bottom=None, right=0.95, top=None, wspace=0.22, hspace=None)

ax = fig_1.add_subplot(1, 3, 1)
plt.plot(eps_all, sig_all, linestyle='none', color='r', alpha=0.25, label='Exact',
                 marker='o', markerfacecolor='none', markeredgewidth=0.5, markersize=8)
plt.tick_params(direction='in')
plt.legend(frameon=False)
ax.set_xlabel('$\epsilon$')
ax.set_ylabel('$\sigma$')

ax = fig_1.add_subplot(1, 3, 2)
ax.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
plt.plot(eps_all, lambda_all, linestyle='none', color='r', alpha=0.25, label='Exact',
                 marker='o', markerfacecolor='none', markeredgewidth=0.5, markersize=8)
plt.tick_params(direction='in')

fig_1.add_subplot(1, 3, 3)
plt.plot(eps_all, sig_bar_all, linestyle='none', color='r', alpha=0.25, label='Exact',
                 marker='o', markerfacecolor='none', markeredgewidth=0.5, markersize=8)
plt.tick_params(direction='in')

plt.savefig('stress-strain.png', dpi = 400)

plt.show()








