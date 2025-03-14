# ----------------------------------------------------------------------
# Project Name: PiNet - Prior Information based NEural neTwork
# Description : A physics informed learning based framework for constitutive modelling and BVPs
# Author      : Pin ZHANG, National University of Singapore
# Contact     : pinzhang@nus.edu.sg
# Created On  : 14 Mar 2025
# Repository  : https://github.com/PinZhang3/PiNet
# ----------------------------------------------------------------------
# Notes:
# This library is under active development. Contributions are welcome!
# Copyright belongs to Pin ZHANG and use of this code for commercial applications or
# profit-driven ventures requires explicit permission from the author(s)
# ----------------------------------------------------------------------

import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
from NN import PiNet

def von_Mises(sig, sig_bar):
    return np.abs(sig) - sig_bar
###############################################################################
################################ Main Function ################################
###############################################################################

random.seed(2022)
store_path = 'Output/'
sig_y = 100.  # kPa
E = 10000.    # kPa
h = E
eps = 0.
sig = 0.
N_step = 200
N_periodic = 800
e_tol = 1E-6

eps_all = []
lambd_all = []
sig_all = []
sig_y_all = []

# model
layer  = [3, 32, 32, 1]
N_epoch= 5000   # Adam training epoch
N_grad = 0      # show distribution of gradients per N_grad
model  = PiNet(E, h, layer, N_grad, store_path)
knob   = 0
lambd = 0.
deps_p = 0.
for i in range(N_step):
    deps = 0.01 / 100 * np.sign(np.cos(2 * np.pi * i / N_periodic))
    eps += deps
    sig_hist = sig
    eps_hist = eps
    sig = sig_hist + E * deps
    f = von_Mises(sig, sig_y)
    if f > e_tol:
        print('enter plasticity')
        sig_hist = np.array([sig_hist]).reshape(1, 1)
        eps_hist = np.array([eps_hist]).reshape(1, 1)
        deps = np.array([deps]).reshape(1, 1)

        model.train(N_epoch, sig_hist, eps_hist, deps, sig_y)
        sig, sig_y, deps_p = model.predict(sig_hist, eps_hist, deps, sig_y)

        sig = sig[0][0]
        sig_y = sig_y[0][0]
        deps_p = deps_p[0][0]

    lambd += np.abs(deps_p)
    eps_all.append(eps)
    lambd_all.append(lambd)
    sig_all.append(sig)
    sig_y_all.append(sig_y)

#========================================== Save data ==========================================
# stress_strain_NN = np.hstack((np.array(eps_all).reshape(len(eps_all), 1),
#                        np.array(eps_p_all).reshape(len(eps_p_all), 1),
#                        np.array(sig_all).reshape(len(sig_all), 1),
#                        np.array(sig_bar_all).reshape(len(sig_bar_all), 1)))
loss = model.loss_log
# np.savetxt(store_path+'loss_NN.csv', loss, fmt='%0.10f', delimiter=',')

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
plt.plot(eps_all, lambd_all, linestyle='none', color='r', alpha=0.25, label='Exact',
                 marker='o', markerfacecolor='none', markeredgewidth=0.5, markersize=8)
plt.tick_params(direction='in')

fig_1.add_subplot(1, 3, 3)
plt.plot(eps_all, sig_y_all, linestyle='none', color='r', alpha=0.25, label='Exact',
                 marker='o', markerfacecolor='none', markeredgewidth=0.5, markersize=8)
plt.tick_params(direction='in')

plt.savefig('stress-strain-nn.png', dpi = 400)

plt.show()