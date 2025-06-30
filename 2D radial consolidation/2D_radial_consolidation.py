# ----------------------------------------------------------------------
# Project Name: Data-Driven Modelling & Computation
# Description : This code is for solving 2D radial consolidation equation
# Author      : Pin ZHANG, National University of Singapore
# Contact     : pinzhang@nus.edu.sg
# Created On  : 30 Jun 2025
# Repository  : https://github.com/PinZhang3/Data-Driven-Modelling-and-Computation
# ----------------------------------------------------------------------
# Notes:
# This library is under active development. Contributions are welcome!
# Copyright belongs to Pin ZHANG and use of this code for commercial applications or
# profit-driven ventures requires explicit permission from the author(s)
# ----------------------------------------------------------------------

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import random
from scipy.interpolate import griddata
from pyDOE import lhs

class PiNet:

    def __init__(self, Ini, Inner, Extern, Top, Bot, X_f, layer,
                 lb_input, ub_input, lb_output, ub_output,
                 norm):
        
        # domain boundary
        self.lb_input_r = np.array([lb_input[0], lb_input[1]])
        self.ub_input_r = np.array([ub_input[0], ub_input[1]])

        self.lb_input_z = np.array([lb_input[0], lb_input[2]])
        self.ub_input_z = np.array([ub_input[0], ub_input[2]])

        self.lb_output = lb_output
        self.ub_output = ub_output

        # Architecture
        self.layer = layer
        self.output_norm = norm

        # Init
        self.nn_init(Ini, Inner, Extern, Top, Bot, X_f)
        
        # tf session
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        init = tf.global_variables_initializer()
        self.sess.run(init)

        self.saver_r.save(self.sess, "Ini_wb/NN_wb_ini_r.ckpt")
        self.saver_z.save(self.sess, "Ini_wb/NN_wb_ini_z.ckpt")
        # self.saver_r.restore(self.sess, "../NN_wb/NN_wb_ini_r.ckpt")
        # self.saver_z.restore(self.sess, "../NN_wb/NN_wb_ini_z.ckpt")
    
    def nn_init(self, Ini, Inner, Extern, Top, Bot, X_f):

        # training data
        self.t0 = Ini[:, 0:1]    # Initial Data (time)
        self.r0 = Ini[:, 1:2]    # Initial Data (space)
        self.z0 = Ini[:, 2:3]    # Initial Data (studied variable)
        self.ur0 = Ini[:, 3:4]   # Initial Data (studied variable)
        self.uz0 = Ini[:, 4:5]   # Initial Data (studied variable)

        self.ti = Inner[:, 0:1]  # Internal Data (time)
        self.ri = Inner[:, 1:2]  # Internal Data (space)
        self.ui = Inner[:, 2:3]  # Internal Data (studied variable)

        self.te = Extern[:, 0:1]  # External Data (time)
        self.re = Extern[:, 1:2]  # External Data (space)

        self.tt = Top[:, 0:1]
        self.zt = Top[:, 1:2]
        self.ut = Top[:, 2:3]

        self.tb = Bot[:, 0:1]
        self.zb = Bot[:, 1:2]

        self.t_f = X_f[:, 0:1]   # Collocation Points (time)
        self.r_f = X_f[:, 1:2]   # Collocation Points (space)
        self.z_f = X_f[:, 2:3]   # Collocation Points (space)
        self.X_dim = X_f.shape[1]
        
        # initialize NNs for solution
        self.weights_ur, self.biases_ur = self.initialize_nn(self.layer)
        self.weights_uz, self.biases_uz = self.initialize_nn(self.layer)

        self.saver_r = tf.train.Saver(var_list= [self.weights_ur[l] for l in range(len(self.layer) - 1)]
                                            + [self.biases_ur[l] for l in range(len(self.layer) - 1)])
        self.saver_z = tf.train.Saver(var_list=[self.weights_uz[l] for l in range(len(self.layer) - 1)]
                                               + [self.biases_uz[l] for l in range(len(self.layer) - 1)])

        # tf placeholders
        self.t0_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.r0_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.z0_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.ur0_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.uz0_tf = tf.placeholder(tf.float32, shape=[None, 1])

        self.ti_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.ri_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.ui_tf = tf.placeholder(tf.float32, shape=[None, 1])

        self.te_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.re_tf = tf.placeholder(tf.float32, shape=[None, 1])

        self.tt_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.zt_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.ut_tf = tf.placeholder(tf.float32, shape=[None, 1])

        self.tb_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.zb_tf = tf.placeholder(tf.float32, shape=[None, 1])

        self.t_f_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.r_f_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.z_f_tf = tf.placeholder(tf.float32, shape=[None, 1])

        # radial
        self.ur0_pred, _, _ = self.net_ur(self.t0_tf, self.r0_tf)
        self.ui_pred, _, _ = self.net_ur(self.ti_tf, self.ri_tf)
        _, self.u_r_pred, _ = self.net_ur(self.te_tf, self.re_tf)
        _, _, self.fr_pred = self.net_ur(self.t_f_tf, self.r_f_tf)

        # vertical
        self.uz0_pred, _, _ = self.net_uz(self.t0_tf, self.z0_tf)
        self.ut_pred, _, _ = self.net_uz(self.tt_tf, self.zt_tf)
        _, self.u_z_pred, _ = self.net_uz(self.tb_tf, self.zb_tf)
        _, _, self.fz_pred = self.net_uz(self.t_f_tf, self.z_f_tf)

        # loss
        if self.output_norm == 'on':
            "normalization loss"
            ur0_out_tf_nr = self.min_max_norm(self.ur0_tf)
            ur0_out_pred_nr = self.min_max_norm(self.ur0_pred)
            uz0_out_tf_nr = self.min_max_norm(self.uz0_tf)
            uz0_out_pred_nr = self.min_max_norm(self.uz0_pred)
            ui_out_tf_nr = self.min_max_norm(self.ui_tf)
            ui_out_pred_nr = self.min_max_norm(self.ui_pred)
            ut_out_tf_nr = self.min_max_norm(self.ut_tf)
            ut_out_pred_nr = self.min_max_norm(self.ut_pred)

            self.loss_1 = tf.reduce_mean(tf.square(ur0_out_tf_nr - ur0_out_pred_nr))
            self.loss_2 = tf.reduce_mean(tf.square(uz0_out_tf_nr - uz0_out_pred_nr))
            self.loss_3 = tf.reduce_mean(tf.square(ui_out_tf_nr - ui_out_pred_nr))
            self.loss_4 = tf.reduce_mean(tf.square(ut_out_tf_nr - ut_out_pred_nr))
        else:
            self.loss_1 = tf.reduce_mean(tf.square(self.ur0_tf - self.ur0_pred))
            self.loss_2 = tf.reduce_mean(tf.square(self.uz0_tf - self.uz0_pred))
            self.loss_3 = tf.reduce_mean(tf.square(self.ui_tf - self.ui_pred))
            self.loss_4 = tf.reduce_mean(tf.square(self.ut_tf - self.ut_pred))

        self.loss_5 = tf.reduce_mean(tf.square(self.u_r_pred))
        self.loss_6 = tf.reduce_mean(tf.square(self.u_z_pred))
        self.loss_7 = tf.reduce_mean(tf.square(self.fr_pred))
        self.loss_8 = tf.reduce_mean(tf.square(self.fz_pred))

        self.loss_k = self.loss_1 + self.loss_2 + self.loss_3 + self.loss_4 + self.loss_5 + self.loss_6
        self.loss_f = self.loss_7 + self.loss_8

        self.loss = self.loss_k + self.loss_f

        # record loss
        self.loss_k_log = []
        self.loss_f_log = []

        # optimization
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                             method='L-BFGS-B',
                             options={'maxiter': 10000,
                                      'maxfun': 10000,
                                      'maxcor': 100,
                                      'maxls': 100,
                                      'gtol': 1e-04})

        self.optimizer_Adam = tf.train.AdamOptimizer()    # default learning rate = 0.001
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)

    def initialize_nn(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev, dtype=tf.float32),
                           dtype=tf.float32)

    def neural_network(self, X, weights, biases):
        num_layers = len(weights) + 1
        H = X
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.sin(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_ur(self, t, r):
        X = tf.concat([t, r], 1)
        H = 2.0 * (X - self.lb_input_r) / (self.ub_input_r - self.lb_input_r) - 1.0
        u = self.neural_network(H, self.weights_ur, self.biases_ur)
        u_r = tf.gradients(u, r)[0]
        u_rr = tf.gradients(u_r, r)[0]
        u0 = 50.0
        re = 2.5  # m
        rw = 0.2  # m
        ch = 25.0/365.0
        n = re / rw
        Fn = n ** 2 / (n**2 - 1.0) * tf.math.log(n) - (3*n**2 - 1.0) / (4.0*n**2)
        Th = ch * t/ (4.0*re**2)
        lamb = -8.0*Th/Fn
        u_bar = u0*tf.math.exp(lamb)
        u_bar_t = tf.gradients(u_bar, t)[0]
        f = u_bar_t - ch * (u_r / r + u_rr)
        # f = u_t-ch*(u_r/r+u_rr)
        return u, u_r, f

    def net_uz(self, t, z):
        X = tf.concat([t, z], 1)
        H = 2.0 * (X - self.lb_input_z) / (self.ub_input_z - self.lb_input_z) - 1.0
        u = self.neural_network(H, self.weights_uz, self.biases_uz)
        ch = 25.0 / 365.0
        cv = ch/2.0
        u_t = tf.gradients(u, t)[0]
        u_z = tf.gradients(u, z)[0]
        u_zz = tf.gradients(u_z, z)[0]
        f = u_t - cv * u_zz
        return u, u_z, f

    def min_max_norm(self, u):
        u_norm = 2.0 * (u - self.lb_output) / (self.ub_output - self.lb_output) - 1.0
        return u_norm
    
    def callback(self, loss_k_value, loss_f_value):
        # print('loss_k_value: %e,loss_r_value: %e' % (loss_k_value, loss_f_value))
        self.loss_k_log.append(loss_k_value)
        self.loss_f_log.append(loss_f_value)
        
    def train(self, N_iter, N_grad):
        tf_dict = {self.t0_tf: self.t0, self.r0_tf: self.r0, self.z0_tf: self.z0, self.ur0_tf: self.ur0, self.uz0_tf: self.uz0,
                   self.tt_tf: self.tt, self.zt_tf: self.zt, self.ut_tf: self.ut,
                   self.tb_tf: self.tb, self.zb_tf: self.zb,
                   self.ti_tf: self.ti, self.ri_tf: self.ri, self.ui_tf: self.ui,
                   self.te_tf: self.te, self.re_tf: self.re,
                   self.t_f_tf: self.t_f, self.r_f_tf: self.r_f, self.z_f_tf: self.z_f}

        self.loss_history = []
        for it in range(N_iter):
            self.sess.run(self.train_op_Adam, tf_dict)
            loss_k_value, loss_f_value = self.sess.run([self.loss_k, self.loss_f], tf_dict)
            self.loss_k_log.append(loss_k_value)
            self.loss_f_log.append(loss_f_value)

            if it % (N_grad) == 0:
                print('It: %d, loss_k_value: %e,loss_f_value: %e' % (it, loss_k_value, loss_f_value))

        self.optimizer.minimize(self.sess,
                                    feed_dict=tf_dict,
                                    fetches=[self.loss_k, self.loss_f],
                                    loss_callback=self.callback)

        self.saver_r.save(self.sess, "Final_wb/paras_NN_r.ckpt")
        self.saver_z.save(self.sess, "Final_wb/paras_NN_z.ckpt")
    
    def predict(self, t, r, z):
        ur_pred = self.sess.run(self.ur0_pred, {self.t0_tf: t, self.r0_tf: r})
        uz_pred = self.sess.run(self.uz0_pred, {self.t0_tf: t, self.z0_tf: z})
        return ur_pred, uz_pred

###############################################################################
################################ Main Function ################################
###############################################################################

if __name__ == "__main__":
    random.seed(2025)
    # parameters
    u0 = 50.0
    H = 5.0       # m
    re = 2.5      # m
    rw = 0.2      # m
    T = 200       # day

    # load data #
    data_z = scipy.io.loadmat('Exact/VC_1D.mat')
    data_r = scipy.io.loadmat('Exact/RC_1D.mat')

    uz_exact = np.real(data_z['u'])
    ur_exact = np.real(data_r['u'])

    N_T = 51
    N_R = 51
    N_Z = 51

    t = np.linspace(0, T, N_T)
    r = np.linspace(rw, re, N_R)
    z = np.linspace(0, H, N_Z)

    # domain bounds
    lb_input = np.array([np.min(t), np.min(r), np.min(z)])
    ub_input = np.array([np.max(t), np.max(r), np.max(z)])
    lb_output = np.array([0])
    ub_output = np.array([u0])

    # initial and boundary conditions
    t_m, r_m = np.meshgrid(t, r)
    t_m, z_m = np.meshgrid(t, z)

    t_pred = t_m.flatten()[:, None]
    r_pred = r_m.flatten()[:, None]
    z_pred = z_m.flatten()[:, None]
    Xr_pred = np.hstack((t_pred, r_pred))
    Xz_pred = np.hstack((t_pred, z_pred))
    X_pred = np.hstack((t_pred, r_pred, z_pred))

    # initial t=0
    Ini = np.zeros((N_R, 5))
    Ini[:, 1] = r
    Ini[:, 2] = z
    Ini[0, 3] = 0
    Ini[1:, 3] = u0
    Ini[0,  4] = 0
    Ini[1:, 4] = u0

    # boundary-inner
    Inner = np.zeros((N_T, 3))
    Inner[:, 0] = t
    Inner[:, 1] = rw
    Inner[:, 2] = 0.0

    # boundary-external
    Extern = np.zeros((N_T, 2))
    Extern[:, 0] = t
    Extern[:, 1] = re

    # boundary-top
    Top = np.zeros((N_T, 3))
    Top[:, 0] = t
    Top[:, 1] = 0.0
    Top[:, 2] = 0.0

    # boundary-bottom
    Bot = np.zeros((N_T, 2))
    Bot[:, 0] = t
    Bot[:, 1] = H

    # PDE constraints
    N_f = 10000
    sampling_mode = 'random'
    if sampling_mode == 'uniform':
        sample = X_pred
        X_f_train = sample
    else:
        sample = lb_input + (ub_input - lb_input) * lhs(3, N_f)
        X_f_train = sample

    # architecture
    layer = [2, 32, 32, 32, 32, 1]

    # model
    norm = 'on'    # predicted output normalization
    model = PiNet(Ini, Inner, Extern, Top, Bot, X_f_train, layer,
                    lb_input, ub_input, lb_output, ub_output, norm)
    
    # train the solver
    N_epoch = 5000 # Adam training epoch
    N_grad  = 1000 # Print results
    model.train(N_epoch, N_grad)
    ur_pred, uz_pred = model.predict(t_pred, r_pred, z_pred)
    ur_pred = griddata(Xr_pred, ur_pred.flatten(), (t_m, r_m), method='cubic')
    uz_pred = griddata(Xz_pred, uz_pred.flatten(), (t_m, z_m), method='cubic')

    ############################# plotting ###############################

    rc = {"font.family": "serif", "mathtext.fontset": "stix", "font.size": 16}
    plt.rcParams.update(rc)
    plt.rcParams["font.serif"] = ["Times New Roman"]

    #################################### profiles of uz and ur  ###################################

    fig_1 = plt.figure(1, figsize=(11, 7))
    fig_1.add_subplot(2, 3, 1)
    plt.pcolor(t_m / T, r_m / re, ur_exact / u0, cmap='jet')
    plt.gca().invert_yaxis()
    plt.xlabel('$t_n$')
    plt.ylabel('$r_n$')
    plt.title('Exact $u_{rn}(t,r)$', fontsize=16)
    plt.tick_params(direction='in')
    plt.colorbar(orientation="horizontal", pad = 0.22)
    plt.clim(0.0, 1.0)
    plt.tight_layout()

    fig_1.add_subplot(2, 3, 2)
    plt.pcolor(t_m / T, r_m / re, ur_pred / u0, cmap='jet')
    plt.gca().invert_yaxis()
    plt.xlabel('$t_n$')
    plt.ylabel('$r_n$')
    plt.title('Predicted $u^p_{rn}(t,r)$', fontsize=16)
    plt.tick_params(direction='in')
    plt.colorbar(orientation="horizontal", pad = 0.22)
    plt.clim(0.0, 1.0)
    plt.tight_layout()

    fig_1.add_subplot(2, 3, 3)
    relative_error = np.abs(ur_pred[1:, :] - ur_exact[1:, :]) / ur_exact[1:, :] * 100
    mean_error_r = np.mean(relative_error)
    plt.pcolor(t_m[1:, :] / T, r_m[1:, :] / re, relative_error, cmap='jet')
    plt.gca().invert_yaxis()
    plt.xlabel('$t_n$')
    plt.ylabel('$z_n$')
    plt.title('$|u_{rn}-u^p_{rn}|/u_{rn}*100%$', fontsize=16)
    plt.tick_params(direction='in')
    plt.colorbar(orientation="horizontal", pad=0.22)
    plt.tight_layout()

    fig_1.add_subplot(2, 3, 4)
    plt.pcolor(t_m/ T, z_m / H, uz_exact / u0, cmap='jet')
    plt.gca().invert_yaxis()
    plt.xlabel('$t_n$')
    plt.ylabel('$z_n$')
    plt.title('Exact $u_{zn}(t,z)$', fontsize=16)
    plt.tick_params(direction='in')
    plt.colorbar(orientation="horizontal", pad=0.22)
    plt.clim(0.0, 1.0)
    plt.tight_layout()

    fig_1.add_subplot(2, 3, 5)
    plt.pcolor(t_m / T, z_m / H, uz_pred / u0, cmap='jet')
    plt.gca().invert_yaxis()
    plt.xlabel('$t_n$')
    plt.ylabel('$z_n$')
    plt.title('Predicted $u^p_{zn}(t,z)$', fontsize=16)
    plt.tick_params(direction='in')
    plt.colorbar(orientation="horizontal", pad=0.22)
    plt.clim(0.0, 1.0)
    plt.tight_layout()

    fig_1.add_subplot(2, 3, 6)
    relative_error = np.abs(uz_pred[1:, :] - uz_exact[1:, :]) / uz_exact[1:, :] * 100
    mean_error_z = np.mean(relative_error)
    plt.pcolor(t_m[1:, :] / T, z_m[1:, :] / H, relative_error, cmap='jet')
    plt.gca().invert_yaxis()
    plt.xlabel('$t_n$')
    plt.ylabel('$z_n$')
    plt.title('$|u_{zn}-u^p_{zn}|/u_{zn}*100%$', fontsize=16)
    plt.tick_params(direction='in')
    plt.colorbar(orientation="horizontal", pad=0.22)
    plt.tight_layout()

    #################################### 2D contour at two specific time ###################################

    fig_2 = plt.figure(2, figsize=(11, 7))
    rm, zm = np.meshgrid(r, z)
    N_snapshots = [5, 10]
    u_pred_snapshot = np.zeros((N_Z, N_R))
    u_exact_snapshot = np.zeros((N_Z, N_R))
    for i in range(N_Z):
        for j in range(N_R):
            u_pred_snapshot[i][j] = ur_pred[j][N_snapshots[0]]*uz_pred[i][N_snapshots[0]]/u0
            u_exact_snapshot[i][j] = ur_exact[j][N_snapshots[0]]*uz_exact[i][N_snapshots[0]]/u0

    fig_2.add_subplot(2, 3, 1)
    plt.pcolor(rm / re, zm / H, u_exact_snapshot / u0, cmap='jet')
    plt.gca().invert_yaxis()
    plt.xlabel('$r_n$')
    plt.ylabel('$z_n$')
    plt.title('Exact $u_n(r,z)$', fontsize=16)
    plt.tick_params(direction='in')
    plt.colorbar(orientation="horizontal", pad=0.22)
    plt.clim(0.0, 1.0)
    plt.tight_layout()

    fig_2.add_subplot(2, 3, 2)
    plt.pcolor(rm / re, zm / H, u_pred_snapshot / u0, cmap='jet')
    plt.gca().invert_yaxis()
    plt.xlabel('$r_n$')
    plt.ylabel('$z_n$')
    plt.title('Predicted $u^p_n(r,z)$', fontsize=16)
    plt.tick_params(direction='in')
    plt.colorbar(orientation="horizontal", pad=0.22)
    plt.clim(0.0, 1.0)
    plt.tight_layout()

    fig_2.add_subplot(2, 3, 3)
    relative_error = np.divide(np.abs(u_pred_snapshot[1:, 1:] - u_exact_snapshot[1:, 1:]),
                               u_exact_snapshot[1:, 1:]) * 100
    plt.pcolor(rm[1:, 1:] / re, zm[1:, 1:] / H, relative_error, cmap='jet')
    plt.gca().invert_yaxis()
    plt.xlabel('$r_n$')
    plt.ylabel('$z_n$')
    plt.title('$|u_n-u^p_n|/u_n*100%$', fontsize=16)
    plt.tick_params(direction='in')
    plt.colorbar(orientation="horizontal", pad=0.22)
    # plt.clim(0.0, 1.0)
    plt.tight_layout()

    u_pred_snapshot = np.zeros((N_Z, N_R))
    u_exact_snapshot = np.zeros((N_Z, N_R))
    for i in range(N_Z):
        for j in range(N_R):
            u_pred_snapshot[i][j] = ur_pred[j][N_snapshots[1]]*uz_pred[i][N_snapshots[1]]/u0
            u_exact_snapshot[i][j] = ur_exact[j][N_snapshots[1]]*uz_exact[i][N_snapshots[1]]/u0

    fig_2.add_subplot(2, 3, 4)
    plt.pcolor(rm / re, zm / H, u_exact_snapshot / u0, cmap='jet')
    plt.gca().invert_yaxis()
    plt.xlabel('$r_n$')
    plt.ylabel('$z_n$')
    plt.title('Exact $u_n(r,z)$', fontsize=16)
    plt.tick_params(direction='in')
    plt.colorbar(orientation="horizontal", pad=0.22)
    plt.clim(0.0, 1.0)
    plt.tight_layout()

    fig_2.add_subplot(2, 3, 5)
    plt.pcolor(rm / re, zm / H, u_pred_snapshot / u0, cmap='jet')
    plt.gca().invert_yaxis()
    plt.xlabel('$r_n$')
    plt.ylabel('$z_n$')
    plt.title('Predicted $u^p_n(r,z)$', fontsize=16)
    plt.tick_params(direction='in')
    plt.colorbar(orientation="horizontal", pad=0.22)
    plt.clim(0.0, 1.0)
    plt.tight_layout()

    fig_2.add_subplot(2, 3, 6)
    relative_error = np.divide(np.abs(u_pred_snapshot[1:, 1:] - u_exact_snapshot[1:, 1:]),
                               u_exact_snapshot[1:, 1:]) * 100
    plt.pcolor(rm[1:, 1:] / re, zm[1:, 1:] / H, relative_error, cmap='jet')
    plt.gca().invert_yaxis()
    plt.xlabel('$r_n$')
    plt.ylabel('$z_n$')
    plt.title('$|u_n-u^p_n|/u_n*100%$', fontsize=16)
    plt.tick_params(direction='in')
    plt.colorbar(orientation="horizontal", pad=0.22)
    # plt.clim(0.0, 1.0)
    plt.tight_layout()

    #################################### isochrones ###################################

    fig_7 = plt.figure(7, figsize=(9, 4))
    fig_7.add_subplot(1, 2, 1)
    N_isochrones=[0, 1, 5, 10, 50]
    for i in range(len(N_isochrones)):
        plt.plot(ur_exact[:, N_isochrones[i]]/u0, r/re, linestyle='none', color='r', label='Exact',
                 marker='o', markerfacecolor='none', markeredgewidth=0.5, markersize=8, markevery=5)
        plt.plot(ur_pred[:, N_isochrones[i]]/u0, r/re, color='k', label='Predicted')
        if i == 0:
            plt.legend(loc=(0.15, 0.05), frameon=False, fontsize=16)

    plt.xlabel('$u_{rn}$')
    plt.ylabel('$r_n$')
    plt.xlim(0.0, 1.01)
    plt.ylim(rw/re, 1.0)
    plt.gca().invert_yaxis()
    plt.tick_params(direction='in')
    plt.tight_layout()

    fig_7.add_subplot(1, 2, 2)
    for i in range(len(N_isochrones)):
        plt.plot(uz_exact[:, N_isochrones[i]] / u0, z / H, linestyle='none', color='r', label='Exact',
                marker='o', markerfacecolor='none', markeredgewidth=0.5, markersize=8, markevery=5)
        plt.plot(uz_pred[:, N_isochrones[i]] / u0, z / H, color='k', label='Predicted')
        if i == 0:
            plt.legend(loc='best', frameon=False, fontsize=16)
    plt.xlabel('$u_{zn}$')
    plt.ylabel('$z_n$')
    plt.xlim(0.0, 1.01)
    plt.ylim(0.0, 1.0)
    plt.gca().invert_yaxis()
    plt.tick_params(direction='in')
    plt.tight_layout()

    #################################### loss values ###################################

    loss_ib = model.loss_k_log
    loss_r = model.loss_f_log
    loss = np.array(loss_ib) + np.array(loss_r)
    fig_4 = plt.figure(4, figsize=(9, 4))
    fig_4.add_subplot(1, 2, 1)
    plt.plot(loss, label='$\mathcal{L}$', color='k')
    plt.plot(loss_ib, label='$\mathcal{L}_\mathrm{k}$', color='b', linestyle='dashdot')
    plt.plot(loss_r, label='$\mathcal{L}_\mathrm{f}$', color='r', linestyle='--')
    plt.xlim(0, 12500)
    plt.ylim(10e-6, 10e3)
    plt.yscale('log')
    plt.xlabel('Number of epochs')
    plt.ylabel('Loss value')
    plt.tick_params(direction='in', which='both')
    plt.legend(frameon=False)
    plt.tight_layout()

    ################################ save results ###############################

    figs = [fig_1, fig_2, fig_4, fig_7]
    nf = 1
    for fig in figs:
        fig.savefig('Fig.' + str(nf) + '.jpg', dpi=1000)
        nf += 1

    mean_error = np.mean([mean_error_r, mean_error_z])
    print(mean_error)
    np.savetxt('2D_RC_ur_exact.csv', ur_exact, fmt='%0.10f', delimiter=',')
    np.savetxt('2D_RC_ur_pred.csv', ur_pred, fmt='%0.10f', delimiter=',')
    np.savetxt('2D_RC_uz_exact.csv', uz_exact, fmt='%0.10f', delimiter=',')
    np.savetxt('2D_RC_uz_pred.csv', uz_pred, fmt='%0.10f', delimiter=',')
    plt.show()