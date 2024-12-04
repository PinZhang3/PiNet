# ----------------------------------------------------------------------
# Project Name: PiNet - Prior Information based NEural neTwork
# Description : A physics informed learning based framework for constitutive modelling and BVPs
# Author      : Pin ZHANG, National University of Singapore
# Contact     : pinzhang@nus.edu.sg
# Created On  : 4 Dec 2024
# Repository  : https://github.com/PinZhang3/PiNet
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
from scipy.interpolate import griddata
from pyDOE import lhs
import time


class PiNet:
    def __init__(self, Ini, Top_t, Bot_t, X_f, layers, lb, ub):

        # domain boundary
        self.lb = lb
        self.ub = ub

        # training data
        self.t0 = Ini[:, 0:1]
        self.x0 = Ini[:, 1:2]
        self.u0 = Ini[:, 2:3]

        self.tb = Bot_t[:,0:1]
        self.xb = Bot_t[:,1:2]

        self.tt = Top_t[:,0:1]
        self.xt = Top_t[:,1:2]

        self.tf = X_f[:,0:1]
        self.xf = X_f[:,1:2]
        
        # tf placeholders
        self.t0_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.x0_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.u0_tf = tf.placeholder(tf.float32, shape=[None, 1])

        self.tt_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.xt_tf = tf.placeholder(tf.float32, shape=[None, 1])

        self.tb_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.xb_tf = tf.placeholder(tf.float32, shape=[None, 1])

        self.tf_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.xf_tf = tf.placeholder(tf.float32, shape=[None, 1])

        # initialize weights and biases
        self.layers = layers
        self.weights, self.biases = self.initialize_nn(layers)
        
        # loss function
        self.u0_pred, _, _   = self.net_u(self.t0_tf, self.x0_tf)
        _, self.ub_x_pred, _ = self.net_u(self.tb_tf, self.xb_tf)
        self.ut_pred, _, _   = self.net_u(self.tt_tf, self.xt_tf)
        _, _, self.f_pred    = self.net_u(self.tf_tf, self.xf_tf)

        self.loss = tf.reduce_sum(tf.square(self.u0_tf - self.u0_pred)) + \
                    tf.reduce_sum(tf.square(self.ut_pred))   + \
                    tf.reduce_sum(tf.square(self.ub_x_pred)) + \
                    tf.reduce_sum(tf.square(self.f_pred))
        
        # Optimizer for Solution
        self.optimizer_LBFGS = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                             var_list = self.weights + self.biases,
                             method = 'L-BFGS-B',
                             options = {'maxiter': 10000,
                                        'maxfun': 10000,
                                        'maxcor': 100,
                                        'maxls': 100,
                                        'gtol': 1e-03})
    
        self.optimizer_Adam = tf.train.AdamOptimizer().minimize(self.loss,
                                 var_list = self.weights + self.biases)

        # tf session
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        init = tf.global_variables_initializer()
        self.sess.run(init)

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
    
    def net_u(self, t, x):
        X = tf.concat([t, x], 1)
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        u = self.neural_network(H, self.weights, self.biases)
        u_t = tf.gradients(u, t)[0]
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        f = u_t - 2.24 / 365 / 24 * u_xx
        return u, u_x, f
    
    def callback(self, loss):
        print('Loss: %e' % (loss))
        self.loss_history.append(loss)
        
    def train(self, N_iter):
        tf_dict = {self.t0_tf: self.t0, self.x0_tf: self.x0, self.u0_tf: self.u0,
                   self.tb_tf: self.tb, self.xb_tf: self.xb,
                   self.tt_tf: self.tt, self.xt_tf: self.xt,
                   self.tf_tf: self.tf, self.xf_tf: self.xf}

        start_time = time.time()
        self.loss_history=[]
        for it in range(N_iter):
            
            self.sess.run(self.optimizer_Adam, tf_dict)
            
            # Print
            if it % 100 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                print('It: %d, Loss: %.3e, Time: %.2f' % 
                      (it, loss_value, elapsed))
                start_time = time.time()
                
        self.optimizer_LBFGS.minimize(self.sess,
                                    feed_dict = tf_dict,
                                    fetches = [self.loss],
                                    loss_callback = self.callback)
    
    def predict(self, t, x):
        
        u = self.sess.run(self.u0_pred, {self.t0_tf: t, self.x0_tf: x})
        f = self.sess.run(self.f_pred, {self.tf_tf: t, self.xf_tf: x})
               
        return u, f

if __name__ == "__main__": 

    # Doman bounds
    lb = np.array([0.0, -0.03])
    ub = np.array([4.0, 0])
    
    # Load Data

    data_test = scipy.io.loadmat('consol_1D_sub.mat')

    t_test = data_test['tspan'].flatten()[:, None]
    x_test = data_test['z'].flatten()[:, None]
    u_test = np.real(data_test['u_sub'])
    
    T_test, X_test = np.meshgrid(t_test, x_test)
    
    t_test_col = T_test.flatten()[:, None]
    x_test_col = X_test.flatten()[:, None]
    X_test_col = np.hstack((t_test_col, x_test_col))
    u_test_col = u_test.flatten()[:, None]
     
    ### Training Data ###
    T = t_test.shape[0]
    N = x_test.shape[0]

    # Initial t=0
    Ini = np.zeros((N, 3))
    Ini[:, 0:1] = np.zeros((N, 1))
    Ini[:, 1:2] = x_test
    Ini[:, 2:3] = u_test[:, 0:1]

    # Boundary
    Top_t = np.zeros((T, 2))
    Top_t[:, 0:1] = t_test
    Top_t[:, 1:2] = ub[1] * np.ones((T, 1))

    Bot_t = np.zeros((T, 2))
    Bot_t[:, 0:1] = t_test
    Bot_t[:, 1:2] = lb[1] * np.ones((T, 1))

    # PDE condition
    N_f = 20000
    X_f_train = lb + (ub - lb)*lhs(2, N_f)

    # Layers
    layers = [2, 60, 60, 60, 60, 60, 1]
    
    # Model
    model = PiNet(Ini, Top_t, Bot_t, X_f_train, layers, lb, ub)

    # training and testing
    model.train(N_iter=1000)
    u_pred, f_pred = model.predict(t_test_col, x_test_col)

    U_pred = griddata(X_test_col, u_pred.flatten(), (T_test, X_test), method='cubic')
    U_exact = griddata(X_test_col, u_test_col.flatten(), (T_test, X_test), method='cubic')
    '../Data/burgers_sine.mat'

    ############################# Plotting ###############################

    rc = {"font.family": "serif", "mathtext.fontset": "stix", "font.size": 16}
    plt.rcParams.update(rc)
    plt.rcParams["font.serif"] = ["Times New Roman"]

    fig = plt.figure(1, figsize=(14, 4.5))
    fig.add_subplot(1, 3, 1)
    plt.pcolor(T_test, X_test, U_exact, cmap='RdYlBu_r')
    plt.xlabel('$t$')
    plt.ylabel('$x$')
    plt.title('Exact $u^e(t,z)$', fontsize=16)
    plt.tick_params(direction='in')
    plt.colorbar(orientation="horizontal", pad=0.22)
    # plt.clim(0.4, 1.0)

    fig.add_subplot(1, 3, 2)
    plt.pcolor(T_test, X_test, U_pred, cmap='RdYlBu_r')
    plt.xlabel('$t$')
    plt.ylabel('$x$')
    plt.title('Predicted $u^p(t,z)$', fontsize=16)
    plt.tick_params(direction='in')
    plt.colorbar(orientation="horizontal", pad=0.22)

    fig.add_subplot(1, 3, 3)
    relative_error = np.abs(U_pred - U_exact)
    plt.pcolor(T_test, X_test, relative_error, cmap='RdYlBu_r')
    plt.xlabel('$t$')
    plt.ylabel('$x$')
    plt.title('$|u^e-u^p|$', fontsize=16)
    plt.tick_params(direction='in')
    plt.colorbar(orientation="horizontal", pad=0.22)

    plt.show()
