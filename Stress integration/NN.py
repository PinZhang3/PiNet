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

class PiNet:
    def __init__(self, E, h, layer, N_grad, store_path):

        self.layer = layer

        # mode
        self.E = E
        self.h = h
        self.num_grad = N_grad
        self.store_path = store_path

        # Init for Solution
        self.nn_init()

        # tf session
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        init = tf.global_variables_initializer()
        self.sess.run(init)

        self.saver.save(self.sess, "NNi_wb/nn_wb.ckpt")

    def initialize_NN(self, layers):
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

    def neural_net(self, X, weights, biases):
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

    def nn_init(self):
        self.weights, self.biases = self.initialize_NN(self.layer)
        self.saver = tf.train.Saver(var_list=[self.weights[l] for l in range(len(self.layer) - 1)]
                                             + [self.biases[l] for l in range(len(self.layer) - 1)])

        # tf placeholders for solution
        self.sig_hist_tf = tf.placeholder(tf.float32)
        self.eps_hist_tf = tf.placeholder(tf.float32)
        self.sig_y_tf = tf.placeholder(tf.float32)
        self.deps_tf = tf.placeholder(tf.float32)

        # physics-informed
        self.sig_pred, self.sig_y_pred, self.deps_p_pred, self.f_pred = \
            self.net(self.sig_hist_tf, self.eps_hist_tf, self.deps_tf)


        # loss
        self.loss = tf.reduce_mean(tf.square(self.f_pred))

        # record loss
        self.loss_log = []

        # optimization
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, var_list=self.weights + self.biases,
                                                                method='L-BFGS-B',
                                                                options={'maxiter': 10000,
                                                                         'maxfun': 10000,
                                                                         'maxcor': 100,
                                                                         'maxls': 100,
                                                                         'gtol': 1e-04})

        self.optimizer_Adam = tf.train.AdamOptimizer()      # default learning rate = 0.001
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss,
                                                          var_list=self.weights + self.biases)

    def net(self, sig_hist_tf, eps_hist_tf, deps_tf):
        X = tf.concat([sig_hist_tf, eps_hist_tf, deps_tf], 1)
        dsig = self.neural_net(X, self.weights, self.biases)
        deps_p = deps_tf - dsig/ self.E
        sig = sig_hist_tf + dsig
        sig_y = self.sig_y_tf + self.h * tf.abs(deps_p)
        f = tf.abs(sig) - sig_y

        return sig, sig_y, deps_p, f

    def callback(self, loss_value):
        print('loss_value: %e' % (loss_value))
        self.loss_log.append(loss_value)

    def train(self, N_epoch, sig_hist, eps_hist, deps, sig_y):

        tf_dict = {self.sig_hist_tf: sig_hist, self.eps_hist_tf: eps_hist,
                   self.deps_tf: deps, self.sig_y_tf: sig_y}

        self.loss_history = []

        for it in range(N_epoch):
            self.sess.run(self.train_op_Adam, tf_dict)
            # if it % (self.num_grad/2) == 0:
            loss_value = self.sess.run(self.loss, tf_dict)
            self.loss_log.append(loss_value)
            # print('It: %d, loss_value: %e' % (it, loss_value))

        self.optimizer.minimize(self.sess,
                                feed_dict=tf_dict,
                                fetches=[self.loss],
                                loss_callback=self.callback)

    def predict(self, sig_hist, eps_hist, deps, sig_y):

        sig_pred = self.sess.run(self.sig_pred, {self.sig_hist_tf: sig_hist, self.eps_hist_tf: eps_hist,
                                                 self.deps_tf: deps, self.sig_y_tf: sig_y})
        sig_y_pred = self.sess.run(self.sig_y_pred, {self.sig_hist_tf: sig_hist, self.eps_hist_tf: eps_hist,
                                                 self.deps_tf: deps, self.sig_y_tf: sig_y})
        deps_p_pred = self.sess.run(self.deps_p_pred, {self.sig_hist_tf: sig_hist, self.eps_hist_tf: eps_hist,
                                                 self.deps_tf: deps, self.sig_y_tf: sig_y})

        return sig_pred, sig_y_pred, deps_p_pred