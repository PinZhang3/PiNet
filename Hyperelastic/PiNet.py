"""
@author: Pin ZHANG, 2023, Cambridge
@Reference: Interpretable data-driven constitutive modelling of soils with sparse data.
            Computers and Geotechnics, 160, 105511
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from sklearn.metrics import r2_score

###############################################################################
####################### PiNet for Hyperelastic Modelling ######################
###############################################################################

def initialize_nn(layers):
    weights = []
    biases = []
    num_layers = len(layers) 
    for l in range(0,num_layers-1):
        W = weights_init(size=[layers[l], layers[l+1]])
        b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
        weights.append(W)
        biases.append(b)
    return weights, biases
    
def weights_init(size):
    in_dim = size[0]
    out_dim = size[1]
    weights_stddev = np.sqrt(2/(in_dim + out_dim))
    return tf.Variable(tf.random.truncated_normal([in_dim, out_dim], stddev=weights_stddev, dtype=tf.float32), dtype=tf.float32)

def neural_net(X, weights, biases):
    num_layers = len(weights) + 1
    H = X
    for l in range(0,num_layers-2):
        W = weights[l]
        b = biases[l]
        H = tf.tanh(tf.add(tf.matmul(H, W), b))
    W = weights[-1]
    b = biases[-1]
    Y = tf.add(tf.matmul(H, W), b)
    return Y

###############################################################################
################################ PiNet Class ##################################
###############################################################################

class PiNet:
    def __init__(self, Data, layers, N_out, lb, ub):
        # Domain Boundary
        self.lb_output = lb[:N_out]
        self.ub_output = ub[:N_out]
        self.lb_input = lb[N_out:]
        self.ub_input = ub[N_out:]

        self.nn_init(Data, layers)

        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        
        init = tf.global_variables_initializer()
        self.sess.run(init)
    
    def nn_init(self, Data, layers):

        self.F_out = Data[:, 0:1]
        self.p_out = Data[:, 1:2]
        self.q_out = Data[:, 2:3]

        self.F = Data[:, 3:4]
        self.epsv = Data[:, 4:5]
        self.epsd = Data[:, 5:6]
        self.depsv = Data[:, 6:7]
        self.depsd = Data[:, 7:8]
        
        # Layers for Solution
        self.layers = layers
        
        # Initialize NNs for Solution
        self.weights, self.biases = initialize_nn(layers)
        self.saver = tf.train.Saver(var_list=[self.weights[l] for l in range(len(self.layers) - 1)]
                                             + [self.biases[l] for l in range(len(self.layers) - 1)])
        
        # tf placeholders for Solution
        self.F_out_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.p_out_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.q_out_tf = tf.placeholder(tf.float32, shape=[None, 1])

        self.F_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.epsv_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.epsd_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.depsv_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.depsd_tf = tf.placeholder(tf.float32, shape=[None, 1])
        
        # tf graphs for Solution
        self.F_out_pred, self.p_out_pred, self.q_out_pred = self.nn_output(
            self.F_tf, self.epsv_tf,self.epsd_tf,self.depsv_tf,self.depsd_tf)

        # loss for Solution
        F_out_tf_nr, p_out_tf_nr, q_out_tf_nr = \
            self.min_max_norm(self.F_out_tf, self.p_out_tf, self.q_out_tf)
        F_out_pred_nr, p_out_pred_nr, q_out_pred_nr = \
            self.min_max_norm(self.F_out_pred, self.p_out_pred, self.q_out_pred)
        nn_loss_1 = tf.reduce_sum(tf.square(F_out_tf_nr - F_out_pred_nr))
        nn_loss_2 = tf.reduce_sum(tf.square(p_out_tf_nr - p_out_pred_nr))
        nn_loss_3 = tf.reduce_sum(tf.square(q_out_tf_nr - q_out_pred_nr))
        nn_loss = nn_loss_1 + nn_loss_2 + nn_loss_3

        self.nn_loss = nn_loss_1 ** 2 / nn_loss + nn_loss_2 ** 2 / nn_loss + nn_loss_3 ** 2 / nn_loss
        
        # Optimizer for Solution
        self.nn_optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.nn_loss,
                             var_list = self.weights + self.biases,
                             method = 'L-BFGS-B',
                             options = {'maxiter': 50000,
                                        'maxfun': 50000,
                                        'maxcor': 100,
                                        'maxls': 100,
                                        'gtol': 1e-03})
    
        self.nn_optimizer_Adam = tf.train.AdamOptimizer()
        self.nn_train_op_Adam = self.nn_optimizer_Adam.minimize(self.nn_loss,
                                 var_list = self.weights + self.biases)
    
    def nn_output(self, F,epsv,epsd,depsv,depsd):
        X = tf.concat([F,epsv,epsd,depsv,depsd],1)
        H = 2.0*(X - self.lb_input)/(self.ub_input - self.lb_input) - 1.0
        F_out = neural_net(H, self.weights, self.biases)
        p_out = tf.gradients(F_out, epsv)[0]
        q_out = tf.gradients(F_out, epsd)[0]
        print(F_out)
        print(p_out)
        return F_out, p_out, q_out

    def min_max_norm(self, F, p, q):
        Y = tf.concat([F, p, q], 1)
        Y_norm = 2.0 * (Y - self.lb_output) / (self.ub_output - self.lb_output) - 1.0
        return Y_norm[:, 0], Y_norm[:, 1], Y_norm[:, 2]
    
    def callback(self, loss):
        print('Loss: %.3e' %(loss))
        self.loss_history.append(loss)
        
    def nn_train(self, N_iter, N_interv):
        tf_dict = {self.F_out_tf: self.F_out,
                   self.p_out_tf: self.p_out,
                   self.q_out_tf: self.q_out,
                   self.F_tf: self.F,
                   self.epsv_tf: self.epsv,
                   self.epsd_tf: self.epsd,
                   self.depsv_tf: self.depsv,
                   self.depsd_tf: self.depsd}

        start_time = time.time()
        self.loss_history=[]
        for it in range(N_iter):
            
            self.sess.run(self.nn_train_op_Adam, tf_dict)
            
            # Print
            if it % N_interv == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.nn_loss, tf_dict)
                self.loss_history.append(loss_value)
                print('It: %d, Loss: %.3e, Time: %.2f' % 
                      (it, loss_value, elapsed))
                start_time = time.time()
                
        self.nn_optimizer.minimize(self.sess,
                                    feed_dict = tf_dict,
                                    fetches = [self.nn_loss],
                                    loss_callback = self.callback)

        F_out_pred = self.sess.run(self.F_out_pred, tf_dict)
        p_out_pred = self.sess.run(self.p_out_pred, tf_dict)
        q_out_pred = self.sess.run(self.q_out_pred, tf_dict)
        store_path = os.getcwd()
        self.saver.save(self.sess, store_path + "\paras_NN.ckpt")
        return self.loss_history, F_out_pred, p_out_pred, q_out_pred
    
    def nn_predict(self, Te, N_each, N_te):
        F_te = np.zeros((Te.shape[0], 1))
        p_te = np.zeros((Te.shape[0], 1))
        q_te = np.zeros((Te.shape[0], 1))
        for i in range(N_te):
            testX = Te[N_each * i: N_each * (i + 1), :]
            for j in range(0, N_each):
                tf_dict = {self.F_tf: testX[j:j+1, 3:4],
                           self.epsv_tf: testX[j:j+1, 4:5],
                           self.epsd_tf: testX[j:j+1, 5:6],
                           self.depsv_tf: testX[j:j+1, 6:7],
                           self.depsd_tf: testX[j:j+1, 7:8]}
                F_te[j+N_each * i, :] = self.sess.run(self.F_out_pred, tf_dict)
                p_te[j+N_each * i, :] = self.sess.run(self.p_out_pred, tf_dict)
                q_te[j+N_each * i, :] = self.sess.run(self.q_out_pred, tf_dict)
                if j != (N_each-1):
                    testX[j + 1, 3:4] = F_te[j+N_each * i, :]
        return F_te, p_te, q_te

    def error_indicator(self, actu, pred, N_out):
        names = locals()
        model_order = 1
        Indicator = np.zeros((N_out + 1, model_order * 2))
        for mi in range(1, model_order + 1):
            names['R' + str(mi)] = 0
            names['MAE' + str(mi)] = 0
            for oi in range(N_out):
                names['R' + str(mi) + '_' + str(oi + 1)] = r2_score(actu[:, oi], pred[:, oi])
                names['MAE' + str(mi) + '_' + str(oi + 1)] = np.mean(np.abs(actu[:, oi] - pred[:, oi]))
                names['R' + str(mi)] = names['R' + str(mi)] + names['R' + str(mi) + '_' + str(oi + 1)]
                names['MAE' + str(mi)] = names['MAE' + str(mi)] + names['MAE' + str(mi) + '_' + str(oi + 1)]
                Indicator[oi, mi - 1] = names['R' + str(mi) + '_' + str(oi + 1)]
                Indicator[oi, mi + model_order - 1] = names['MAE' + str(mi) + '_' + str(oi + 1)]
            Indicator[N_out, mi - 1] = names['R' + str(mi)] / N_out
            Indicator[N_out, mi + model_order - 1] = names['MAE' + str(mi)] / N_out
        return Indicator

    def AP_scatter(self, actu, pred, N_out):
        plt.rcParams["figure.figsize"] = (15, 5)
        fig, ax = plt.subplots(1, N_out)
        for i in range(N_out):
            ax[i].scatter(actu[:,i], pred[:,i], marker='o')
            ax[i].set_xlim(np.min(actu[:,i]), np.max(actu[:,i]))
            ax[i].set_ylim(np.min(actu[:,i]), np.max(actu[:,i]))
            ax[i].set_xlabel('Actual $\sigma$')
            ax[i].set_ylabel('Predicted $\sigma$')
        plt.show()

    def Loss_curve(self, history):
        plt.rcParams["figure.figsize"] = (5, 5)
        fig, ax = plt.subplots(1, 1)
        ax.plot(history)
        ax.set_xlabel('Number of epochs')
        ax.set_ylabel('Loss')
        ax.set_yscale('log')
        plt.show()

###############################################################################
################################ Main Function ################################
###############################################################################

if __name__ == "__main__":

    # Data
    Tr = pd.read_csv('data_training.csv', header=None).values
    Tv = pd.read_csv('data_validation.csv', header=None).values
    Te = pd.read_csv('data_testing.csv', header=None).values

    D_input = 5
    N_out = 3  # total number of outputs
    N_nc = 1   # not constraint terms
    N_loading = 100
    N_each = 128
    N_te = int(0.2*N_loading)

    Data = np.vstack((Tr, Tv))

    lb = np.min(Data, axis=0)
    ub = np.max(Data, axis=0)

    # Model
    layers = [D_input, 64, 64,  N_nc]
    model = PiNet(Data, layers, N_out, lb, ub)
    
    # Training and Testing
    history, F_pred_tr, p_predtr, q_predtr = model.nn_train(N_iter=0, N_interv=1)
    F_pred, p_pred, q_pred = model.nn_predict(Te, N_each, N_te)

    # Save
    np.savetxt('loss.csv', history, fmt='%0.10f', delimiter=',')
    model.Loss_curve(history)

    tr_actu = Data[:, :3]
    tr_pred = np.hstack((F_pred_tr, p_predtr, q_predtr))
    tr_error = model.error_indicator(tr_actu, tr_pred, 3)
    model.AP_scatter(tr_actu, tr_pred, 3)
    np.savetxt('out_training.csv', np.hstack((tr_actu, tr_pred)), fmt='%.10f', delimiter=',')
    np.savetxt('out_training_error.csv', tr_error, fmt='%.10f', delimiter=',')

    te_actu = Te[:, :3]
    te_pred = np.hstack((F_pred, p_pred, q_pred))
    te_error = model.error_indicator(te_actu, te_pred, 3)
    model.AP_scatter(te_actu, te_pred, 3)

    np.savetxt('out_testing.csv', np.hstack((te_actu, te_pred)), fmt='%.10f', delimiter=',')
    np.savetxt('out_testing_error.csv', te_error, fmt='%.10f', delimiter=',')