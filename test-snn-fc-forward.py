#!/usr/bin/env python3
"""
Test C++ SNN-TC dense layer forward module: snnfc_ops.so
"""
import numpy as np
import tensorflow as tf
import time
import os
from scipy.io import loadmat

if np.random.randn(1) > 0: Bias = False
else: Bias = True

Test_GPU = True  # switching between testing CPU or GPU version of the ops (CPU version has very limited functionality)

if not Test_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for i in physical_devices: tf.config.experimental.set_memory_growth(i, True)

New_Rand_Data = True   # check with new data or previously saved data
if New_Rand_Data:  # use new random parameters
    M, N, K = np.random.randint(1, 1000, 3)
    # M, N, K = 500, 500, 500
    # M = np.random.randint(1, 100)
    # M, N, K = 4, 5, 3
    # M, N, K = 8, 4, 6
    x_rand = np.exp(np.abs(np.random.randn(M, N).astype(np.float32)))
    # x_rand[:, -1] = 1.0   ## last column must be 1 since this is default bias
    w_rand = np.random.randn(N, K).astype(np.float32)
else:  # or use stored data for debuging
    D = loadmat('tf2test-snnfc1.mat')
    M, N = D['x1'].shape
    _, K = D['w1'].shape
    x_rand, w_rand = D['x1'], D['w1']

X = tf.Variable(x_rand, dtype=tf.float32)   # variables are differentiable,
W = tf.Variable(w_rand, dtype=tf.float32)   # convert_to_tensor is not differentiable

print('M, N, K, bias = ', M, N, K, Bias)
print('start c++ code: .....')
start_time = time.time()
################### c++ SNN-TC module ##################
from SNN_TC_Modules import SNN_dense
if Bias:
    snnfc = SNN_dense(N-1, K, weight=w_rand, biasvolt=float(Bias))
    y = snnfc(X[:, :N-1])
else:
    snnfc = SNN_dense(N, K, weight=w_rand, biasvolt=float(Bias))
    y = snnfc(X)
ycc = np.array(y)
time1 = time.time() - start_time;

print('start tf reference code: ....')
start_time = time.time()
################### Python SNN-TC reference moddule ##########################
from SNN_TC_pyref import SNNLayer
if Bias:
    layer_in = SNNLayer(N-1, K, w=w_rand, bias=Bias)
    ytf = layer_in.forward(X[:, :N-1])
else:
    layer_in = SNNLayer(N, K, w=w_rand, bias=Bias)
    ytf = layer_in.forward(X)
ytf = np.array(ytf)
time2 = time.time() - start_time

################## compare results ######################
err = np.abs(ycc-ytf)
print('M, N, K=', M, N, K)
print('max err = ', np.max(err))
print('max err (normalized) = ', np.max(err/(np.abs(ytf)+np.abs(ycc))))
if np.max(err) > 100:  # this may be due to one has MAX_SPIKE_TIME, one does not have
    print('  Large err usually is due to one has MAX_SPIKE_TIME, one does not')
    print('  number of large error items = ', np.sum(err>100))
    print('  max err without such large terms =', np.max(err[(err<100)&(ytf<1000)&(ycc<1000)]))

print('c++ time =', str(time1), ', tf time =', str(time2))
# savemat('tf2test-snnfc1.mat', {'x1':x_rand, 'w1':w_rand, 'yt':np.array(ytf), 'yc':ycc})

# following commands are for hand-calculating SNN outputs
"""
x, w = x_rand[245, :], w_rand[:, 330]
ind = np.argsort(x)
x2, w2 = x[ind], w[ind]
y2 = np.cumsum(x2*w2)/(np.cumsum(w2)-1)
ind2 = (y2>x2)
"""