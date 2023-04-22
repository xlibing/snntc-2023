#!/usr/bin/env python3
"""
Test C++ SNN-TC dense (fully-connected) layer module
Check accuracy of both forward & backward (gradient) pass
"""
import time
import numpy as np
import tensorflow as tf
from scipy.io import loadmat
import os

if np.random.randn(1) > 0: Bias = False
else: Bias = True

Test_GPU = True   # switching between testing CPU or GPU
if not Test_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for i in physical_devices:
        tf.config.experimental.set_memory_growth(i, True)

New_Rand_Data = True
if New_Rand_Data:  # use new random parameters
    M, N, K = np.random.randint(1,1000,3)
    M = np.random.randint(1, 100)
    # M, N, K = 7, 5, 3
    #M, N, K = 8*10, 10*10, 6*10
    x_rand = np.exp(np.abs(np.random.randn(M, N).astype(np.float32)))
    w_rand = np.random.randn(N, K).astype(np.float32)
else:  # or use stored data for debuging
    D = loadmat('tf2test-snnfcgrad1.mat')
    M, N = D['x1'].shape
    _, K = D['w1'].shape
    x_rand, w_rand = D['x1'], D['w1']

X = tf.Variable(x_rand, dtype=tf.float32)
W = tf.Variable(w_rand, dtype=tf.float32)

print('M, N, K=', M, N, K, '. Out dim=', M, 'x', K, '. Bias = ', Bias)

################### c++ SNN-TC module ##################
print('start c++ version ..................................')
from SNN_TC_Modules import SNN_dense
t2 = time.time()
if Bias:
    snnfc = SNN_dense(in_size=N-1, out_size=K, weight=w_rand, biasvolt=float(Bias))
    with tf.GradientTape(persistent=True) as cctape:
        ycc = snnfc(X[:, :N-1])
else:
    snnfc = SNN_dense(in_size=N, out_size=K, weight=w_rand, biasvolt=float(Bias))
    with tf.GradientTape(persistent=True) as cctape:
        ycc = snnfc(X)

g2x, g2w = cctape.gradient(ycc, [X, snnfc.weight])
ycca, g2xa, g2wa = np.array(ycc), np.array(g2x), np.array(g2w)
t2 = time.time() - t2

################### tf SNN model ##########################
print('start tf reference version ...................................')

from SNN_TC_pyref import SNNLayer
t1 = time.time()
if Bias:
    layer_in = SNNLayer(N-1, K, w=w_rand, bias=Bias)
    with tf.GradientTape(persistent=True) as tftape:
        ytf = layer_in.forward(X[:, :N-1])
else:
    layer_in = SNNLayer(N, K, w=w_rand, bias=Bias)
    with tf.GradientTape(persistent=True) as tftape:
        ytf = layer_in.forward(X)

g1x, g1w = tftape.gradient(ytf, [X, layer_in.weight])   # last column gradient = 0 because last column is constant 1.
ytfa, g1xa, g1wa = np.array(ytf), np.array(g1x), np.array(g1w)
t1 = time.time() - t1

################## check errors #########################
print('M, N, K=', M, N, K, '. Out dim=', M, 'x', K)
print('out min & max (except MAX_SPIKE) =', np.min(ycca), ', ', np.max(ycca[ycca<1e5]))
print('out max err |y1-y2| =', np.max(np.abs(ytfa-ycca)))
print('out max |err| normalized =', np.max(np.abs(ytfa-ycca)/(np.abs(ytfa)+np.abs(ycca))))
print('time spent: tf = ', t1, '. cc = ', t2)
if np.size(np.where(ycca<=1)[0])>0:
    print('Something wrong: some outputs <=1')

print('grad value max: |Gx|= ', np.max(np.abs(g2xa)), ', |Gw|= ', np.max(np.abs(g2wa)))
print('grad max err: |Gx1-Gx2| = ', np.max(np.abs(g1xa[:,:N-1]-g2xa[:,:N-1])),
      '. |Gw1-Gw2| = ', np.max(np.abs(g1wa-g2wa)))
eg1 = np.abs(g1xa[:,:N-1]-g2xa[:,:N-1])/(np.abs(g1xa[:,:N-1])+np.abs(g2xa[:,:N-1])+1e-10)
eg2 = np.abs(g1wa-g2wa)/(np.abs(g1wa)+np.abs(g2wa)+1e-10)
print('grad max |err| normalized: Gx = ', np.max(eg1), '. Gw = ', np.max(eg2))

#savemat('tf2test-snnfcgrad1.mat', {'x1':x_rand, 'w1':w_rand, 'yt':np.array(ytfa), 'yc':ycca,
#                                   'gxt':g1xa, 'gxc':g2xa, 'gwt':g1wa, 'gwc':g2wa})

