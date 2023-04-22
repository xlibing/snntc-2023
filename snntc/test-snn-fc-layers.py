#!/usr/bin/env python3
"""
Test C++ SNN-TC dense (fully-connected module) with a multiple layer model.
Check accuracy of forward & backward pass & gradient.
"""
import time
import numpy as np
import tensorflow as tf
from scipy.io import loadmat
import os

Bias = True

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
    # M, N, K = 4, 5, 3
    #M, N, K = 8*10, 10*10, 6*10
    x_rand = np.exp(np.abs(np.random.randn(M, N-1).astype(np.float32)))
    w_rand = np.random.randn(N, K).astype(np.float32)
    w_rand2 = np.random.randn(K+1, K-5).astype(np.float32)
    y_rand = np.random.randn(M, K-5).astype(np.float32)
else:  # or use stored data for debuging
    D = loadmat('tf2test-snn-layers1.mat')
    M, N = D['x1'].shape
    _, K = D['w1'].shape
    x_rand, w_rand, w_rand2, y_rand = D['x1'], D['w1'], D['w2'], D['y']

X = tf.Variable(x_rand, dtype=tf.float32)

print('M, N, K=', M, N, K, '. Out dim=', M, 'x', K, '. Bias = ', Bias)

################### c++ SNN module ##################
print('start c++ version ..................................')
from SNN_TC_Modules import SNN_dense
My_SNN_FC_Layer = SNN_dense(in_size=N-1, out_size=K, weight=w_rand, biasvolt=Bias)
My_SNN_FC_Layer2 = SNN_dense(in_size=K, out_size=K-5, weight=w_rand2, biasvolt=Bias)

t2 = time.time()
with tf.GradientTape(persistent=True) as cctape:
    ycc1 = My_SNN_FC_Layer(X)
    ycc = My_SNN_FC_Layer2(ycc1)
    losscc = tf.reduce_mean(tf.square(ycc - y_rand))
g2x, g2w = cctape.gradient(losscc, [X, My_SNN_FC_Layer2.weight])
ycca, g2xa, g2wa = np.array(ycc), np.array(g2x), np.array(g2w)
t2 = time.time() - t2

################### SNN tf reference ##########################
print('start tf version ...................................')
from SNN_TC_pyref import SNNLayer
layer_in = SNNLayer(N-1, K, w_rand, bias=Bias)
layer_in2 = SNNLayer(K, K-5, w_rand2, bias=Bias)
t1 = time.time()
with tf.GradientTape(persistent=True) as tftape:
    ytf1 = layer_in.forward(X)
    ytf = layer_in2.forward(ytf1)
    losstf = tf.reduce_mean(tf.square(ytf - y_rand))
g1x, g1w = tftape.gradient(losstf, [X, layer_in2.weight])
ytfa, g1xa, g1wa = np.array(ytf), np.array(g1x), np.array(g1w)
t1 = time.time() - t1

################## check errors #########################
print('M, N, K=', M, N, K, '. Out dim=', M, 'x', K)
print('time spent: tf = ', t1, '. cc = ', t2)

print('Compare foward:')
print('out min & max (except MAX_SPIKE) =', np.min(ycca), ', ', np.max(ycca[ycca<1e5]))
print('out max err |y1-y2| =', np.max(np.abs(ytfa-ycca)))
print('out max |err| normalized =', np.max(np.abs(ytfa-ycca)/(np.abs(ytfa)+np.abs(ycca))))
print('Loss err(%): ', np.array(tf.abs(losstf-losscc)/(losstf+losscc)), '. tf= ', np.array(losstf), ', cc= ', np.array(losscc))

if np.size(np.where(ycca<=1)[0])>0:
    print('Something wrong: some outputs <=1')

print('Compare gradients:')
print('grad value max: |Gx|= ', np.max(np.abs(g2xa)), ', |Gw|= ', np.max(np.abs(g2wa)))
print('grad max err: |Gx1-Gx2| = ', np.max(np.abs(g1xa-g2xa)),
      '. |Gw1-Gw2| = ', np.max(np.abs(g1wa-g2wa)))
eg1 = np.abs(g1xa-g2xa)/(np.abs(g1xa)+np.abs(g2xa)+1e-10)
eg2 = np.abs(g1wa-g2wa)/(np.abs(g1wa)+np.abs(g2wa)+1e-10)
print('grad max |err| normalized: Gx = ', np.max(eg1), '. Gw = ', np.max(eg2))

#savemat('tf2test-snn-layers1.mat', {'x1':x_rand, 'w1':w_rand, 'w2':w_rand2, 'y':y_rand, 'yt':np.array(ytfa), 'yc':ycca,
#                                   'gxt':g1xa, 'gxc':g2xa, 'gwt':g1wa, 'gwc':g2wa})

