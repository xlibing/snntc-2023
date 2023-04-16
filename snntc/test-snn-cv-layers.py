#!/usr/bin/env python3
"""
Test C++ implementation of the SNN-TC convolutional layer module:
check accuracy of forward & backward & gradient over a multi-layer model
"""

import time
import numpy as np
import tensorflow as tf
import os

if np.random.randn(1) > 0: Bias = True
else: Bias = False

Test_GPU = True   # switching between testing CPU or GPU
if not Test_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for i in physical_devices: tf.config.experimental.set_memory_growth(i, True)

# Note: dim of X does not include the 1 extra bias, but weight does include
# dim X (B, H, W, C),  dim of weight (Kw*Kw*C+1, Kc)
H, W, C, Kc, Kc2 = np.random.randint(1, 32, 5)
B = np.random.randint(1, 6)
Kh, Kw, Kh2, Kw2 = np.random.randint(2, 5, 4)   #
Sh, Sw, Sh2, Sw2 = np.random.randint(1, 4, 4)
if np.random.rand(1)>0.5: padding = 'SAME'
else: padding = 'VALID'
#     B, H, W, C, Kc = 3, 224, 224, 3, 64   # typical Imagenet
#     B, H, W, C, Kc = 128, 32, 32, 3, 64  # typical CIFAR10
#     B, H, W, C, Kc = 128, 28, 28, 1, 64  # typical MNIST
x_rand = np.exp(np.abs(np.random.randn(B, H, W, C).astype(np.float32)))
if Bias:
    w_rand = np.random.randn(Kh*Kw*C+1, Kc).astype(np.float32)
    w_rand2 = np.random.randn(Kh2*Kw2*Kc+1, Kc2).astype(np.float32)
else:
    w_rand = np.random.randn(Kh*Kw*C, Kc).astype(np.float32)
    w_rand2 = np.random.randn(Kh2*Kw2*Kc, Kc2).astype(np.float32)

if padding == 'SAME':
    Op1 = [int((np.ceil(H/Sh)-1)*Sh-H+Kh), int((np.ceil(W/Sw)-1)*Sw-W+Kw)]
    Ot1 = [int((H-Kh+Op1[0])/Sh+1), int((W-Kw+Op1[1])/Sw+1)]
    Op2 = [int((np.ceil(Ot1[0]/Sh2)-1)*Sh2-Ot1[0]+Kh2), int((np.ceil(Ot1[1]/Sw2)-1)*Sw2-Ot1[1]+Kw2)]
    Ot2 = [int((Ot1[0]-Kh2+Op2[0])/Sh2+1), int((Ot1[1]-Kw2+Op2[1])/Sw2+1)]
else:
    Ot1 = [int((H-Kh)/Sh+1), int((W-Kw)/Sw+1)]
    Ot2 = [int((Ot1[0]-Kh2)/Sh2+1), int((Ot1[1]-Kw2)/Sw2+1)]
    Op1 = [0, 0]
    Op2 = [0, 0]
y_rand = np.random.randn(B, Ot2[0], Ot2[1], Kc2)

X = tf.Variable(x_rand, dtype=tf.float32)

print('Bias = ', Bias, ', Padding = ', padding)
print('Parameters Layer1(In,Ker,Str,Pad,Out):', [B,H,W,C], '->', [Kh,Kw],[Sh,Sw],Op1, '->', [B,Ot1[0],Ot1[1],Kc])
print('Parameters Layer2(In,Ker,Str,Pad,Out):', [B,Ot1[0],Ot1[1],Kc], '->', [Kh2,Kw2],[Sh2,Sw2],Op2, '->', [B,Ot2[0],Ot2[1],Kc2])

################### c++ SNN Conv module ##################
print('------------------------ start c++ version -------------------------------------------')
print('max tensor size is ', max(np.prod(np.array(X.shape)), B*np.prod(Ot1)*Kc,
                                 Kh*Kw*X.shape[-1]*Kc, B*np.prod(Ot2)*Kc2, Kh2*Kw2*Kc*Kc2)/1024/1024, 'M')
print('complexity (billion iteration): ', max(Kh*Kw*C*Kc*B*Ot1[0]*Ot1[1]*Kh*Kw*C/1e9,
                                              Kh2*Kw2*Kc*Kc2*B*Ot2[0]*Ot2[1]*Kh2*Kw2*Kc/1e9))
t1 = time.time()

from SNN_TC_Modules import SNN_conv
snncvlayer1 = SNN_conv(in_channel=C, out_channel=Kc, kernel_sizes=(Kh, Kw), strides=(Sh, Sw),
                       weight=w_rand, padding=padding, biasvolt=Bias)
snncvlayer2 = SNN_conv(in_channel=Kc, out_channel=Kc2, kernel_sizes=(Kh2, Kw2), strides=(Sh2, Sw2),
                       weight=w_rand2, padding=padding, biasvolt=Bias)
with tf.GradientTape(persistent=True) as cctape:
    ycc0 = snncvlayer1(X)
    ycc = snncvlayer2(ycc0)
    losscc = tf.reduce_mean(tf.square(ycc - y_rand))
gxc, gwc = cctape.gradient(losscc, [X, snncvlayer2.weight])
ycca, gxca, gwca = np.array(ycc), np.array(gxc), np.array(gwc)
t2 = time.time()

################### tf SCNN model ##########################
print('------------------------ start tf version -------------------------------------------')
print('max tensor size is ', max(B*Ot1[0]*Ot1[1]*Kh*Kc*C/1024/1024, B*Ot2[0]*Ot2[1]*Kh2*Kc2*Kc/1024/1024), 'M')
print('Need much more than ', max(B*Ot1[0]*Ot1[1]*Kh*Kc*C*Kc/1024/1024/1024,
                                  B*Ot2[0]*Ot2[1]*Kh2*Kc2*Kc*Kc2/1024/1024/1024), 'GB memory.')

from SNN_TC_pyref2 import SNN_CV_Layer as SCNNLayer
layer_in1 = SCNNLayer(kernel_sizes=(Kh, Kw), in_channel=C, out_channel=Kc, strides=(Sh, Sw),
                     weight=w_rand, padding=padding, bias=Bias)
layer_in2 = SCNNLayer(kernel_sizes=(Kh2, Kw2), in_channel=Kc, out_channel=Kc2, strides=(Sh2, Sw2),
                     weight=w_rand2, padding=padding, bias=Bias)
with tf.GradientTape(persistent=True) as tftape:
    ytf0 = layer_in1(X)
    ytf = layer_in2(ytf0)
    losstf = tf.reduce_mean(tf.square(ytf - y_rand))
gxt, gwt = tftape.gradient(losstf, [X, layer_in2.kernel.weight])   # last column gradient = 0 because last column is constant 1.
ytfa, gxta, gwta = np.array(ytf), np.array(gxt), np.array(gwt)
t3 = time.time()

################## check errors #########################
print('------------------------ Performance Comparison -------------------------------------------')
print('time spent: tf = ', t2-t1, '. cc = ', t3-t2)

print('Compare foward:')
print('out min & max (except MAX_SPIKE) =', np.min(ycca), ', ', np.max(ycca[ycca<1e5]))
print('out max err |y1-y2| =', np.max(np.abs(ytfa-ycca)))
print('out max |err| normalized =', np.max(np.abs(ytfa-ycca)/(np.abs(ytfa)+np.abs(ycca))))
print('Loss err(%): ', np.array(tf.abs(losstf-losscc)/(losstf+losscc)), '. tf= ', np.array(losstf), ', cc= ', np.array(losscc))

if np.size(np.where(ycca<=1)[0])>0:
    print('Something wrong: some outputs <=1')

print('Compare gradients:')
print('grad value max: |Gx|= ', np.max(np.abs(gxta)), ', |Gw|= ', np.max(np.abs(gwta)))
print('grad max err: |Gx1-Gx2| = ', np.max(np.abs(gxca-gxta)),
      '. |Gw1-Gw2| = ', np.max(np.abs(gwca-gwta)))
eg1 = np.abs(gxca-gxta)/(np.abs(gxca)+np.abs(gxta)+1e-10)
eg2 = np.abs(gwca-gwta)/(np.abs(gwca)+np.abs(gwta)+1e-10)
print('grad max |err| normalized: Gx = ', np.max(eg1), '. Gw = ', np.max(eg2))

#savemat('tf2test-snn-layers1.mat', {'x1':x_rand, 'w1':w_rand, 'w2':w_rand2, 'y':y_rand, 'yt':np.array(ytfa), 'yc':ycca,
#                                   'gxt':g1xa, 'gxc':g2xa, 'gwt':g1wa, 'gwc':g2wa})

