#!/usr/bin/env python3
"""
Test C++ implementation of the SNN-TC convolutional layer module:
check accuracy of forward & backward & gradient
"""
import time
import numpy as np
import tensorflow as tf
from scipy.io import loadmat
import os

if np.random.randn(1) > 0: Bias = True
else: Bias = False

Test_GPU = True  # switching between testing CPU or GPU
if not Test_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for i in physical_devices: tf.config.experimental.set_memory_growth(i, True)

# Note: dim of X does not include the 1 extra bias, but weight does include
# dim X (B, H, W, C),  dim of weight (Kw*Kw*C+1, Kc)
New_Rand_Data = True   # check with new data or previously saved data (for debugging)
if New_Rand_Data:  # use new random parameters
    B, H, W, C, Kc = np.random.randint(1, 100, 5)
    B = np.random.randint(1, 10)
    Kh, Kw = np.random.randint(2, 7, 2)   #
    Sh, Sw = np.random.randint(1, 4, 2)
    if np.random.rand(1)>0.5: padding = 'SAME'
    else: padding = 'VALID'
#    W, Kw = H, Kh
#     B, H, W, C, Kc = 3, 224, 224, 3, 64   # typical Imagenet
#     B, H, W, C, Kc = 128, 32, 32, 3, 64  # typical CIFAR10
#     B, H, W, C, Kc = 128, 28, 28, 1, 64  # typical MNIST
    x_rand = np.exp(np.abs(np.random.randn(B, H, W, C).astype(np.float32)))
    w_rand = np.random.randn(Kh*Kw*C+1, Kc).astype(np.float32)

## the following are some simple examples, used in debuging programs
    testmode = 5
# # testing case 4
    if testmode == 4:
        B, H, W, C,   Kh, Kw, Kc,   Sh, Sw = 2, 8, 8, 3,   3, 3, 4,    2, 2
        padding = 'SAME'
        # Kh, Kw, Sh, Sw = 5, 5, 3, 3
        x_rand = np.exp(np.abs(np.random.randn(B, H, W, C).astype(np.float32)))
        w_rand = np.random.randn(Kh*Kw*C+1, Kc).astype(np.float32)

# # testing case 3: random
    if testmode == 3:
        B, H, W, C,   Kh, Kw, Kc,   Sh, Sw = 1, 4, 4, 1, 3, 3, 1, 2, 2
        padding = 'SAME'
        x_rand = np.exp(np.abs(np.random.randn(B, H, W, C).astype(np.float32)))
        w_rand = np.random.randn(Kh*Kw*C+1, Kc).astype(np.float32)

# # testing case 2: second simplest
    if testmode == 2:
        B, H, W, C, Kh, Kw, Kc, Sh, Sw = 2, 4, 4, 2, 3, 3, 2, 2, 2  # small dataset to show all chain0 & chai
        padding = 'SAME'
        x_rand = np.arange(int(B * H * W * C)) + 1.1
        x_rand = np.reshape(x_rand.astype(np.float32), (B, H, W, C))
        w_rand = np.reshape((np.arange((Kh * Kw * C + 1)*Kc)+ 1.) / 10., (-1, Kc)).astype(np.float32)

# # testing case 1: second simplest
    if testmode == 1:
        B, H, W, C, Kh, Kw, Kc, Sh, Sw = 1, 4, 4, 2, 3, 3, 2, 2, 2  # small dataset to show all chain0 & chai
        padding = 'SAME'
        x_rand = np.arange(int(B * H * W * C)) + 1.1
        x_rand = np.reshape(x_rand.astype(np.float32), (B, H, W, C))
        w_rand = np.reshape((np.arange((Kh * Kw * C + 1)*Kc)+ 1.) / 10., (-1, Kc)).astype(np.float32)

# # testing case 0: simplest
    if testmode == 0:
        B, H, W, C,   Kh, Kw, Kc,   Sh, Sw = 1, 4, 4, 2,   3, 3, 1,    2, 2    # small dataset to show all chain0 & chai
        padding = 'SAME'
        x_rand = np.arange(int(B*H*W*C))+1.1
        x_rand = np.reshape(x_rand.astype(np.float32), (B, H, W, C))
        w_rand = np.reshape((np.arange(Kh*Kw*C+1)+1.)/10., (-1, 1)).astype(np.float32)

    # x_rand = np.reshape(np.arange(B*H*W*C)+1., (B, H, W, C)).astype(np.float32)  # specialized for B=1
    # w_rand0 = np.reshape((np.arange(Kh*Kw*C+1)+1.)/10., (-1, 1)).astype(np.float32)
    # w_rand = np.concatenate((w_rand0, -w_rand0, w_rand0+2, w_rand0+20), axis=1)
    #
    # x_rand = np.arange((int)(B/2)*H*W*C)+1.
    # x_rand = np.concatenate((x_rand, -(x_rand)))
    # x_rand = np.reshape(x_rand.astype(np.float32), (B, H, W, C))
    # w_rand0 = np.reshape((np.arange(Kh*Kw*C+1)+1.)/10., (-1, 1)).astype(np.float32)
    # w_rand = np.concatenate((w_rand0, -w_rand0, w_rand0+2, w_rand0+20), axis=1)
    # x_rand = x_rand+0.1
# # #    w_rand = np.random.randn(Kh*Kw*C+1, Kc).astype(np.float32)
else:  # or use stored data for debuging
    D = loadmat('tf2test-snncv1.mat')
#    B, H, W, C = D['x1'].shape
    x_rand, w_rand = D['x1'], D['w1']
    B, H, W, C, Kc, Kh, Kw, Sh, Sw = D['params']

if not Bias: w_rand = w_rand[:Kh*Kw*C]

X = tf.Variable(x_rand, dtype=tf.float32)

print('Input dim: B, H, W, C =', B, H, W, C, '. Kernel dim: Kh, Kw, C, Kc =', Kh, Kw, C, Kc, '. Strides=', Sh, Sw)
print('Bias = ', Bias, '. Padding = ', padding)
Op1 = [(np.ceil(H/Sh)-1)*Sh-H+Kh, (np.ceil(W/Sw)-1)*Sw-W+Kw]
Ot1 = [(H-Kh+Op1[0])/Sh+1, (W-Kw+Op1[1])/Sw+1]
Ot2 = [(H-Kh)/Sh+1, (W-Kw)/Sw+1]
Op2 = [0, 0]
print('SAME:  2P =', int(Op1[0]), int(Op1[1]), '. out dim =', B, int(Ot1[0]), int(Ot1[1]), Kc)
print('VALID: 2P =', int(Op2[0]), int(Op2[1]), '. out dim =', B, int(Ot2[0]), int(Ot2[1]), Kc)

print('------------------------ start c++ version -------------------------------------------')
print('max tensor size is ', max(np.prod(np.array(X.shape)), B*np.prod(Ot1)*Kc, Kh*Kw*X.shape[-1]*Kc)/1024/1024, 'M')
print('complexity (billion iterations): Forward:', int(B*(2*Ot1[0]*Ot1[1]*Kc+H*W*C*Kc*Kh*Kw/Sh/Sw)/1e9),
      ', Grad(X):', int(B*H*W*C*Kh*Kw/Sh/Sw*(2.*Kc+Kh*Kw*C)/1e9), ', Grad(W):', int(Kh*Kw*C*Kc*B*Ot1[0]*Ot1[1]*Kh*Kw*C/1e9))
t1 = time.time()

from SNN_TC_Modules import SNN_conv
snncv = SNN_conv(in_channel=C, out_channel=Kc, kernel_sizes=(Kh, Kw), strides=(Sh, Sw),
                 weight=w_rand, padding=padding, biasvolt=float(Bias))
with tf.GradientTape(persistent=True) as cctape:
    ycc = snncv(X)
gxc, gwc = cctape.gradient(ycc, [X, snncv.weight])   # last column gradient !=0, but is useless, should be discarded
ycca, gxca, gwca = np.array(ycc), np.array(gxc), np.array(gwc)
t2 = time.time()

print('------------------------- start tf version --------------------------------------------')
print('max tensor is (', B, int(Ot1[0]*Ot1[1]), Kh*Kw*C, ') = ', (B*Ot1[0]*Ot1[1]*Kh*Kc*C/1024/1024), 'M. Need much more than ',
      (B*Ot1[0]*Ot1[1]*Kh*Kc*C*Kc/1024/1024/1024), 'GB memory.')

from SNN_TC_pyref2 import SNN_CV_Layer as SCNNLayer
layer_in = SCNNLayer(kernel_sizes=(Kh, Kw), in_channel=C, out_channel=Kc, strides=(Sh, Sw),
                     weight=w_rand, padding=padding, bias=Bias)
with tf.GradientTape(persistent=True) as tftape:
    ytf = layer_in(X)
gxt, gwt = tftape.gradient(ytf, [X, layer_in.kernel.weight])   # last column gradient = 0 because last column is constant 1.
ytfa, gxta, gwta = np.array(ytf), np.array(gxt), np.array(gwt)
t3 = time.time()

#print('Output ytf shape = ', np.array(tf.shape(ytf)))

################## check errors #########################
print('-------------------------- Performance Comparison -----------------------------------')
print('Time difference: C++:', t2-t1, ', TF:', t3-t2)
print('out min & max (except MAX_SPIKE) =', np.min(ycca), ', ', np.max(ycca[ycca<1e5]))
print('out max err |y1-y2| =', np.max(np.abs(ytfa-ycca)))
print('out max |err| normalized =', np.max(np.abs(ytfa-ycca)/(np.abs(ytfa)+np.abs(ycca))))
if np.size(np.where(ycca<=1)[0])>0: print('Something wrong: some outputs <=1')

print('grad value max: |Gx|= ', np.max(np.abs(gxta)), ', |Gw|= ', np.max(np.abs(gwta)))
print('grad max err: |Gx1-Gx2| = ', np.max(np.abs(gxca-gxta)),
      '. |Gw1-Gw2| = ', np.max(np.abs(gwca-gwta)))
eg1 = np.abs(gxca-gxta)/(np.abs(gxca)+np.abs(gxta)+1e-10)
eg2 = np.abs(gwca-gwta)/(np.abs(gwca)+np.abs(gwta)+1e-10)
print('grad max |err| normalized: Gx = ', np.max(eg1), '. Gw = ', np.max(eg2))


