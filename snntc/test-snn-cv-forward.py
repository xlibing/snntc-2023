#!/usr/bin/env python3
"""
Test C++ implementation of SNN-TC convolutional layer: forward pass accuracy
"""
import numpy as np
import tensorflow as tf
import time
from scipy.io import loadmat
import os

if np.random.randn(1) > 0: Bias = True
else: Bias = False

Test_GPU = True    # switching between testing CPU or GPU
if not Test_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for i in physical_devices: tf.config.experimental.set_memory_growth(i, True)

New_Rand_Data = True   # check with new data or previously saved data (for debugging)
if New_Rand_Data:  # use new random parameters
    B, H, W, C, Kc = np.random.randint(1, 100, 5)
    B = np.random.randint(1, 10)*2
    Kh, Kw = np.random.randint(2, 7, 2)   #
    Sh, Sw = np.random.randint(1, 4, 2)
    if np.random.rand(1)>0.5: padding = 'SAME'
    else: padding = 'VALID'
#    W, Kw = H, Kh
#    B, H, W, C, Kc = 3, 224, 224, 3, 64
    # B, H, W, C, Kc = 128, 32, 32, 3, 64

    # B, H, W, C,   Kh, Kw, Kc,   Sh, Sw = 2, 8, 8, 3,   3, 3, 5,    2, 2
    # padding = 'SAME'
    # B, H, W, C = 32, 224, 224, 3
    # Kh, Hw, Sh, Sw, Kc = 3, 3, 2, 2, 64
# Kh, Kw, Sh, Sw = 5, 5, 3, 3
#    print('Input dim: B, H, W, C =', B, H, W, C, '. Kernel dim: Kh, Kw, C, Kc =', Kh, Kw, C, Kc, '. Stride=', Ks)

    # Note: dim of X does not include the 1 extra bias, but weight does include
    # dim X (B, H, W, C),  dim of weight (Kw*Kw*C+1, Kc) if bias=True
    x_rand = np.exp(np.abs(np.random.randn(B, H, W, C).astype(np.float32)))
    if Bias:
        w_rand = np.random.randn(Kh*Kw*C+1, Kc).astype(np.float32)
    else:
        w_rand = np.random.randn(Kh*Kw*C, Kc).astype(np.float32)

#    B, H, W, C,   Kh, Kw, Kc,   Sh, Sw = 2, 4, 4, 2,   3, 3, 4,    2, 2    # small dataset to show all chain0 & chain
    # x_rand = np.reshape(np.arange(B*H*W*C)+1., (B, H, W, C)).astype(np.float32)  # specialized for B=1
    # w_rand0 = np.reshape((np.arange(Kh*Kw*C+1)+1.)/10., (-1, 1)).astype(np.float32)
    # w_rand = np.concatenate((w_rand0, -w_rand0, w_rand0+2, w_rand0+20), axis=1)

    # x_rand = np.arange((int)(B/2)*H*W*C)+1.
    # x_rand = np.concatenate((x_rand, -(x_rand)))
    # x_rand = np.reshape(x_rand.astype(np.float32), (B, H, W, C))
    # w_rand0 = np.reshape((np.arange(Kh*Kw*C+1)+1.)/10., (-1, 1)).astype(np.float32)
    # w_rand = np.concatenate((w_rand0, -w_rand0, w_rand0+2, w_rand0+20), axis=1)
#    w_rand = np.random.randn(Kh*Kw*C+1, Kc).astype(np.float32)
else:  # or use stored data for debuging
    D = loadmat('tf2test-snncv1.mat')
#    B, H, W, C = D['x1'].shape
    x_rand, w_rand = D['x1'], D['w1']
    B, H, W, C, Kc, Kh, Kw, Sh, Sw = D['params']

X = tf.Variable(x_rand, dtype=tf.float32)

print('Input dim: B, H, W, C =', B, H, W, C, '. Kernel dim: Kh, Kw, C, Kc =', Kh, Kw, C, Kc, '. Strides=', Sh, Sw)
print('Bias = ', Bias, '. Padding = ', padding)
Op1 = [(np.ceil(H/Sh)-1)*Sh-H+Kh, (np.ceil(W/Sw)-1)*Sw-W+Kw]
Ot1 = [(H-Kh+Op1[0])/Sh+1, (W-Kw+Op1[1])/Sw+1]
Ot2 = [(H-Kh)/Sh+1, (W-Kw)/Sw+1]
Op2 = [0, 0]
print('SAME:  2P =', int(Op1[0]), int(Op1[1]), '. out dim =', B, int(Ot1[0]), int(Ot1[1]), Kc)
print('VALID: 2P =', int(Op2[0]), int(Op2[1]), '. out dim =', B, int(Ot2[0]), int(Ot2[1]), Kc)

################### c++ SNN module ##################
print('start c++ code: .....')
t1 = time.time()
from SNN_TC_Modules import SNN_conv
snncv = SNN_conv(in_channel=C, out_channel=Kc, kernel_sizes=(Kh, Kw), strides=(Sh, Sw),
                 weight=w_rand, padding=padding, biasvolt=float(Bias))
ycc0 = snncv(X)
ycc = np.array(ycc0)

################### tf SNN model ##########################
t2 = time.time()

print('start tf code: ....')
print('max tensor is (', B, int(Ot1[0]*Ot1[1]), Kh*Kw*C, ') = ', (B*Ot1[0]*Ot1[1]*Kh*Kc*C/1024/1024), 'M')

from SNN_TC_pyref2 import SNN_CV_Layer as SCNNLayer
layer_in = SCNNLayer(kernel_sizes=(Kh, Kw), in_channel=C, out_channel=Kc, strides=(Sh, Sw),
                     weight=w_rand, padding=padding, bias=Bias)
ytf = layer_in(X)
ytf = np.array(ytf)

print('Output ytf shape = ', np.array(tf.shape(ytf)))
t3 = time.time()

################## compare results ######################

# check difference between c++ patched X (ycc) and tf extracted patches ytf1
# b = np.random.randint(0, B)
# print('B=', b, ' Patch (0,0) err:', np.max(np.abs(ycc[b, :Kh, :Kw].reshape(-1)-ytf1[b, 0, 0])))
# print('B=', b, ' patch (1,2) err:', np.max(np.abs(ycc[b, Ks:Ks+Kh, 2*Ks:2*Ks+Kw].reshape(-1)-ytf1[b, 1, 2])))
# print('B=', b, ' patch (2,1) err:', np.max(np.abs(ycc[b, 2*Ks:2*Ks+Kh, Ks:Ks+Kw].reshape(-1)-ytf1[b, 2, 1])))
# print('B=', b, ' last patch err :', np.max(np.abs(ycc[b, -Kh:, -Kw:].reshape(-1)-ytf1[b, -1, -1])))
#
err = np.abs(ycc-ytf)
print('max err = ', np.max(err))
print('max err (normalized) = ', np.max(err/(np.abs(ytf)+np.abs(ycc))))
if np.max(err) > 100:  # this may be due to one has MAX_SPIKE_TIME, one does not have
    print('  Large err is usually one has MAX_SPIKE_TIME, one does not')
    print('  Caused by tiny difference between input time x and output time y')
    print('  number of large error items = ', np.sum(err>100))
    print('  max err without such large terms =', np.max(err[(err<100)&(ytf<1000)&(ycc<1000)]))

print('Time difference: C++:', t2-t1, ', TF:', t3-t2)
# compare the output element with the maximum err
# following commands are for hand-calculating SNN outputs for a position (b, nh, nw, kc)
#def snnout(b, nh, nw, kc):
ma = np.where(err == np.max(err))
b, nh, nw, kc = ma[0][0], ma[1][0], ma[2][0], ma[3][0]
#b, nh, nw, kc = 0, 2, 3, 0
if padding == 'SAME':
    ZeroPad2 = tf.identity([[0, 0], [int(Op1[0]/2), int(Op1[0])-int(Op1[0]/2)], [int(Op1[1]/2), int(Op1[1])-int(Op1[1]/2)], [0, 0]])
    if tf.reduce_min(ZeroPad2) < 0:
        print(' negative zero-pad: this set of parameter not appropriate, stop & exit')
    X1 = tf.pad(X, ZeroPad2, mode='CONSTANT', constant_values=1e5)
else: X1 = X[:, 0:(int(Ot2[0])-1)*Sh+Kh, 0:(int(Ot2[1])-1)*Sw+Kw, :]
x1 = np.array(X1)
x2 = x1[b, nh*Sh:nh*Sh+Kh, nw*Sw:nw*Sw+Kw]
if Bias:
    x3 = np.concatenate((x2.reshape(-1), [1]))
else:
    x3 = x2.reshape(-1)
w3 = w_rand[:, kc]
x3r = np.argsort(x3)
x4, w4 = x3[x3r], w3[x3r]
y4 = np.cumsum(x4*w4)/(np.cumsum(w4)-1)
ind2 = (y4>x4)&(y4<=np.concatenate((x4[1:], [1e5])))
trueout = y4[ind2]
print('For element [', b, nh, nw, kc, '] where C++ and TF differ most: C++ =', ycc[b, nh, nw, kc], ', TF =', ytf[b, nh, nw, kc])
if trueout.size > 0:
    print('True output seems ', trueout[0], '. Calculated with k =', np.where(ind2==True)[0][0], 'weights.')
#print('C++ used ', np.array(ycc0[B+b, nh, nw, kc],int), 'weights. You can check x4[k] and y4[k] for input and output spike time.')

# check C++ and hand-calculated difference in (out, k, wsum)
def mycompare():
    size = ycc.shape
    for gi in range(np.prod(size)):
        b = int(gi/(np.prod(size[1:])))
        nh = int((gi%np.prod(size[1:]))/np.prod(size[2:]))
        nw = int((gi%np.prod(size[2:]))/size[3])
        kc = int(gi%size[3])
        x2 = x1[b, nh*Sh:nh*Sh+Kh, nw*Sw:nw*Sw+Kw]
        x3 = np.concatenate((x2.reshape(-1), [1]))
        w3 = w_rand[:, kc]
        x3r = np.argsort(x3)
        x4, w4 = x3[x3r], w3[x3r]
        y4 = np.cumsum(x4*w4)/(np.cumsum(w4)-1)
        ind2 = (y4>x4)&(y4<=np.concatenate((x4[1:], [1e5])))
        ind21 = np.where(ind2==True)[0]
        # if len(ind21) == 0:
        #     print("True-C++ (y,k,w):", [b, nh, nw, kc])
        #     continue
        if (len(ind21) > 1): ind21 = ind21[0]
        trueout, truek, truew = y4[ind21], ind21, (np.cumsum(w4)-1)[ind21]
        ccout, cck, ccw = ycc0[b, nh, nw, kc].numpy(), ycc0[B+b, nh, nw, kc].numpy().astype(int), ycc0[2*B+b, nh, nw, kc].numpy()
        print("True-C++ (y,k,w):", [b, nh, nw, kc], ': ', [trueout, truek, truew], ', ', [ccout, cck, ccw])

# mycompare()

# compare not so large error item
"""
ma = np.where(err == np.max(err[(err<100)&(ytf<1000)&(ycc<1000)]))

k = np.where(ind2)[0]
print(k)
print('Input time: ', x4[k[0]-1:k[0]+3])
print('Output time:', y4[k[0]-1:k[0]+3])
"""
