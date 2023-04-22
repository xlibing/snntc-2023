# modules/classes for SNN with temporal coding
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

# Please change to the right direction where all the *.so file are in
C_MODULE_ROOT = '../snntc/'

# change to True if checking tensor dimensions of large models
# we require every tensor under 2**31, or 2GB, in memory size
Check_Dim = False


# weight initialization method
# input parameters: in_size & out_size: size of weight matrix
#                   bias_volt: SNN bias voltage (bias_volt>1e-8 means there is bias, otherwise no bias)
#                   init_param: variance of random weight initialization
# It seems randn() * np.sqrt(init_param / in_size) is the best one
def init_weight(in_size=1, out_size=1, bias_volt=1.0, init_param=15.0):
    # weight = np.random.randn(in_size, out_size) * np.sqrt(init_param)
    # weight = np.random.uniform(low=0. / in_size, high=0.01 / in_size, size=[in_size, out_size])
    weight = np.random.randn(in_size, out_size) * np.sqrt(init_param / in_size)
    if bias_volt > 1e-8:
        weight = np.concatenate((weight, np.zeros([1, out_size])), axis=0)
    return weight.astype(np.float32)

# SNN-TC dense (fully connected) layer module.
# load SNN-TC dense layer forward calculation ops, then load its gradient calculation ops
snnfc = tf.load_op_library(C_MODULE_ROOT + 'snnfc_ops.so').snn_fc
snnfc_grad = tf.load_op_library(C_MODULE_ROOT + 'snnfcgrad_ops.so').snn_fc_grad
@ops.RegisterGradient("SnnFc")
def _snn_fc_grad_cc(op, grad):
    return snnfc_grad(grad, op.inputs[0], op.inputs[1], op.inputs[2], op.outputs[0])


class SNN_dense(tf.keras.layers.Layer):
    # parameters:
    #   in_size, out_size: input/output dimension (in_size not include bias)
    #   SNN params: biasvolt, MAX_SPIKE_TIME, Spike_Threshold, Epsilon (avoid dividing 0)
    #   GPU params: Thread_per_Block, Block (not fully implemented yet)
    #   Python params: weight (initializing weight tensor), name (layer name)
    def __init__(self, in_size=1, out_size=1, biasvolt=1.0, MAX_SPIKE_TIME=1e5, Epsilon=1e-10,
                 Spike_Threshold=1.0, Thread_per_Block=0, Blocks=0, weight=None, name=None):
        super(SNN_dense, self).__init__(name=name)
        self.in_size = in_size  # The weight matrix dimension is (in_size+bias, out_size)
        self.out_size = out_size
        self.biasvolt = float(biasvolt)  # input voltage corresponding to bias, 0 means no bias.
        self.Params = np.array([MAX_SPIKE_TIME, Epsilon, Spike_Threshold, Thread_per_Block, Blocks, self.biasvolt],
                               dtype=np.float32)
        self.maxD = [tf.ones(1, dtype=tf.int64), tf.ones(1, dtype=tf.int64)]  # save maximum dim and Tensor size

        if weight is None:
            weight = init_weight(in_size=self.in_size, out_size=self.out_size, bias_volt=self.biasvolt)
        else:
            weight = np.array(weight).astype(np.float32)

        if name is None:
            name = 'fckernel'  # needed in tensorflow model zoo environment
        if self.biasvolt > 1e-8:
            self.in_size += 1   # correct weight dim includes 1 bias
        self.weight = self.add_weight(shape=[self.in_size, self.out_size], name=name,
                                      initializer=tf.constant_initializer(weight), trainable=True)

    def call(self, X):
        # 1. Pass important parameters in PARAM to SNN C++ ops
        # 2. SNN C++ module's output Y consists of desired output and 1/(\sum_w -1) (used in grad calculation)
        # 3. input X dim is (batchsize, in_size), where in_size does not include bias
        xs = tf.shape(X)
        if Check_Dim:  # if needs to check tensor dimensions
            self.maxD[0] = tf.where(tf.cast(xs[0], tf.int64) > self.maxD[0], tf.cast(xs[0], tf.int64), self.maxD[0])
            check_size = tf.cast(xs[0], tf.int64) * tf.cast(xs[1], tf.int64)
            self.maxD[1] = tf.where(check_size > self.maxD[1], check_size, self.maxD[1])

        Y = snnfc(X, self.weight, self.Params)
        return Y[:xs[0]]   # remove redundant outputs

    def w_sum_cost(self):
        part1 = tf.subtract(self.Params[2], tf.reduce_sum(self.weight, 0))
        part2 = tf.where(part1 > 0, part1, tf.zeros_like(part1))
        return tf.reduce_mean(part2)

    def l2_cost(self):
        w_sqr = tf.square(self.weight)
        return tf.reduce_mean(w_sqr)


# SNN-TC 2D convolutional layer module.
# load SNN-TC conv forward calculation module, then load its gradient calculation module
snncv = tf.load_op_library(C_MODULE_ROOT + 'snncv_ops.so').snn_cv
snncv_grad = tf.load_op_library(C_MODULE_ROOT + 'snncvgrad_ops.so').snn_cv_grad
@ops.RegisterGradient("SnnCv")
def _snn_cv_grad_cc(op, grad):
    return snncv_grad(grad, op.inputs[0], op.inputs[1], op.inputs[2], op.outputs[0])


class SNN_conv(tf.keras.layers.Layer):
    # parameters:
    #   in_channel, out_channel: input/output dimension
    #   conv kernel params: kernel_sizes, strides, padding, rates (not implemented)
    #   SNN params: biasvolt, MAX_SPIKE_TIME, Spike_Threshold, Epsilon (avoid dividing 0)
    #   GPU params: Thread_per_Block, Block (not fully implemented yet)
    #   Python params: weight (initializing weight tensor), name (layer name)
    def __init__(self, in_channel=1, out_channel=1, kernel_sizes=1, strides=1, padding='SAME', rates=1,
                 biasvolt=1.0, MAX_SPIKE_TIME=1e5, Spike_Threshold=1.0, Epsilon=1e-10,
                 Thread_per_Block=0, Blocks=0, weight=None, name=None):
        super(SNN_conv, self).__init__(name=name)
        if not isinstance(kernel_sizes, (list, tuple)):
            kernel_sizes = [kernel_sizes, kernel_sizes]
        self.Kh, self.Kw = kernel_sizes[0], kernel_sizes[1]  # kernel sizes [Kh, Kw]
        self.C = in_channel    # input channel dimension
        self.Kc = out_channel  # output channel dimension
        if not isinstance(strides, (list, tuple)):
            strides = [strides, strides]
        self.Sh, self.Sw = strides[0], strides[1]  # stride sizes [Sh, Sw]
        self.padding = padding.upper()  # 'SAME' or 'VALID'
        self.cudaflag = 0   # reserved, used to choose cuda c++ algorithm, not in use yet
        if not isinstance(rates, (list, tuple)):  # rates have not been implemented yet
            rates = [rates, rates]
        self.rates = rates
        self.biasvolt = float(biasvolt)  # input voltage for bias, 0 means no bias.
        self.Params = np.array([MAX_SPIKE_TIME, Thread_per_Block, Blocks, Epsilon, Spike_Threshold, self.cudaflag,
                                self.Kh, self.Kw, self.Sh, self.Sw, (self.padding == 'SAME'), self.biasvolt],
                               dtype=np.float32)

        self.maxD = [tf.zeros(1, dtype=tf.int64), tf.zeros(1, dtype=tf.int64)]  # record maximum im and Tensor size

        # formulate weight as a matrix of dim [in_size+bias, out_size]=[Kh*Kw*C+bias, Kc]
        self.in_size = self.Kh * self.Kw * self.C
        if weight is None:
            weight = init_weight(in_size=self.in_size, out_size=self.Kc, bias_volt=self.biasvolt)
        else:
            weight = np.array(weight).astype(np.float32)

        if name is None:
            name = 'convkernel'  # needed in tensorflow model zoo environment
        if self.biasvolt > 1e-8:
            self.in_size += 1   # correct weight dim includes 1 bias
        self.weight = self.add_weight(shape=[self.in_size, self.Kc], name=name,
                                      initializer=tf.constant_initializer(weight), trainable=True)

    # we add padding in Python so input X is in regular size H = Ho * Sh + Kh - 1 (easier for cuda grad calculation)
    # In c++ code, tensor X is changed into a 2D patch matrix using Eigen. Output Y is calculated as a 2D matrix.
    # inputs (B, Hi, Wi, C) --> (B, H, W, C) --> (B, Nh, Nw, Kc), where H = (Nh-1)*Sh+Kh, W = (Nw-1)*Sw+Kw
    def call(self, input):
        input_size = tf.shape(input)
        B, H, W, C = input_size[0], input_size[1], input_size[2], input_size[3]
        if self.padding == 'SAME':
            # zero-padding size [top-bottom, left-right]
            ZeroPadSize = [tf.cast((tf.math.ceil(H/self.Sh)-1.0)*self.Sh+self.Kh, tf.int32)-H,
                           tf.cast((tf.math.ceil(W/self.Sw)-1.0)*self.Sw+self.Kw, tf.int32)-W]
            # output dimension [NH, NW]
            # OutDim = [tf.cast((H-self.Kh+ZeroPadSize[0])/self.Sh+1, tf.int32),
            #           tf.cast((W-self.Kw+ZeroPadSize[1])/self.Sw+1, tf.int32)]
            # tensor of zero-padding sizes
            ZeroPad2 = tf.identity([[0, 0],
                                    [tf.cast(ZeroPadSize[0]/2, tf.int32), ZeroPadSize[0]-tf.cast(ZeroPadSize[0]/2, tf.int32)],
                                    [tf.cast(ZeroPadSize[1]/2, tf.int32), ZeroPadSize[1]-tf.cast(ZeroPadSize[1]/2, tf.int32)],
                                    [0, 0]
                                    ])
            # padding with Max-Spike-Time, not 0
            X = tf.pad(input, ZeroPad2, mode='CONSTANT', constant_values=self.Params[0])

        else:  # 'VALID'  # ZeroPadSize = [0, 0]
            # remove extra rows/column to make c++ cuda programming easier
            OutDim = [tf.cast((H-self.Kh)/self.Sh+1.0, tf.int32), tf.cast((W-self.Kw)/self.Sw+1.0, tf.int32)]
            X = input[:, 0:(OutDim[0]-1)*self.Sh+self.Kh, 0:(OutDim[1]-1)*self.Sw+self.Kw, :]

        # call SNN Conv
        Y = snncv(X, self.weight, self.Params)
        return Y[:B]  # discard redundant data (second half of B used for grad calculation only)

    def w_sum_cost(self):
        part1 = tf.subtract(self.Params[4],  tf.reduce_sum(self.weight, 0))
        part2 = tf.where(part1 > 0, part1, tf.zeros_like(part1))
        return tf.reduce_mean(part2)

    def l2_cost(self):
        w_sqr = tf.square(self.weight)
        return tf.reduce_mean(w_sqr)


class SNN_LRN(tf.keras.layers.Layer):
    #  https://gist.github.com/joelouismarino/a2ede9ab3928f999575423b9887abd14?permalink_comment_id=3451596
    # def __init__(self, alpha=0.0001, k=1, beta=0.75, n=5, **kwargs):
    def __init__(self, depth_radius=5, bias=2, alpha=0.0001, beta=0.75, **kwargs):
        # original paper's param. TF default param: (5, 1, 1, 0.5)
        self.depth_radius = depth_radius
        self.bias = bias
        self.alpha = alpha
        self.beta = beta
        super(SNN_LRN, self).__init__(**kwargs)

    def call(self, x):
        b, h, w, c = x.shape   # tf.shape(x)
        half_n = self.depth_radius // 2  # half the local region
        input_sqr = tf.math.square(x)  # square the input
        input_sqr = tf.pad(input_sqr, tf.identity([[0, 0], [0, 0], [0, 0], [half_n, half_n]]),
                           mode='CONSTANT', constant_values=1e5)
        scale = self.bias  # offset for the scale
        for i in range(self.depth_radius):
            scale += self.alpha * input_sqr[:, :, :, i:i+c]
        scale = scale ** self.beta
        x = x / scale
        return x

    def get_config(self):
        config = {"alpha": self.alpha,
                  "depth_radius": self.depth_radius,
                  "beta": self.beta,
                  "bias": self.bias}
        base_config = super(SNN_LRN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# SNN-TC maxpool: major difference is that we do not do 0 padding, but padding with 1e5 (max spike time)
class SNN_maxpool2d(tf.keras.layers.Layer):
    def __init__(self, kernel_sizes=1, strides=1, padding='SAME', padvalue=1e5, name=None):
        super(SNN_maxpool2d, self).__init__(name=name)
        if not isinstance(kernel_sizes, (list, tuple)):
            kernel_sizes = [kernel_sizes, kernel_sizes]
        self.Kh, self.Kw = kernel_sizes[0], kernel_sizes[1]  # kernel sizes Kh*Kw
        if not isinstance(strides, (list, tuple)):
            strides = [strides, strides]
        self.Sh, self.Sw = strides[0], strides[1]  # stride sizes [Sh, Sw]
        self.padding = padding.upper()  # 'SAME' or 'VALID'
        self.padvalue = padvalue
        if name is None:
            name = 'maxpoolkernel'  # needed in tensorflow model zoo environment

    def call(self, input):
        if self.padding == 'SAME':
            input_size = tf.shape(input)
            B, H, W, C = input_size[0], input_size[1], input_size[2], input_size[3]
            # padding size [top-bottom, left-right]
            PadSize = [tf.cast((tf.math.ceil(H/self.Sh)-1.0)*self.Sh+self.Kh, tf.int32)-H,
                       tf.cast((tf.math.ceil(W/self.Sw)-1.0)*self.Sw+self.Kw, tf.int32)-W]
            # tensor of padding sizes
            PadSize2 = tf.identity([[0, 0],
                                    [tf.cast(PadSize[0]/2, tf.int32), PadSize[0]-tf.cast(PadSize[0]/2, tf.int32)],
                                    [tf.cast(PadSize[1]/2, tf.int32), PadSize[1]-tf.cast(PadSize[1]/2, tf.int32)],
                                    [0, 0]
                                    ])
            # padding with Max-Spike-Time
            input = tf.pad(input, PadSize2, mode='CONSTANT', constant_values=self.padvalue)
        x = -tf.keras.layers.MaxPool2D((self.Kh, self.Kw), (self.Sh, self.Sw), padding='VALID')(-input)
        return x


# average pooling of SNN-TC may use directly CNN's average pooling with 0 padding
# this is another choice of padding with 1e5
class SNN_avgpool2d(tf.keras.layers.Layer):
    def __init__(self, kernel_sizes=1, strides=1, padding='SAME', padvalue=1e5, name=None):
        super(SNN_avgpool2d, self).__init__(name=name)
        if not isinstance(kernel_sizes, (list, tuple)):
            kernel_sizes = [kernel_sizes, kernel_sizes]
        self.Kh, self.Kw = kernel_sizes[0], kernel_sizes[1]  # kernel sizes Kh*Kw
        if not isinstance(strides, (list, tuple)):
            strides = [strides, strides]
        self.Sh, self.Sw = strides[0], strides[1]  # stride sizes [Sh, Sw]
        self.padding = padding.upper()  # 'SAME' or 'VALID'
        self.padvalue = padvalue
        if name is None:
            name = 'avgpoolkernel'  # needed in tensorflow model zoo environment

    def call(self, input):
        if self.padding == 'SAME':
            input_size = tf.shape(input)
            B, H, W, C = input_size[0], input_size[1], input_size[2], input_size[3]
            # padding size [top-bottom, left-right]
            PadSize = [tf.cast((tf.math.ceil(H/self.Sh)-1.0)*self.Sh+self.Kh, tf.int32)-H,
                       tf.cast((tf.math.ceil(W/self.Sw)-1.0)*self.Sw+self.Kw, tf.int32)-W]
            # tensor of padding sizes
            PadSize2 = tf.identity([[0, 0],
                                    [tf.cast(PadSize[0]/2, tf.int32), PadSize[0]-tf.cast(PadSize[0]/2, tf.int32)],
                                    [tf.cast(PadSize[1]/2, tf.int32), PadSize[1]-tf.cast(PadSize[1]/2, tf.int32)],
                                    [0, 0]
                                    ])
            # padding with Max-Spike-Time
            input = tf.pad(input, PadSize2, mode='CONSTANT', constant_values=self.padvalue)
        x = tf.keras.layers.AveragePooling2D((self.Kh, self.Kw), (self.Sh, self.Sw), padding='VALID')(input)
        return x


# Just a cross-entropy loss, we negate logits in order to find the minimum one, not the maximum one
# directly use standard tf loss function
def loss_func(logit=None, label=None):
    loss = tf.keras.losses.categorical_crossentropy(label, -logit, from_logits=True, label_smoothing=0.0)
    return tf.reduce_mean(loss)


