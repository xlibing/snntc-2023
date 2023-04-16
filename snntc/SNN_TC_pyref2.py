# Reference Python tensorflow implementation of SNN modules with temporal coding,
# tested with tensorflow 2.9, but main work in other tensorflow v2.* as well.
# Mainly used to compare and debug c++ implementation.
# It is not very useful because it needs too big GPU memory. A CIFAR10 model may exhaust all GPU memory.

import numpy as np
import tensorflow as tf


class SNN_FC_Layer(tf.keras.layers.Layer):
    def __init__(self, in_size, out_size, weight=None, bias=False):
        super(SNN_FC_Layer, self).__init__()
        self.MAX_SPIKE_TIME = 1e5
        self.out_size = out_size
        self.bias = bias
        self.in_size = in_size
        if self.bias: self.in_size = in_size + 1
        if weight is None:
            if self.bias:
                weight = np.concatenate((np.random.uniform(low=0. / self.in_size, high=8. / self.in_size,
                                                           size=[self.in_size - 1, self.out_size]),
                                         np.zeros([1, self.out_size])), axis=0).astype(np.float32)
            else:
                weight = np.random.uniform(low=0. / self.in_size, high=8. / self.in_size,
                                                           size=[self.in_size, self.out_size]).astype(np.float32)
        else:
            weight = np.array(weight).astype(np.float32)

        self.weight = self.add_weight(shape=[self.in_size, self.out_size],
                                      initializer=tf.constant_initializer(weight), trainable=True)

    def call(self, layer_in):
        batch_num = tf.shape(layer_in)[0]
        if self.bias:
            bias_layer_in = tf.ones([batch_num, 1])
            layer_in = tf.concat([layer_in, bias_layer_in], 1)
        _, input_sorted_indices = tf.nn.top_k(-layer_in, self.in_size, False)
        input_sorted = tf.gather(layer_in, input_sorted_indices, batch_dims=1)
        input_sorted_outsize = tf.tile(tf.reshape(input_sorted, [batch_num, self.in_size, 1]), [1, 1, self.out_size])
        weight_sorted = tf.gather(
            tf.tile(tf.reshape(self.weight, [1, self.in_size, self.out_size]), [batch_num, 1, 1]),
            input_sorted_indices, batch_dims=1)

        weight_input_mul = tf.multiply(weight_sorted, input_sorted_outsize)
        weight_sumed = tf.cumsum(weight_sorted, axis=1)
        weight_input_sumed = tf.cumsum(weight_input_mul, axis=1)
        out_spike_all = tf.divide(weight_input_sumed, tf.clip_by_value(weight_sumed - 1, 1e-10, 1e10))
        out_spike_ws = tf.where(weight_sumed < 1, self.MAX_SPIKE_TIME * tf.ones_like(out_spike_all), out_spike_all)
        out_spike_large = tf.where(out_spike_ws < input_sorted_outsize,
                                   self.MAX_SPIKE_TIME * tf.ones_like(out_spike_ws), out_spike_ws)
        input_sorted_outsize_slice = tf.slice(input_sorted_outsize, [0, 1, 0],
                                              [batch_num, self.in_size - 1, self.out_size])
        input_sorted_outsize_left = tf.concat(
            [input_sorted_outsize_slice, self.MAX_SPIKE_TIME * tf.ones([batch_num, 1, self.out_size])], 1)
        out_spike_valid = tf.where(out_spike_large > input_sorted_outsize_left,
                                   self.MAX_SPIKE_TIME * tf.ones_like(out_spike_large), out_spike_large)
        out_spike = tf.reduce_min(out_spike_valid, axis=1)
        return out_spike

    def w_sum_cost(self):
        threshold = 1.
        part1 = tf.subtract(threshold, tf.reduce_sum(self.weight, 0))
        part2 = tf.where(part1 > 0, part1, tf.zeros_like(part1))
        return tf.reduce_mean(part2)

    def l2_cost(self):
        w_sqr = tf.square(self.weight)
        return tf.reduce_mean(w_sqr)


class SNN_CV_Layer(tf.keras.layers.Layer):
    def __init__(self, kernel_sizes=3, in_channel=1, out_channel=1, strides=1, weight=None, padding=None, bias=False):
        super(SNN_CV_Layer, self).__init__()
        self.MAX_SPIKE_TIME = 1e5
        if not isinstance(kernel_sizes, (list, tuple)):
            kernel_sizes = [kernel_sizes, kernel_sizes]
        self.kernel_size = kernel_sizes
        self.in_channel = in_channel
        self.out_channel = out_channel
        if not isinstance(strides, (list, tuple)):
            strides = [strides, strides]
        self.strides = strides
        self.padding = padding
        self.kernel = SNN_FC_Layer(in_size=self.kernel_size[0] * self.kernel_size[1] * self.in_channel,
                                   out_size=self.out_channel, weight=weight, bias=bias)

    def call(self, layer_in):
        input_size = tf.shape(layer_in)
        patches = tf.image.extract_patches(images=layer_in, sizes=[1, self.kernel_size[0], self.kernel_size[1], 1],
                                           strides=[1, self.strides[0], self.strides[1], 1], rates=[1, 1, 1, 1],
                                           padding=self.padding)
        output_size = tf.shape(patches)
        patches_flatten = tf.reshape(patches,
                                     [input_size[0], -1, self.in_channel * self.kernel_size[0] * self.kernel_size[1]])
        patches_flatten = tf.where(tf.less(patches_flatten, 0.9),
                                   self.MAX_SPIKE_TIME * tf.ones_like(patches_flatten), patches_flatten)
        img_raw = tf.map_fn(self.kernel.call, patches_flatten)
        img_reshaped = tf.reshape(img_raw, [output_size[0], output_size[1], output_size[2], self.out_channel])
        # img_reshaped = tf.reshape(img_raw,
        #                           [input_size[0], tf.cast(tf.math.ceil(input_size[1] / self.strides[0]), tf.int32),
        #                            tf.cast(tf.math.ceil(input_size[2] / self.strides[1]), tf.int32),
        #                            self.out_channel])
        return img_reshaped


def loss_func(logit=None, label=None):
    z1 = tf.exp(tf.subtract(0., tf.reduce_sum(tf.multiply(logit, label), axis=1)))
    z2 = tf.reduce_sum(tf.exp(tf.subtract(0., logit)), axis=1)
    loss = tf.subtract(
        0., tf.math.log(
            tf.clip_by_value(tf.divide(
                z1, tf.clip_by_value(
                    z2, 1e-10, 1e10)), 1e-10, 1)))
    return tf.reduce_mean(loss)


def old_loss_func(both):
    """
    function to calculate loss, refer to paper p.7, formula 11
    :param both: a tensor, it put both layer output and expected output together, its' shape
            is [batch_size,out_size*2], where the left part is layer output(real output), right part is
            the label of input(expected output), the tensor both should be looked like this:
            [[2.13,3.56,7.33,3.97,...0,0,1,0,...]
             [3.14,5.56,2.54,15.6,...0,0,0,1,...]...]
                ↑                   ↑
             layer output           label of input
    :return: a tensor, it is a scalar of loss
    """
    output = tf.slice(both, [0], [tf.cast(tf.shape(both)[0] / 2, tf.int32)])
    index = tf.slice(both, [tf.cast(tf.shape(both)[0] / 2, tf.int32)],
                     [tf.cast(tf.shape(both)[0] / 2, tf.int32)])
    z1 = tf.exp(tf.subtract(0., tf.reduce_sum(tf.multiply(output, index))))
    z2 = tf.reduce_sum(tf.exp(tf.subtract(0., output)))
    loss = tf.subtract(
        0., tf.log(
            tf.clip_by_value(tf.divide(
                z1, tf.clip_by_value(
                    z2, 1e-10, 1e10)), 1e-10, 1)))
    return loss

