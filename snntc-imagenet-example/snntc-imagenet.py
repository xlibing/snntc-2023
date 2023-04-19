# A sample code showing how to implement GoogleNet of IMAGENET with SNN-TC.
# This should be taken as a skeleton example illustrating the usage of SNN-TC modules,
# not really useful to train the model over IMAGENET realistically.

import numpy as np
import tensorflow as tf
import config
from data_utils import train_iterator, test_iterator
import time

import SNN_TC_Modules as SNN

physical_devices = tf.config.experimental.list_physical_devices('GPU')
for my_gpu in physical_devices:
    tf.config.experimental.set_memory_growth(device=my_gpu, enable=True)

K = 100
K2 = 1e-2

# noinspection PyCallingNonCallable
class MyModel(tf.keras.Model):
    def __init__(self, weight_k=100, weight_k2=1e-2, scaling=1.0):
        self.K = weight_k
        self.K2 = weight_k2
        self.scaling = scaling

        inputs = tf.keras.Input(shape=config.input_shape, dtype=tf.float32, name='myinput')
        input_real_resize = tf.exp((1.0 - inputs) * self.scaling)

        conv1_7x7_s2 = SNN.SNN_conv(kernel_sizes=7, in_channel=3, out_channel=64, strides=2, padding='same')(input_real_resize)  #112,64
        pool1_3x3_s2 = SNN.SNN_maxpool2d(kernel_sizes=(3, 3), strides=(2, 2), padding='same')(conv1_7x7_s2)   # 56,64

        # LRN produces elements < 1 or bias, not convenient for SNN-TC.
        # If using LRN, we need to set biasvolt carefully to make sure bias always in use. This still needs more studying.
        # pool1_norm1 = SNN.SNN_LRN(name='pool1/norm1')(pool1_3x3_s2)

        conv2_3x3_reduce = SNN.SNN_conv(kernel_sizes=1, in_channel=64, out_channel=64, strides=1)(pool1_3x3_s2)
        conv2_3x3 = SNN.SNN_conv(kernel_sizes=3, in_channel=64, out_channel=192, strides=1)(conv2_3x3_reduce)
        pool2_3x3_s2 = SNN.SNN_maxpool2d(kernel_sizes=(3, 3), strides=(2, 2), padding='same')(conv2_3x3)   # 28,192

        inception_3a_1x1 = SNN.SNN_conv(kernel_sizes=1, in_channel=192, out_channel=64, strides=1)(pool2_3x3_s2)
        inception_3a_3x3_reduce = SNN.SNN_conv(kernel_sizes=1, in_channel=192, out_channel=96, strides=1)(pool2_3x3_s2)
        inception_3a_3x3 = SNN.SNN_conv(kernel_sizes=3, in_channel=96, out_channel=128, strides=1)(inception_3a_3x3_reduce)
        inception_3a_5x5_reduce = SNN.SNN_conv(kernel_sizes=1, in_channel=192, out_channel=16, strides=1)(pool2_3x3_s2)
        inception_3a_5x5 = SNN.SNN_conv(kernel_sizes=5, in_channel=16, out_channel=32, strides=1)(inception_3a_5x5_reduce)
        inception_3a_pool = SNN.SNN_maxpool2d(kernel_sizes=(3, 3), strides=(1, 1), padding='same')(pool2_3x3_s2)
        inception_3a_pool_proj = SNN.SNN_conv(kernel_sizes=1, in_channel=192, out_channel=32, strides=1)(inception_3a_pool)
        inception_3a_output = tf.concat([inception_3a_1x1, inception_3a_3x3, inception_3a_5x5, inception_3a_pool_proj], axis=3) #28,256

        inception_3b_1x1 = SNN.SNN_conv(kernel_sizes=1, in_channel=256, out_channel=128, strides=1)(inception_3a_output)
        inception_3b_3x3_reduce = SNN.SNN_conv(kernel_sizes=1, in_channel=256, out_channel=128, strides=1)(inception_3a_output)
        inception_3b_3x3 = SNN.SNN_conv(kernel_sizes=3, in_channel=128, out_channel=192, strides=1)(inception_3b_3x3_reduce)
        inception_3b_5x5_reduce = SNN.SNN_conv(kernel_sizes=1, in_channel=256, out_channel=32, strides=1)(inception_3a_output)
        inception_3b_5x5 = SNN.SNN_conv(kernel_sizes=5, in_channel=32, out_channel=96, strides=1)(inception_3b_5x5_reduce)
        inception_3b_pool = SNN.SNN_maxpool2d(kernel_sizes=(3, 3), strides=(1, 1), padding='same')(inception_3a_output)
        inception_3b_pool_proj = SNN.SNN_conv(kernel_sizes=1, in_channel=256, out_channel=64, strides=1)(inception_3b_pool)
        inception_3b_output = tf.concat([inception_3b_1x1, inception_3b_3x3, inception_3b_5x5, inception_3b_pool_proj], axis=3) #28,480
        pool3_3x3_s2 = SNN.SNN_maxpool2d(kernel_sizes=(3, 3), strides=(2, 2), padding='same')(inception_3b_output)  #14,480

        inception_4a_1x1 = SNN.SNN_conv(kernel_sizes=1, in_channel=480, out_channel=192, strides=1)(pool3_3x3_s2)
        inception_4a_3x3_reduce = SNN.SNN_conv(kernel_sizes=1, in_channel=480, out_channel=96, strides=1)(pool3_3x3_s2)
        inception_4a_3x3 = SNN.SNN_conv(kernel_sizes=3, in_channel=96, out_channel=208, strides=1)(inception_4a_3x3_reduce)
        inception_4a_5x5_reduce = SNN.SNN_conv(kernel_sizes=1, in_channel=480, out_channel=16, strides=1)(pool3_3x3_s2)
        inception_4a_5x5 = SNN.SNN_conv(kernel_sizes=5, in_channel=16, out_channel=48, strides=1)(inception_4a_5x5_reduce)
        inception_4a_pool = SNN.SNN_maxpool2d(kernel_sizes=(3, 3), strides=(1, 1), padding='same')(pool3_3x3_s2)
        inception_4a_pool_proj = SNN.SNN_conv(kernel_sizes=1, in_channel=480, out_channel=64, strides=1)(inception_4a_pool)
        inception_4a_output = tf.concat([inception_4a_1x1, inception_4a_3x3, inception_4a_5x5, inception_4a_pool_proj], axis=3) #14,512

        inception_4b_1x1 = SNN.SNN_conv(kernel_sizes=1, in_channel=512, out_channel=160, strides=1)(inception_4a_output)
        inception_4b_3x3_reduce = SNN.SNN_conv(kernel_sizes=1, in_channel=512, out_channel=112, strides=1)(inception_4a_output)
        inception_4b_3x3 = SNN.SNN_conv(kernel_sizes=3, in_channel=112, out_channel=224, strides=1)(inception_4b_3x3_reduce)
        inception_4b_5x5_reduce = SNN.SNN_conv(kernel_sizes=1, in_channel=512, out_channel=24, strides=1)(inception_4a_output)
        inception_4b_5x5 = SNN.SNN_conv(kernel_sizes=5, in_channel=24, out_channel=64, strides=1)(inception_4b_5x5_reduce)
        inception_4b_pool = SNN.SNN_maxpool2d(kernel_sizes=(3, 3), strides=(1, 1), padding='same')(inception_4a_output)
        inception_4b_pool_proj = SNN.SNN_conv(kernel_sizes=1, in_channel=512, out_channel=64, strides=1)(inception_4b_pool)
        inception_4b_output = tf.concat([inception_4b_1x1, inception_4b_3x3, inception_4b_5x5, inception_4b_pool_proj], axis=3)

        inception_4c_1x1 = SNN.SNN_conv(kernel_sizes=1, in_channel=512, out_channel=128, strides=1)(inception_4b_output)
        inception_4c_3x3_reduce = SNN.SNN_conv(kernel_sizes=1, in_channel=512, out_channel=128, strides=1)(inception_4b_output)
        inception_4c_3x3 = SNN.SNN_conv(kernel_sizes=3, in_channel=128, out_channel=256, strides=1)(inception_4c_3x3_reduce)
        inception_4c_5x5_reduce = SNN.SNN_conv(kernel_sizes=1, in_channel=512, out_channel=24, strides=1)(inception_4b_output)
        inception_4c_5x5 = SNN.SNN_conv(kernel_sizes=5, in_channel=24, out_channel=64, strides=1)(inception_4c_5x5_reduce)
        inception_4c_pool = SNN.SNN_maxpool2d(kernel_sizes=(3, 3), strides=(1, 1), padding='same')(inception_4b_output)
        inception_4c_pool_proj = SNN.SNN_conv(kernel_sizes=1, in_channel=512, out_channel=64, strides=1)(inception_4c_pool)
        inception_4c_output = tf.concat([inception_4c_1x1, inception_4c_3x3, inception_4c_5x5, inception_4c_pool_proj], axis=3)

        inception_4d_1x1 = SNN.SNN_conv(kernel_sizes=1, in_channel=512, out_channel=112, strides=1)(inception_4c_output)
        inception_4d_3x3_reduce = SNN.SNN_conv(kernel_sizes=1, in_channel=512, out_channel=144, strides=1)(inception_4c_output)
        inception_4d_3x3 = SNN.SNN_conv(kernel_sizes=3, in_channel=144, out_channel=288, strides=1)(inception_4d_3x3_reduce)
        inception_4d_5x5_reduce = SNN.SNN_conv(kernel_sizes=1, in_channel=512, out_channel=32, strides=1)(inception_4c_output)
        inception_4d_5x5 = SNN.SNN_conv(kernel_sizes=5, in_channel=32, out_channel=64, strides=1)(inception_4d_5x5_reduce)
        inception_4d_pool = SNN.SNN_maxpool2d(kernel_sizes=(3, 3), strides=(1, 1), padding='same')(inception_4c_output)
        inception_4d_pool_proj = SNN.SNN_conv(kernel_sizes=1, in_channel=512, out_channel=64, strides=1)(inception_4d_pool)
        inception_4d_output = tf.concat([inception_4d_1x1, inception_4d_3x3, inception_4d_5x5, inception_4d_pool_proj], axis=3)

        inception_4e_1x1 = SNN.SNN_conv(kernel_sizes=1, in_channel=528, out_channel=256, strides=1)(inception_4d_output)
        inception_4e_3x3_reduce = SNN.SNN_conv(kernel_sizes=1, in_channel=528, out_channel=160, strides=1)(inception_4d_output)
        inception_4e_3x3 = SNN.SNN_conv(kernel_sizes=3, in_channel=160, out_channel=320, strides=1)(inception_4e_3x3_reduce)
        inception_4e_5x5_reduce = SNN.SNN_conv(kernel_sizes=1, in_channel=528, out_channel=32, strides=1)(inception_4d_output)
        inception_4e_5x5 = SNN.SNN_conv(kernel_sizes=5, in_channel=32, out_channel=128, strides=1)(inception_4e_5x5_reduce)
        inception_4e_pool = SNN.SNN_maxpool2d(kernel_sizes=(3, 3), strides=(1, 1), padding='same')(inception_4d_output)
        inception_4e_pool_proj = SNN.SNN_conv(kernel_sizes=1, in_channel=528, out_channel=128, strides=1)(inception_4e_pool)
        inception_4e_output = tf.concat([inception_4e_1x1, inception_4e_3x3, inception_4e_5x5, inception_4e_pool_proj], axis=3) #14,832
        pool4_3x3_s2 = SNN.SNN_maxpool2d(kernel_sizes=(3, 3), strides=(2, 2), padding='same')(inception_4e_output) #7,832

        inception_5a_1x1 = SNN.SNN_conv(kernel_sizes=1, in_channel=832, out_channel=256, strides=1)(pool4_3x3_s2)
        inception_5a_3x3_reduce = SNN.SNN_conv(kernel_sizes=1, in_channel=832, out_channel=160, strides=1)(pool4_3x3_s2)
        inception_5a_3x3 = SNN.SNN_conv(kernel_sizes=3, in_channel=160, out_channel=320, strides=1)(inception_5a_3x3_reduce)
        inception_5a_5x5_reduce = SNN.SNN_conv(kernel_sizes=1, in_channel=832, out_channel=32, strides=1)(pool4_3x3_s2)
        inception_5a_5x5 = SNN.SNN_conv(kernel_sizes=5, in_channel=32, out_channel=128, strides=1)(inception_5a_5x5_reduce)
        inception_5a_pool = SNN.SNN_maxpool2d(kernel_sizes=(3, 3), strides=(1, 1), padding='same')(pool4_3x3_s2)
        inception_5a_pool_proj = SNN.SNN_conv(kernel_sizes=1, in_channel=832, out_channel=128, strides=1)(inception_5a_pool)
        inception_5a_output = tf.concat([inception_5a_1x1, inception_5a_3x3, inception_5a_5x5, inception_5a_pool_proj], axis=3) #7,832

        inception_5b_1x1 = SNN.SNN_conv(kernel_sizes=1, in_channel=832, out_channel=384, strides=1)(inception_5a_output)
        inception_5b_3x3_reduce = SNN.SNN_conv(kernel_sizes=1, in_channel=832, out_channel=192, strides=1)(inception_5a_output)
        inception_5b_3x3 = SNN.SNN_conv(kernel_sizes=3, in_channel=192, out_channel=384, strides=1)(inception_5b_3x3_reduce)
        inception_5b_5x5_reduce = SNN.SNN_conv(kernel_sizes=1, in_channel=832, out_channel=48, strides=1)(inception_5a_output)
        inception_5b_5x5 = SNN.SNN_conv(kernel_sizes=5, in_channel=48, out_channel=128, strides=1)(inception_5b_5x5_reduce)
        inception_5b_pool = SNN.SNN_maxpool2d(kernel_sizes=(3, 3), strides=(1, 1), padding='same')(inception_5a_output)
        inception_5b_pool_proj = SNN.SNN_conv(kernel_sizes=1, in_channel=832, out_channel=128, strides=1)(inception_5b_pool)
        inception_5b_output = tf.concat([inception_5b_1x1, inception_5b_3x3, inception_5b_5x5, inception_5b_pool_proj], axis=3) #7,1024

        # average pooling is tricky to SNN-TC since spiking-time average is not very reasonable
        pool5_7x7_s1 = tf.keras.layers.GlobalAveragePooling2D(name='GAPL')(inception_5b_output)  # 1,1024
        final_output = -SNN.SNN_dense(in_size=1024, out_size=config.category_num, name='fc49')(pool5_7x7_s1)
        # add (-) because we need to find the minimum final_output

        super(MyModel, self).__init__(inputs=inputs, outputs=final_output)

    def constraint_cost(self):
        wsc = [tf.reduce_mean(tf.maximum(0.0, 1.0 - tf.reduce_sum(w, 0))) for w in self.trainable_variables]
        l2 = [tf.reduce_mean(tf.square(w)) for w in self.trainable_variables]
        cost = self.K * tf.reduce_sum(wsc) + self.K2 * tf.reduce_sum(l2)
        return cost


my_model = MyModel(weight_k=K, weight_k2=K2, scaling=1.0)

# run a dummy run to construct model before print summary
temp = my_model(np.random.uniform(0, 1, [config.batch_size] + list(config.input_shape)).astype(np.float32))
my_model.summary()

# check tensor sizes, not only weight tensor, but also all the data tensors
# Please change Check_Dim = True in SNN_TC_Modules.py if setting CheckModel=True
CheckModel = False
if CheckModel:  # test to see model layers and size
    def CheckTensorDim(model, print_all=False):
        maxd = [0, 0]
        for i, lay in enumerate(my_model.layers):
            temp = [lay.maxD[0].numpy()[0], lay.maxD[1].numpy()[0]]
            if print_all:
                print('Layer ', str(i), ' max Dim: (', str(temp[0]), ', ', str(temp[1]),
                      ') -> 2**(', str(np.log2(temp[0])), ', ', str(np.log2(temp[1])), ')')
            if maxd[0] < temp[0]: maxd[0] = temp[0]
            if maxd[1] < temp[1]: maxd[1] = temp[1]
        print('Maximum Dimension (Batch, Tensor elements): (', str(maxd[0]), ', ', str(maxd[1]),
              ') -> 2**(', str(np.log2(maxd[0])), ', ', str(np.log2(maxd[1])), ')')
    CheckTensorDim(my_model, print_all=True)


class CosineDecayWithWarmUP(tf.keras.experimental.CosineDecay):
    def __init__(self, initial_learning_rate, decay_steps, alpha=0.0, warm_up_step=0, name=None):
        self.warm_up_step = warm_up_step
        super(CosineDecayWithWarmUP, self).__init__(initial_learning_rate=initial_learning_rate,
                                                    decay_steps=decay_steps,
                                                    alpha=alpha,
                                                    name=name)

    @tf.function
    def __call__(self, step):
        if step <= self.warm_up_step:
            return step / self.warm_up_step * self.initial_learning_rate
        else:
            return super(CosineDecayWithWarmUP, self).__call__(step - self.warm_up_step)


@tf.function
def train_step(model, images, labels, optimizer):
    with tf.GradientTape() as tape:
        prediction = model(images, training=True)
        loss_con = model.constraint_cost()
        loss = SNN.loss_func(logit=prediction, label=labels) + loss_con
        gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, prediction


def train(model, data_iterator, optimizer):
    sum_loss, sum_correct_num, sum_data = 0, 0, 0
    start_time = time.time()
    for i in range(config.iterations_per_epoch):
        images, labels = data_iterator.next()
        loss, prediction = train_step(model, images, labels, optimizer)
        correct_num = tf.reduce_sum(tf.cast(tf.argmin(prediction, axis=1) == tf.argmax(labels, axis=1), tf.float32))

        sum_data += len(labels)
        sum_loss += loss * len(labels)
        sum_correct_num += correct_num
        if i % 10 == 0:
            print('train iter {:4d} data: {:6d}, loss: {:.2f}, accuracy: {:.4f}, time: {:.0f}'
                  .format(i, sum_data, sum_loss/sum_data, sum_correct_num/sum_data, time.time()-start_time))


@tf.function
def test_step(model, images, labels):
    prediction = model(images, training=False)
    loss_con = model.constraint_cost()
    loss = SNN.loss_func(logit=prediction, label=labels) + loss_con
    return loss, prediction


def test(model, data_iterator):
    sum_data, sum_loss, sum_correct_num = 0, 0, 0
    start_time = time.time()
    for i in range(config.test_iterations):
        images, labels = data_iterator.next()
        loss, prediction = test_step(model, images, labels)
        correct_num = tf.reduce_sum(tf.cast(tf.argmin(prediction, axis=1) == tf.argmax(labels, axis=1), tf.float32))

        sum_data += len(labels)
        sum_loss += loss * len(labels)
        sum_correct_num += correct_num
        if i % 10 == 0:
            print('     test iter {:4d} data: {:6d}, loss: {:.2f}, accuracy: {:.4f}, time: {:.0f}'
                  .format(i, sum_data, sum_loss/sum_data, sum_correct_num/sum_data, time.time()-start_time))


# load data
train_data_iterator = train_iterator()
test_data_iterator = test_iterator()

# train
learning_rate_schedules = CosineDecayWithWarmUP(
    initial_learning_rate=config.initial_learning_rate,
    decay_steps=config.epoch_num * config.iterations_per_epoch - config.warm_iterations,
    alpha=config.minimum_learning_rate, warm_up_step=config.warm_iterations)
optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate_schedules, momentum=0.9)

my_time = time.time()

for epoch_num in range(config.epoch_num):
        print('\n epoch ', str(epoch_num), ' / ', str(config.epoch_num),
              ': time = ', str((time.time()-my_time)/3600), ' hours')
        train(my_model, train_data_iterator, optimizer)
        test(my_model, test_data_iterator)

print('training done in ', str((time.time() - my_time) / 3600), ' hours')
