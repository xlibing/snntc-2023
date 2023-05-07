##### same as trainsnn1.py, but use snntc/SNN_TC_Modules.py

# compare: https://github.com/geifmany/cifar10-vgg, ACC 93.56, or https://github.com/SeHwanJoo/cifar10-vgg16, ACC 93.15

# batch 128, lr .001cos, w 0.1:  0.1763/0.1761
# batch 128, lr .001cos, w 0.01: .5855/.5851
# batch 128, lr .001cos, w 0.001: .3494/.3578, .6768/.6524
# batch 128, lr .001cos, w 1/insize: .5929/.5831, 0.4378/.4469, .5170/.5146
# batch 128, lr .001cos, w 10/size: .7016/.6761, 0.1/0.1, 0.791/.727, 0.7/0.6672, 0.7205/0.6868, .7878/.9599 (copy3)
    # v1: 14 hrs
    # v2 (switch x/y): 20 hrs
    # copy3 (sgemm): 13 hrs, 15 hrs
# batch 128, lr .001cos, w 20/size, bias, .9104/.81, .9212/.8096
# batch 128, lr .001cos, w 15/size, bias, .9291/.8265, .9304/.8145
#        32, .7050/.7118, .8347/.7715, 18 hrs,
#       256, .1/.1, .6836/.6708, 14 hrs
#       scale 3: .2/.2
# batch 128, lr .001cos, w 10/size, bias, .9530/.8329, .8194/.7489, .9347/.8184
# batch 128, lr .001cos, w 5/size, bias, .7368/.7158, .8084/.744
# batch 128, lr .001cos, w 1/size, bias, .7419/.6969, .5/.5,
# batch 128, lr .001cos, w 100/size, bias, .5916/.5795, .4989/.474
# batch 128, lr .001cos, w 50/size, bias, .7216/.6781, .6922/.6621

# batch 128, lr .001cos, w 100/size: .5838/.5828, .4164/.4169, .5070/.5182, .4544/.4539

# batch 128, lr .01cos, w 10/size: .1/.1, .16/.16

# batch 128, lr .001cos, w 10/size uniform, .1/.1

# TrainSNNCifar10L.py: biasvolt=true, 128, w 20/size, .9936/.7728, 18 hrs(v1, 250epoch),  .9828/.7802, .9901/.7495

# true VGG16: batch 128, lr .001cos, w 15/size, .91/.81, 12 hrs/200epoch,
# with -input only: .3220/.33754, 13 hrs,
# with max-input: .7737/.7454, .1/.1,
# with (max-input)/(max-min): .9458/.8452, .9242/.8375
# init_par=1: .8598/.7733
# lr = .01: .5104/.5177
# lr (1e-3 -> 1e-5): .9374/.8402, .9416/.8388,
# lr (qe-3 -> 1e-4): .9436/.843, .9487/.8425

import argparse
import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt

# setting hyper-parameters
K = 100
K2 = 1e-2
BIAS = True
BATCH_SIZE = 128
LEARNING_RATE = 0.001
EPOCHS = 200

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='select gpu number')
parser.add_argument('--snnversion', type=int, default=0, help='select snn version')
parser.add_argument('--batchsize', type=int, default=BATCH_SIZE, help='batch size')
args = parser.parse_args()
my_note = 'GPU=0'
if args.gpu != 0:
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    my_note = 'GPU=' + str(args.gpu)
if args.snnversion == 0:
    import SNN_TC_Modules as SNN
    my_note = 'SNN_TC_Modules' + ', ' + my_note

physical_devices = tf.config.experimental.list_physical_devices('GPU')
for gpu in physical_devices:
    try:
        tf.config.experimental.set_memory_growth(device=gpu, enable=True)
    except:
        pass

# prepare dataset with data augomentation
dataset = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = dataset.load_data()
x_train, x_test = x_train.astype(np.float32) / 255.0, x_test.astype(np.float32) / 255.0
y_train = y_train[:, 0].astype(np.int64)
y_test = y_test[:, 0].astype(np.int64)

total_data = x_train.shape[0]
num_class = y_test.max() + 1
print(total_data, num_class)

norm_mean = (0.4914, 0.4822, 0.4465)
norm_var = (0.04092529, 0.03976036, 0.040401)
train_data_aug = tf.keras.Sequential([
    tf.keras.layers.ZeroPadding2D(padding=4), # pad with zeros so random crop is valid (Pytorch: RandomCrop(Pad=4)
    tf.keras.layers.RandomCrop(32, 32),
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.Normalization(mean=norm_mean, variance=norm_var)
])
test_data_aug = tf.keras.Sequential([
    tf.keras.layers.Normalization(mean=norm_mean, variance=norm_var)
])

# need max pixel value in order for TTFS encoding to spiking time t_i with t_i>0 and exp(t_i)>1
pixel_value_range =[np.min((0 - np.array(norm_mean))/np.sqrt(np.array(norm_var))),
                    np.max((1 - np.array(norm_mean))/np.sqrt(np.array(norm_var)))]
print('normalized pixel range = [', pixel_value_range[0], ', ', pixel_value_range[1], ']')

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(len(x_train)).batch(BATCH_SIZE)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(100)

train_ds = train_ds.map(lambda x, y: (train_data_aug(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.map(lambda x, y: (test_data_aug(x), y), num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

# prepare SNN model
class MyModel(tf.keras.Model):
    def __init__(self, weight_k=100, weight_k2=1e-2, scale=1.0):

        self.weight_K = weight_k
        self.weight_K2 = weight_k2
        self.scale = scale

        input = tf.keras.Input(shape=(32, 32, 3), dtype=tf.float32)
        x = tf.exp(self.scale * (pixel_value_range[1] - input) / (pixel_value_range[1]-pixel_value_range[0]))  # make sure ti >= 0, and x >= 1
        layerout1 = SNN.SNN_conv(kernel_sizes=3, in_channel=3, out_channel=64, strides=1, biasvolt=BIAS)(x)
        layerout2 = SNN.SNN_conv(kernel_sizes=3, in_channel=64, out_channel=64, strides=1, biasvolt=BIAS)(layerout1)
        maxpool1 = SNN.SNN_maxpool2d(kernel_sizes=(2, 2), strides=(2, 2), name='maxpool1')(layerout2)  # (32, 32, 64)
        
        layerout3 = SNN.SNN_conv(kernel_sizes=3, in_channel=64, out_channel=128, strides=1, biasvolt=BIAS)(maxpool1)
        layerout4 = SNN.SNN_conv(kernel_sizes=3, in_channel=128, out_channel=128, strides=1, biasvolt=BIAS)(layerout3)
        maxpool2 = SNN.SNN_maxpool2d(kernel_sizes=(2, 2), strides=(2, 2), name='maxpool2')(layerout4)  # (16, 16, 128)
        
        layerout5 = SNN.SNN_conv(kernel_sizes=3, in_channel=128, out_channel=256, strides=1, biasvolt=BIAS)(maxpool2)
        layerout6 = SNN.SNN_conv(kernel_sizes=3, in_channel=256, out_channel=256, strides=1, biasvolt=BIAS)(layerout5)
        layerout7 = SNN.SNN_conv(kernel_sizes=3, in_channel=256, out_channel=256, strides=1, biasvolt=BIAS)(layerout6)
        maxpool3 = SNN.SNN_maxpool2d(kernel_sizes=(2, 2), strides=(2, 2), name='maxpool3')(layerout7)  # (8, 8, 256)
        
        layerout8 = SNN.SNN_conv(kernel_sizes=3, in_channel=256, out_channel=512, strides=1, biasvolt=BIAS)(maxpool3)
        layerout9 = SNN.SNN_conv(kernel_sizes=3, in_channel=512, out_channel=512, strides=1, biasvolt=BIAS)(layerout8)
        layerout10 = SNN.SNN_conv(kernel_sizes=3, in_channel=512, out_channel=512, strides=1, biasvolt=BIAS)(layerout9)
        maxpool4 = SNN.SNN_maxpool2d(kernel_sizes=(2, 2), strides=(2, 2), name='maxpool4')(layerout10)  # (4, 4, 512)
        
        layerout11 = SNN.SNN_conv(kernel_sizes=3, in_channel=512, out_channel=512, strides=1, biasvolt=BIAS)(maxpool4)
        layerout12 = SNN.SNN_conv(kernel_sizes=3, in_channel=512, out_channel=512, strides=1, biasvolt=BIAS)(layerout11)
        layerout13 = SNN.SNN_conv(kernel_sizes=3, in_channel=512, out_channel=512, strides=1, biasvolt=BIAS)(layerout12)  # (2, 2, 1024)
        maxpool5 = SNN.SNN_maxpool2d(kernel_sizes=(2, 2), strides=(2, 2), name='maxpool5')(layerout13)  # (1, 1, 1024)
        
        layerout14 = SNN.SNN_dense(in_size=512, out_size=512, biasvolt=BIAS)(tf.reshape(maxpool5, [tf.shape(x)[0], -1]))
        layerout15 = SNN.SNN_dense(in_size=512, out_size=10, biasvolt=BIAS)(layerout14)

        # return layerout15
        super(MyModel, self).__init__(inputs=input, outputs=layerout15)

    def constraint_cost(self, cost_index=0):
        wsc = [tf.reduce_mean(tf.maximum(0.0, 1.0 - tf.reduce_sum(w, 0))) for w in self.trainable_variables]
        l2 = [tf.reduce_mean(tf.square(w)) for w in self.trainable_variables]
        if cost_index == 0:
            cost = self.weight_K * tf.reduce_sum(wsc) + self.weight_K2 * tf.reduce_sum(l2)
        elif cost_index == 1:
            cost = wsc
        elif cost_index == 2:
            cost = l2
        return cost


my_model = MyModel(weight_k=K, weight_k2=K2, scale=1.0)

# load pre-trained weights for continue training
# print('retrieve weights in cifar10_best_checkpoint, start training from there')
# my_model.load_weights('cifar10_best_checkpoint')

# run a dummy run to construct model before print summary
temp = my_model(np.random.uniform(0, 1, (BATCH_SIZE, 32, 32, 3)).astype(np.float32))
# function for check the maximum tensor dimension using the dummy run.
# Please change Check_Dim = True in tf2_SNN_Models_*.py
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
#CheckTensorDim(my_model, print_all=True)
# my_model.summary()

trainableParams = np.sum([np.prod(v.get_shape()) for v in my_model.trainable_weights]).astype(int)
nonTrainableParams = np.sum([np.prod(v.get_shape()) for v in my_model.non_trainable_weights]).astype(int)
totalParams = trainableParams + nonTrainableParams
# print(f'{len(my_model.trainable_variables)} layers, with {totalParams} parameters ({trainableParams} trainable, {nonTrainableParams} nontrainable)')
print("{} layers, {:,} parameters ({:,} trainable, {:,} nontrainable)".format(
    len(my_model.trainable_variables), totalParams, trainableParams, nonTrainableParams))

# Training setup
decay_steps = x_train.shape[0] // BATCH_SIZE * EPOCHS
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(LEARNING_RATE, decay_steps, alpha=1)
# opt = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9)
opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

@tf.function
def train_step(model, images, labels):
    outputs = tf.one_hot(labels, 10)
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss_con = model.constraint_cost()
        loss = SNN.loss_func(logit=predictions, label=outputs) + loss_con
    gradients = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(gradients, model.trainable_variables))
    acc = tf.reduce_sum(tf.cast(tf.argmin(predictions, axis=1) == labels, tf.float32))
    gradrange = [tf.reduce_mean([tf.reduce_mean(tf.abs(g)) for g in gradients]),
                 tf.reduce_max([tf.reduce_max(tf.abs(g)) for g in gradients])]
    return loss, acc, gradrange


@tf.function
def test_step(model, images, labels):
    outputs = tf.one_hot(labels, 10)
    predictions = model(images, training=False)
    loss = SNN.loss_func(logit=predictions, label=outputs) + model.constraint_cost()
    acc = tf.reduce_sum(tf.cast(tf.argmin(predictions, axis=1) == labels, tf.float32))
    return loss, acc, predictions

# training
start_time = time.time()
best_test_acc = [0.0, 0]  # save best test ACC and its epoch
train_loss, train_acc, test_loss, test_acc = [], [], [], []
for epoch in range(EPOCHS):
    loss, acc, accunum, max_grad = 0.0, 0.0, 0, 0.0
    for i, (images, labels) in enumerate(train_ds):
        batch_loss, batch_acc, gradrange = train_step(my_model, images, labels)
        loss += batch_loss * len(labels)
        acc += batch_acc
        accunum += len(labels)
        if (i % 100 == 0) | (accunum == total_data):
            print('Epoch ', str(epoch), ' Train (loss, acc) = (', round(float(loss) / accunum, 4),
                  ', ', round(float(acc) / accunum, 4), '),  ', accunum, '/', total_data,
                  ' lr = ', opt._decayed_lr('float32').numpy(),
                  ' time = ', int(time.time()-start_time)
                  )
        if gradrange[1].numpy() > max_grad:  # check maximum grad
            max_grad = gradrange[1].numpy()
        if epoch == -1:
            weight_range = [tf.reduce_mean([tf.reduce_mean(tf.abs(w)) for w in my_model.trainable_weights]).numpy(),
                    tf.reduce_max([tf.reduce_max(tf.abs(w)) for w in my_model.trainable_weights]).numpy()]
            print(str(i), ': weight range (mean,max): (', round(weight_range[0], 4), ', ', round(weight_range[1], 4),
                  '). grad range: (', round(gradrange[0].numpy(), 4), ', ', round(gradrange[1].numpy(), 4),
                  '). (wsc,l2): (', round(tf.reduce_sum(my_model.constraint_cost(1)).numpy(), 4),
                  ', ', round(tf.reduce_sum(my_model.constraint_cost(2)).numpy(), 4), ')')

    train_loss.append(float(loss) / accunum)  # each sample's loss (batch's mean loss + weight cost), running average
    train_acc.append(float(acc) / accunum)    # train_acc is running average, smaller than test_acc

    # for debugging: check output activation because all neurons may be dead sometimes
    train_activation = np.array(my_model(images))
    train_activation1 = train_activation[train_activation < (1e5-0.1)]
    train_activation_max = 0
    if len(train_activation1) > 0:
        train_activation_max = np.max(np.abs(train_activation1))

    loss, acc, accunum, pred_all = 0., 0., 0, []
    for test_images, test_labels in test_ds:
        batch_loss, batch_acc, pred = test_step(my_model, test_images, test_labels)
        loss += batch_loss * len(test_labels)
        acc += batch_acc
        accunum += len(test_labels)
        pred_all.append(pred)

    pred_all = np.array(tf.concat(pred_all, axis=0))
    pred_all = pred_all[pred_all<(1e5-0.1)]
    if len(pred_all[:]) > 0:
        print('pred logit distribution (min, median, max) = ', round(np.min(pred_all), 4),
              ', ', round(np.median(pred_all), 4), ', ', round(np.max(pred_all), 4))
    else:
        print('pred logit distribution (min, median, ax) = 1e5')

    if accunum != y_test.shape[0]:
        print(f'not all test data are used: {accunum} out of {y_test.shape[0]} were used')
    test_loss.append(float(loss) / accunum)
    test_acc.append(float(acc) / accunum)

    weight_range = [tf.reduce_mean([tf.reduce_mean(tf.abs(w)) for w in my_model.trainable_weights]).numpy(),
                    tf.reduce_max([tf.reduce_max(tf.abs(w)) for w in my_model.trainable_weights]).numpy()]
    print('weight (mean,max) = (', round(weight_range[0], 4), ', ', round(weight_range[1], 4),
          '), (wsc, l2) = (', round(tf.reduce_sum(my_model.constraint_cost(1)).numpy(), 4),
          ', ', round(tf.reduce_sum(my_model.constraint_cost(2)).numpy(), 4), ') '
          ', max|grad| = ', round(max_grad, 4),
          ', max|out| = ', round(train_activation_max, 4))
    # print('weight  grad: mean = ', str(gradrange[0].numpy())[:10], ', max = ', str(gradrange[1].numpy()))

    print('average: Train (loss, acc) = (', round(train_loss[-1], 4), ', ', round(train_acc[-1], 4),
          '),  Validation (loss, acc) = (', round(test_loss[-1], 4), ', ', round(test_acc[-1], 4),
          '), time = ', int(time.time()-start_time), '\n')

    # save the weights of models with the highest valication acc, as well as final model weights
    if test_acc[-1] > best_test_acc[0]:
        best_test_acc = [test_acc[-1], epoch]
        if test_acc[-1] > 0.8:
            my_model.save_weights('cifar10_best_checkpoint')
            print(f'weights saved to cifar10_best_checkpoint with ACC {best_test_acc[0]}')

# my_model.save_weights('cifar10_last_checkpoint')

end_time = time.time() - start_time
print(my_note, 'time = ', str(end_time/3600), ' hours. best test acc = ',
      str(best_test_acc[0]), ', @ epoch ', str(best_test_acc[1]))

maxnorm = max(np.array(train_loss).max(), np.array(test_loss).max())
plt.plot(np.array(train_loss)/maxnorm, 'b--', label='train_loss '+str(np.array(train_loss).max())[:8])
plt.plot(train_acc, 'r--', label='train_acc '+str(np.array(train_acc).max())[:6])
plt.plot(np.array(test_loss)/maxnorm, 'b-', label='test_loss '+str(np.array(test_loss).max())[:8])
plt.plot(test_acc, 'r-', label='test_acc '+str(np.array(test_acc).max())[:6])
plt.legend()
plt.title(my_note)
plt.grid(True)
plt.show()



