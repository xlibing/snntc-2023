# SNN-TC VGG16 Model for CIFAR10 dataset
import argparse
import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt
import SNN_TC_Modules as SNN

# setting hyper-parameters
K = 100
K2 = 1e-2
BIAS = True
BATCH_SIZE = 128
LEARNING_RATE = 0.001
EPOCHS = 200

my_note = 'SNN_TC VGG16 Model'

physical_devices = tf.config.experimental.list_physical_devices('GPU')
for gpu in physical_devices:
    try:
        tf.config.experimental.set_memory_growth(device=gpu, enable=True)
    except:
        pass

# prepare dataset with data augmentation
dataset = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = dataset.load_data()
x_train, x_test = x_train.astype(np.float32) / 255.0, x_test.astype(np.float32) / 255.0
y_train = y_train[:, 0].astype(np.int64)
y_test = y_test[:, 0].astype(np.int64)

total_data = x_train.shape[0]
num_class = y_test.max() + 1
print(total_data, num_class)

train_data_aug = tf.keras.Sequential([
    tf.keras.layers.ZeroPadding2D(padding=4), # pad with zeros so random crop is valid (Pytorch: RandomCrop(Pad=4))
    tf.keras.layers.RandomCrop(32, 32),
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.Normalization(mean=(0.4914, 0.4822, 0.4465), variance=(0.04092529, 0.03976036, 0.040401))
])
test_data_aug = tf.keras.Sequential([
    tf.keras.layers.Normalization(mean=(0.4914, 0.4822, 0.4465), variance=(0.04092529, 0.03976036, 0.040401))
])

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
        x = tf.exp(self.scale * (1.0 - input))
        layerout1 = SNN.SNN_conv(kernel_sizes=3, in_channel=3, out_channel=64, strides=1, biasvolt=BIAS)(x)
        layerout2 = SNN.SNN_conv(kernel_sizes=3, in_channel=64, out_channel=64, strides=1, biasvolt=BIAS)(layerout1)
        maxpool1 = SNN.SNN_maxpool2d((2, 2), (2, 2), 'maxpool1')(layerout2)  # (32, 32, 64)
        
        layerout3 = SNN.SNN_conv(kernel_sizes=3, in_channel=64, out_channel=128, strides=1, biasvolt=BIAS)(maxpool1)
        layerout4 = SNN.SNN_conv(kernel_sizes=3, in_channel=128, out_channel=128, strides=1, biasvolt=BIAS)(layerout3)
        maxpool2 = SNN.SNN_maxpool2d((2, 2), (2, 2), 'maxpool2')(layerout4)  # (16, 16, 128)
        
        layerout5 = SNN.SNN_conv(kernel_sizes=3, in_channel=128, out_channel=256, strides=1, biasvolt=BIAS)(maxpool2)
        layerout6 = SNN.SNN_conv(kernel_sizes=3, in_channel=256, out_channel=256, strides=1, biasvolt=BIAS)(layerout5)
        layerout7 = SNN.SNN_conv(kernel_sizes=3, in_channel=256, out_channel=256, strides=1, biasvolt=BIAS)(layerout6)
        maxpool3 = SNN.SNN_maxpool2d((2, 2), (2, 2), 'maxpool3')(layerout7)  # (8, 8, 256)
        
        layerout8 = SNN.SNN_conv(kernel_sizes=3, in_channel=256, out_channel=512, strides=1, biasvolt=BIAS)(maxpool3)
        layerout9 = SNN.SNN_conv(kernel_sizes=3, in_channel=512, out_channel=512, strides=1, biasvolt=BIAS)(layerout8)
        layerout10 = SNN.SNN_conv(kernel_sizes=3, in_channel=512, out_channel=512, strides=1, biasvolt=BIAS)(layerout9)
        maxpool4 = SNN.SNN_maxpool2d((2, 2), (2, 2), 'maxpool4')(layerout10)  # (4, 4, 512)
        
        layerout11 = SNN.SNN_conv(kernel_sizes=3, in_channel=512, out_channel=1024, strides=1, biasvolt=BIAS)(maxpool4)
        layerout12 = SNN.SNN_conv(kernel_sizes=3, in_channel=1024, out_channel=1024, strides=1, biasvolt=BIAS)(layerout11)
        layerout13 = SNN.SNN_conv(kernel_sizes=3, in_channel=1024, out_channel=1024, strides=1, biasvolt=BIAS)(layerout12)  # (2, 2, 1024)
        maxpool5 = SNN.SNN_maxpool2d((2, 2), (2, 2), 'maxpool5')(layerout13)  # (1, 1, 1024)
        
        layerout14 = SNN.SNN_dense(in_size=1024, out_size=4096, biasvolt=BIAS)(tf.reshape(maxpool5, [tf.shape(x)[0], -1]))
        layerout15 = SNN.SNN_dense(in_size=4096, out_size=4096, biasvolt=BIAS)(layerout14)
        layerout16 = SNN.SNN_dense(in_size=4096, out_size=512, biasvolt=BIAS)(layerout15)
        layerout17 = SNN.SNN_dense(in_size=512, out_size=10, biasvolt=BIAS)(layerout16)

        # return layerout17
        super(MyModel, self).__init__(inputs=input, outputs=layerout17)

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

my_model = MyModel(weight_k=K, weight_k2=K2)

# run a dummy run to construct model before print summary
temp = my_model(np.random.uniform(0, 1, (BATCH_SIZE, 32, 32, 3)).astype(np.float32))
my_model.summary()


# function for checking the maximum tensor dimension using the dummy run.
# sometimes we would like to see if there is any tensor size becomes too big
# Please change Check_Dim = True in SNN_TC_Modules.py if want to do this
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


# Training setup: we need Adam due to gradient normalization. SNN-TC may have too large gradients.
decay_steps = x_train.shape[0] // BATCH_SIZE * EPOCHS
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(LEARNING_RATE, decay_steps)
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
    return loss, acc


@tf.function
def test_step(model, images, labels):
    outputs = tf.one_hot(labels, 10)
    predictions = model(images, training=False)
    loss = SNN.loss_func(logit=predictions, label=outputs) + model.constraint_cost()
    acc = tf.reduce_sum(tf.cast(tf.argmin(predictions, axis=1) == labels, tf.float32))
    return loss, acc, predictions


# training
start_time = time.time()
train_loss, train_acc, test_loss, test_acc = [], [], [], []
for epoch in range(EPOCHS):
    loss, acc, accunum = 0.0, 0.0, 0
    for i, (images, labels) in enumerate(train_ds):
        batch_loss, batch_acc = train_step(my_model, images, labels)
        loss += batch_loss * len(labels)
        acc += batch_acc
        accunum += len(labels)
        if (i % 100 == 0) | (accunum == total_data):
            print('Epoch ', str(epoch), ' Train (loss, acc) = (', str(float(loss) / accunum)[:6],
                  ', ', str(float(acc) / accunum)[:6], '),  ', str(accunum), '/', str(total_data),
                  ' lr = ', str(opt._decayed_lr('float32').numpy()),
                  ' time = ', str(int(time.time()-start_time))
                  )

    train_loss.append(float(loss) / accunum)  # each sample's loss (batch's mean loss + weight cost), running average
    train_acc.append(float(acc) / accunum)    # train_acc is running average, smaller than test_acc

    loss, acc, accunum, pred_all = 0., 0., 0, []
    for test_images, test_labels in test_ds:
        batch_loss, batch_acc, pred = test_step(my_model, test_images, test_labels)
        loss += batch_loss * len(test_labels)
        acc += batch_acc
        accunum += len(test_labels)
        pred_all.append(pred)

    if accunum != y_test.shape[0]:
        print(f'not all test data are used: {accunum} out of {y_test.shape[0]} were used')
    test_loss.append(float(loss) / accunum)
    test_acc.append(float(acc) / accunum)

    print('Validation: Train (loss, acc) = (', str(train_loss[-1])[:6], ', ', str(train_acc[-1])[:6],
          '),  Test (loss, acc) = (', str(test_loss[-1])[:6], ', ', str(test_acc[-1])[:6],
          '), time = ', str(int(time.time()-start_time)), '\n')

end_time = time.time() - start_time
print(my_note, 'time = ', str(end_time/3600), ' hours')

maxnorm = max(np.array(train_loss).max(), np.array(test_loss).max())
plt.plot(np.array(train_loss)/maxnorm, 'b--', label='train_loss '+str(np.array(train_loss).max())[:8])
plt.plot(train_acc, 'r--', label='train_acc '+str(np.array(train_acc).max())[:6])
plt.plot(np.array(test_loss)/maxnorm, 'b-', label='test_loss '+str(np.array(test_loss).max())[:8])
plt.plot(test_acc, 'r-', label='test_acc '+str(np.array(test_acc).max())[:6])
plt.legend()
plt.title(my_note)
plt.grid(True)
plt.show()



