# SNN with Temporal Coding over MNIST dataset
# a sample code showing how to use SNN-TC modules
import numpy as np
import tensorflow as tf
import time

import SNN_TC_Modules as SNN

# observe GPU memory usage
physical_devices = tf.config.experimental.list_physical_devices('GPU')
for my_gpu in physical_devices:
    tf.config.experimental.set_memory_growth(device=my_gpu, enable=True)

# SNN model hyper-parameters
K = 100
K2 = 1e-2
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
NUM_CLASSES = 10
EPOCHS = 10

# MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train[..., tf.newaxis].astype("float32")
x_test = x_test[..., tf.newaxis].astype("float32")
y_train = y_train.astype(np.int64)
y_test = y_test.astype(np.int64)

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(x_train.shape[0]).batch(BATCH_SIZE)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH_SIZE)


# SNN model
class MyModel(tf.keras.Model):
    def __init__(self, weight_k=100, weight_k2=1e-2):
        super(MyModel, self).__init__()
        self.weight_K = weight_k
        self.weight_K2 = weight_k2

        self.layer1 = SNN.SNN_conv(in_channel=1, out_channel=32, kernel_sizes=3, strides=2)
        self.layer2 = SNN.SNN_conv(in_channel=32, out_channel=64, kernel_sizes=3, strides=2)
        self.layer3 = SNN.SNN_dense(in_size=7*7*64, out_size=1024)
        self.layer4 = SNN.SNN_dense(in_size=1024, out_size=10)

    # noinspection PyCallingNonCallable
    def call(self, x):
        layer_out1 = self.layer1(x)
        layer_out2 = self.layer2(layer_out1)
        layer_out3 = self.layer3(tf.reshape(layer_out2, [tf.shape(layer_out2)[0], -1]))
        layer_out4 = self.layer4(layer_out3)
        return layer_out4

    def constraint_cost(self):
        wsc = [tf.reduce_mean(tf.maximum(0.0, 1.0 - tf.reduce_sum(w, 0))) for w in self.trainable_variables]
        l2 = [tf.reduce_mean(tf.square(w)) for w in self.trainable_variables]
        cost = self.weight_K * tf.reduce_sum(wsc) + self.weight_K2 * tf.reduce_sum(l2)
        return cost


with tf.device('/device:GPU:0'):
    my_model = MyModel(weight_k=K, weight_k2=K2)

# run a dummy run to construct model before print summary
temp = my_model(np.random.uniform(0, 1, (1, 28, 28, 1)).astype(np.float32))
my_model.summary()

# training modules
opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)


@tf.function
def train_step(model, images, labels):
    inputs = tf.exp(1.0 * (1.0 - images))
    outputs = tf.one_hot(labels, NUM_CLASSES)
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss_con = model.constraint_cost()
        loss = SNN.loss_func(logit=predictions, label=outputs) + loss_con
    gradients = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(gradients, model.trainable_variables))
    acc = tf.reduce_sum(tf.cast(tf.argmin(predictions, axis=1) == labels, tf.float32))
    return loss, acc


@tf.function
def test_step(model, images, labels):
    inputs = tf.exp(1.0 * (1.0 - images))
    outputs = tf.one_hot(labels, NUM_CLASSES)
    predictions = model(inputs, training=False)
    loss = SNN.loss_func(logit=predictions, label=outputs) + model.constraint_cost()
    acc = tf.reduce_sum(tf.cast(tf.argmin(predictions, axis=1) == labels, tf.float32))
    return loss, acc


# train the model
start_time = time.time()
train_loss, train_acc, test_loss, test_acc = [], [], [], []
for epoch in range(EPOCHS):
    loss, acc, samples = 0.0, 0.0, 0
    for images, labels in train_ds:
        batch_loss, batch_acc = train_step(my_model, images, labels)
        loss += batch_loss * len(labels)
        acc += batch_acc
        samples += len(labels)
        if (samples == BATCH_SIZE) | (samples//BATCH_SIZE % 100 == 0) | (samples == 60000):
            print('Epoch ', str(epoch), ' Train (loss, acc) = (', str(float(loss)/samples)[:6],
                  ', ', str(float(acc)/samples)[:6], '),  ', str(samples), '/', '60000')

    train_loss.append(float(loss) / samples)  # each sample's loss (CE loss + weight cost)
    train_acc.append(float(acc) / samples)    # train_acc is running average, smaller than test_acc

    loss, acc, samples = 0., 0., 0
    for test_images, test_labels in test_ds:
        batch_loss, batch_acc = test_step(my_model, test_images, test_labels)
        loss += batch_loss * len(test_labels)
        acc += batch_acc
        samples += len(test_labels)
    test_loss.append(float(loss) / samples)
    test_acc.append(float(acc) / samples)

    print('Epoch ', str(epoch), ' Train (loss, acc) = (', str(train_loss[-1])[:6], ', ', str(train_acc[-1])[:6],
          '),  Test (loss, acc) = (', str(test_loss[-1])[:6], ', ', str(test_acc[-1])[:6], ') \n')

end_time = time.time() - start_time
print(f'finish training in {end_time//60} minutes')
