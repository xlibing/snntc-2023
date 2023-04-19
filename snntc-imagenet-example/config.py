# Training config
category_num = 1000
batch_size = 64
test_batch_size = 50
input_shape = (224, 224, 3)
weight_decay = 1e-4
label_smoothing = 0.

train_num = 1281167
test_num = 50000
iterations_per_epoch = int(train_num / batch_size)
test_iterations = int(test_num / test_batch_size)
warm_iterations = iterations_per_epoch

initial_learning_rate = 0.05
minimum_learning_rate = 0.0001
epoch_num = 10

log_file = 'GoogleNet_result.txt'
load_weight_file = None  # 'result/weight/ResNet_50_v2.h5' #None
save_weight_file = 'GoogleNet_result.h5'

# Dataset config
train_list_path = 'train_label.txt'
test_list_path = 'validation_label.txt'
train_data_path = '/home/ab/mydata/imagenet/ILSVRC2012_img_train'
test_data_path = '/home/ab/mydata/imagenet/ILSVRC2012_img_val'

# Augmentation config
# From 'Bag of tricks for image classification with convolutional neural networks'
# Or https://github.com/dmlc/gluon-cv
short_side_scale = (256, 384)
aspect_ratio_scale = (0.8, 1.25)
hue_delta = (-36, 36)
saturation_scale = (0.6, 1.4)
brightness_scale = (0.6, 1.4)
pca_std = 0.1

mean = [103.939, 116.779, 123.68]
std = [58.393, 57.12, 57.375]
eigval = [55.46, 4.794, 1.148]
eigvec = [[-0.5836, -0.6948, 0.4203],
          [-0.5808, -0.0045, -0.8140],
          [-0.5675, 0.7192, 0.4009]]

