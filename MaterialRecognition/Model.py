# compatibility with python 2 and 3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import tensorflow.contrib as tfc
import os



class Record(object):
    pass


N = 200
K = 6
L = 10
M = 10
P = 14
R = 12
img_heigth = 240
img_width = 320
img_depth = 1
img_len = img_width * img_heigth * img_depth
num_classes = 15
batch_size = 10


# train

spectr_vect_train = [os.path.join('pliki/train/spectrograms_train', 'output%d.png' % i) for i in range(3000)]

filenames_train = ("pliki/train/Labels_train.txt")

label_vect_train = np.genfromtxt(filenames_train, unpack=True, dtype=None)

# test

spectr_vect_test = [os.path.join('pliki/test/spectrograms_test', 'output%d.png' % i) for i in range(378)]

filenames_test = ("pliki/test/Labels_test.txt")

label_vect_test = np.genfromtxt(filenames_test, unpack=True, dtype=None)

# validation

spectr_vect_val = [os.path.join('pliki/validation/spectrograms_val', 'output%d.png' % i) for i in range(378)]

filenames_val = ("pliki/validation/Labels_val.txt")

label_vect_val = np.genfromtxt(filenames_val, unpack=True, dtype=None)


def parse(img_path, label):
    img = tf.read_file(img_path)
    img = tf.image.decode_png(img, 3)
    # img.set_shape([None, None, None, 3])
    # tf.image.resize_nearest_neighbor
    return img, label


# my_img = [/home/user/.../img.png, ...]
# label = [3, 4, 1, 5, ...]

# train iterator

train = tf.data.Dataset.from_tensor_slices((spectr_vect_train, label_vect_train))
train = train.shuffle(50000)
train = train.map(parse, 8)
train = train.batch(batch_size)
train = train.prefetch(batch_size)

train_iterator = train.make_initializable_iterator()
train_imgs, train_labels = train_iterator.get_next()
train_labels_one_hot = tf.one_hot(train_labels, num_classes, 1, 0)

# test iterator

test = tf.data.Dataset.from_tensor_slices((spectr_vect_test, label_vect_test))
test = test.shuffle(50000)
test = test.map(parse, 8)
test = test.batch(batch_size)
test = test.prefetch(batch_size)

test_iterator = test.make_initializable_iterator()
test_imgs, test_labels = test_iterator.get_next()
test_labels_one_hot = tf.one_hot(test_labels, num_classes, 1, 0)

# validation iterator

val = tf.data.Dataset.from_tensor_slices((spectr_vect_val, label_vect_val))
val = val.shuffle(50000)
val = val.map(parse, 8)
val = val.batch(batch_size)
val = val.prefetch(batch_size)

val_iterator = val.make_initializable_iterator()
val_imgs, val_labels = val_iterator.get_next()
val_labels_one_hot = tf.one_hot(val_labels, num_classes, 1, 0)

def weights(output_size, nazwa):
    weight = tf.get_variable(nazwa, output_size, tf.float32, tfc.layers.xavier_initializer())
    return weight


def biases(output_size, nazwa):
    bias = tf.get_variable(nazwa, output_size, tf.float32, tfc.layers.xavier_initializer())
    return bias


def convolutional_layer(input, weigth, bias):
    layer = tf.nn.conv2d(input, weigth, strides=[1, 1, 1, 1], padding='SAME') + bias
    return layer


def convolutional_layer_2(input, weigth, bias):
    layer = tf.nn.conv2d(input, weigth, strides=[1, 2, 2, 1], padding='SAME') + bias
    return layer


def max_pool(input):
    layer = tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    return layer


def weight_all():
    return


def biase_all():
    return


def model1(img):
    with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
        W1 = weights([5, 5, 3, K], "1_w")
        B1 = biases(K, "first_bias")
        W2 = weights([5, 5, K, L], "2_w")
        B2 = biases(L, "sec_bias")
        W3 = weights([4, 4, L, M], "3_w")
        B3 = biases(M, "third_bias")
        W4 = weights([4, 4, M, P], "4_w")
        B4 = biases(P, "fourth_bias")
        W5 = weights([40 * 30 * P, N], "5_w")
        B5 = biases(N, "fifth_bias")
        W6 = weights([N, 15], "6_w")
        B6 = biases(15, "sixth_bias")

        print(tf.shape(img))
        img = tf.cast(img, tf.float32)
        #img = tf.reshape(img, shape=[-1, 320, 240, 1])

        Y1 = tf.nn.relu(convolutional_layer(img, W1, B1))

        # output 640x480  Y1 = 10x640x480x6 W2 = 5,5,6,10

        Y2 = convolutional_layer(Y1, W2, B2)
        Y2 = max_pool(Y2)
        Y2 = tf.nn.relu(Y2)

        # output 320x240  Y2 = 10x320x240x10 W3 = 5,5,10,14
        Y3 = convolutional_layer(Y2, W3, B3)
        Y3 = max_pool(Y3)
        Y3 = tf.nn.relu(Y3)

        # output 160x120  Y3 = 10x160x120x14 W4 = 5,5,14,18
        Y4 = convolutional_layer_2(Y3, W4, B4)
        Y4 = max_pool(Y4)
        Y4 = tf.nn.relu(Y4)

        # output 80x60  Y4 = 10x80x60x14
        YY = tf.reshape(Y4, shape=[-1, 40 * 30 * P])

        fc1 = tf.nn.relu(tf.matmul(YY, W5) + B5)

        output = tf.matmul(fc1, W6) + B6
        print(output)
        pred = tf.nn.softmax(output)
        print(pred)

        return output, pred


train_output, train_pred = model1(train_imgs)
test_output, test_pred = model1(test_imgs)
validation_output, validation_pred = model1(val_imgs)

# cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=train_labels, logits=train_output)

# train


cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=train_output, labels=train_labels_one_hot)

loss = tf.reduce_mean(cross_entropy)

optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)

our_pred = (tf.argmax(train_pred, 1))

correct_prediction = tf.equal(tf.argmax(train_pred, 1), tf.cast(train_labels, tf.int64))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# test

cross_entropy_t = tf.nn.softmax_cross_entropy_with_logits_v2(logits=test_output, labels=test_labels_one_hot)

loss_t = tf.reduce_mean(cross_entropy_t)

correct_prediction_t = tf.equal(tf.argmax(test_pred, 1), tf.cast(test_labels, tf.int64))

accuracy_t = tf.reduce_mean(tf.cast(correct_prediction_t, tf.float32))

# validation

cross_entropy_v = tf.nn.softmax_cross_entropy_with_logits_v2(logits=validation_output, labels=val_labels_one_hot)

loss_v = tf.reduce_mean(cross_entropy_v)

correct_prediction_v = tf.equal(tf.argmax(validation_pred, 1), tf.cast(val_labels, tf.int64))

# tensorboard

accuracy_v = tf.reduce_mean(tf.cast(correct_prediction_v, tf.float32))
train_accuracy = tf.summary.scalar('metrics/accuracy', accuracy)
train_loss = tf.summary.scalar('metrics/loss', loss)
stats = tf.summary.merge([train_accuracy, train_loss])

test_accuracy = tf.summary.scalar('metrics/accuracy_t', accuracy_t)
test_loss = tf.summary.scalar('metrics/loss_t', loss_t)
stats_t = tf.summary.merge([test_accuracy, test_loss])

validation_accuracy = tf.summary.scalar('metrics/accuracy_v', accuracy_v)
validation_loss = tf.summary.scalar('metrics/loss_v', loss_v)
stats_v = tf.summary.merge([validation_accuracy, validation_loss])

fwtrain = tf.summary.FileWriter(logdir='./training', graph=tf.get_default_graph())
fwtest = tf.summary.FileWriter(logdir='./testing', graph=tf.get_default_graph())
fwvalidation = tf.summary.FileWriter(logdir='./validation', graph=tf.get_default_graph())

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    i = 0
    j = 0
    for epoch in range(80):
        sess.run(train_iterator.initializer)
        while True:
            try:

                _, o_stats = sess.run([optimizer, stats])
                fwtrain.add_summary(o_stats, i)
                i += 1

            except tf.errors.OutOfRangeError:
                break

        print("y", epoch)
        sess.run(val_iterator.initializer)
        while True:
            try:
                validation_stats = sess.run(stats_v)
                fwvalidation.add_summary(validation_stats, j)
                j += 1

            except tf.errors.OutOfRangeError:
                break
    sess.run(test_iterator.initializer)
    while True:
        try:
            test_stats = sess.run(stats_t)
            fwtest.add_summary(test_stats, i)
            i += 1

        except tf.errors.OutOfRangeError:
            break
