
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from network import conv
from preprocess import grayscale

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

import os
import sys
import numpy


model_path = 'models'
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
sess = tf.InteractiveSession()


# Input
y_ = tf.placeholder(tf.float32, shape=[None, 10], name='y_')
x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
x_image = tf.reshape(x, [-1,28,28,1])

# Readout
keep_prob = tf.placeholder(tf.float32)
# y_conv = conv.refLenet5(x_image, keep_prob)
activation = conv.encodeImage(x_image)
y_conv = conv.decodeDigit(activation, keep_prob)

# Training
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.global_variables_initializer())

def train():
    saver = tf.train.Saver(max_to_keep=5)

    for i in range(1000):
        # original range: 20000
        batch = mnist.train.next_batch(50)
        if i%100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                    x:batch[0], y_: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g"%(i, train_accuracy))
            savepath = saver.save(sess, os.path.join(model_path, 'data'), global_step=i)
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    print("test accuracy %g"%accuracy.eval(feed_dict={
            x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

# Prints 28x28 image
def printImage(image):
    for i in range(28):
        for j in range(28):
            pixel = image[28*i + j]
            out = '_'
            if pixel > 0.7:
                out = '#'
            elif pixel > 0.3:
                out = '/'
            sys.stdout.write(out)
        sys.stdout.write('\n')


def load():
    saver = tf.train.Saver()
    latest = tf.train.latest_checkpoint(model_path)
    saver.restore(sess, latest)


def classify(image_nr=0):
    image = mnist.test.images[image_nr]
    printImage(image)
    readout = tf.nn.softmax(y_conv)
    predictions = sess.run(readout, feed_dict={
        x: [image], keep_prob: 1.0
    })

    print(predictions[0])
    print("Predicted: %d"%numpy.argmax(predictions))
    print("Expected: %d"%numpy.argmax(mnist.test.labels[image_nr]))


if __name__ == '__main__':
    train()
    # load()
    # for i in range(10):
    #     classify(i)
