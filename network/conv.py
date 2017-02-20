
import tensorflow as tf
from tensorflow.contrib import slim

''' Lenet-5 with dropout

    Reaches 0.97 test acc on MNIST in 1000 minibatches of 50.
'''
def lenet5(image, keep_prob):
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      weights_initializer=tf.truncated_normal_initializer(0.0, 0.1),
                      weights_regularizer=slim.l2_regularizer(0.0005)):

    net = image
    net = slim.conv2d(net, 32, [5, 5], scope='conv1')
    net = slim.max_pool2d(net, [2, 2], scope='pool1')

    net = slim.conv2d(net, 64, [5, 5], scope='conv2')
    net = slim.max_pool2d(net, [2, 2], scope='pool2')

    net = slim.flatten(net)
    net = slim.fully_connected(net, 1024, scope='fc1')
    net = slim.dropout(net, keep_prob, scope='dropout1')

    net = slim.fully_connected(net, 10, activation_fn=None, scope='output')
    return net


''' Refactored Lenet-5 with dropout

    Reaches 0.975 test acc on MNIST in 1000 minibatches of 50.
'''
def refLenet5(image, keep_prob):
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      weights_initializer=tf.truncated_normal_initializer(0.0, 0.1),
                      weights_regularizer=slim.l2_regularizer(0.0005)):

    net = image
    net = slim.conv2d(net, 32, [3, 3], scope='conv1a')
    net = slim.conv2d(net, 32, [3, 3], scope='conv1b')
    net = slim.max_pool2d(net, [2, 2], scope='pool1')

    net = slim.conv2d(net, 64, [3, 3], scope='conv2a')
    net = slim.conv2d(net, 64, [3, 3], scope='conv2b')
    net = slim.max_pool2d(net, [2, 2], scope='pool2')

    net = slim.flatten(net)
    net = slim.fully_connected(net, 1024, scope='fc1')
    net = slim.dropout(net, keep_prob, scope='dropout1')

    net = slim.fully_connected(net, 10, activation_fn=None, scope='output')
    return net

if __name__ == '__main__':
    convLayers(tf.zeros([20, 100, 1]))
