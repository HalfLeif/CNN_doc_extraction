
import tensorflow as tf
from tensorflow.contrib import slim

import conv


def encodeImage(image):
    with slim.arg_scope([slim.conv2d],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.truncated_normal_initializer(0.0, 0.1),
                        weights_regularizer=slim.l2_regularizer(0.0005)):
        net = image
        net = conv.addLayer(net, 32, '1')
        net = conv.addLayer(net, 64, '2')
        net = conv.addLayer(net, 128, '3')
        return net

def decodeDigit(net, keep_prob):
    with slim.arg_scope([slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.truncated_normal_initializer(0.0, 0.1),
                        weights_regularizer=slim.l2_regularizer(0.0005)):
        net = slim.flatten(net)
        net = slim.fully_connected(net, 1024, scope='fc1')
        net = slim.dropout(net, keep_prob, scope='dropout1')

        net = slim.fully_connected(net, 10, activation_fn=None, scope='output')
        return net
