
import tensorflow as tf
from tensorflow.contrib import slim

def addLayer(net, depth, name):
    net = slim.conv2d(net, depth, [3, 3], padding='SAME', scope='conva_' + name)
    net = slim.conv2d(net, depth, [3, 3], padding='SAME', scope='convb_' + name)
    net = slim.max_pool2d(net, 2, scope='pool_' + name)
    return net

def thinEncoder(image):
    with slim.arg_scope([slim.conv2d],
                        activation_fn=tf.nn.relu,
                        padding='SAME',
                        weights_initializer=tf.truncated_normal_initializer(0.0, 0.1),
                        weights_regularizer=slim.l2_regularizer(0.0005)):
        net = image
        net = slim.conv2d(net, 32, [3, 3], scope='conv_1')
        net = slim.max_pool2d(net, [2, 2], scope='pool_1')

        net = slim.conv2d(net, 64, [3, 3], scope='conv_2')
        net = slim.max_pool2d(net, [2, 2], scope='pool_2')

        net = slim.conv2d(net, 128, [3, 3], scope='conv_3')
        net = slim.max_pool2d(net, [1, 2], scope='pool_3')

        net = slim.conv2d(net, 128, [3, 3], scope='conv_4')
        net = slim.max_pool2d(net, [2, 2], scope='pool_4')

        net = slim.conv2d(net, 256, [1, 5], padding='VALID', scope='conv_wide')
        net = slim.max_pool2d(net, [1, 2], scope='pool_wide')
        return net

def balancedWidthEncoder(image):
    with slim.arg_scope([slim.conv2d],
                        activation_fn=tf.nn.relu,
                        padding='SAME',
                        weights_initializer=tf.truncated_normal_initializer(0.0, 0.1),
                        weights_regularizer=slim.l2_regularizer(0.0005)):
        net = image
        net = addLayer(net, 8, '1')
        net = addLayer(net, 32, '2')

        # net = addLayer(net, 128, '4')
        net = slim.conv2d(net, 64, [3, 3], scope='conv_3')
        net = slim.max_pool2d(net, [1, 2], scope='pool_3')

        net = slim.conv2d(net, 128, [3, 3], scope='conv_4')
        net = slim.max_pool2d(net, [2, 2], scope='pool_4')


        net = slim.conv2d(net, 256, [1, 5], padding='VALID', scope='conv_wide')
        net = slim.max_pool2d(net, [1, 2], scope='pool_wide')
        return net

def minWidthEncoder(image):
    with slim.arg_scope([slim.conv2d],
                        activation_fn=tf.nn.relu,
                        padding='SAME',
                        weights_initializer=tf.truncated_normal_initializer(0.0, 0.1),
                        weights_regularizer=slim.l2_regularizer(0.0005)):
        net = image
        net = addLayer(net, 4, '1')
        net = addLayer(net, 16, '2')

        # net = addLayer(net, 128, '4')
        net = slim.conv2d(net, 32, [3, 3], scope='conv_3')
        net = slim.max_pool2d(net, 2, scope='pool_3')

        net = slim.conv2d(net, 128, [3, 3], scope='conv_4')
        net = slim.max_pool2d(net, 2, scope='pool_4')


        net = slim.conv2d(net, 256, [1, 5], padding='VALID', scope='conv_wide')
        net = slim.max_pool2d(net, 2, scope='pool_wide')
        return net

def deepEncoder(image):
    with slim.arg_scope([slim.conv2d],
                        activation_fn=tf.nn.relu,
                        padding='SAME',
                        weights_initializer=tf.truncated_normal_initializer(0.0, 0.1),
                        weights_regularizer=slim.l2_regularizer(0.0005)):
        net = image
        print('IMG', net.get_shape())
        net = addLayer(net, 32, '1')
        print('P1 ', net.get_shape())
        net = addLayer(net, 64, '2')
        print('P2 ', net.get_shape())

        # net = addLayer(net, 128, '4')
        net = slim.conv2d(net, 128, [3, 3], scope='conv_3')
        net = slim.max_pool2d(net, [2, 2], scope='pool_3')
        print('P3 ', net.get_shape())

        net = slim.conv2d(net, 128, [3, 3], scope='conv_4')
        net = slim.max_pool2d(net, [2, 2], scope='pool_4')
        print('P4 ', net.get_shape())


        net = slim.conv2d(net, 256, [1, 5], padding='VALID', scope='conv_wide')
        print('Cw ', net.get_shape())
        net = slim.max_pool2d(net, [2, 2], scope='pool_wide')
        print('Pw ', net.get_shape())
        return net
