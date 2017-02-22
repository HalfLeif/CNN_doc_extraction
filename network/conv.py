
import tensorflow as tf
from tensorflow.contrib import slim

def addLayer(net, depth, name):
    net = slim.conv2d(net, depth, [3, 3], padding='VALID', scope='conva_' + name)
    net = slim.conv2d(net, depth, [3, 3], padding='VALID', scope='convb_' + name)
    net = slim.max_pool2d(net, [2, 2], scope='pool_' + name)
    return net

def deepEncoder(image):
    with slim.arg_scope([slim.conv2d],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.truncated_normal_initializer(0.0, 0.05),
                        weights_regularizer=slim.l2_regularizer(0.0005)):
        net = image
        net = addLayer(net, 8, '1')
        net = addLayer(net, 8, '2')
        net = addLayer(net, 16, '3')
        net = addLayer(net, 16, '4')
        net = addLayer(net, 32, '5')
        net = addLayer(net, 32, '6')
        net = addLayer(net, 64, '7')
        return net


def encodeImage(image):
    with slim.arg_scope([slim.conv2d],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.truncated_normal_initializer(0.0, 0.1),
                        weights_regularizer=slim.l2_regularizer(0.0005)):
        net = image
        net = addLayer(net, 32, '1')
        net = addLayer(net, 64, '2')
        net = addLayer(net, 128, '3')
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

def switchFirstTwo(ds):
    '''Swaps the first two elements in a list.'''
    if len(ds) < 2:
        return ds
    ds[0], ds[1] = (ds[1], ds[0])
    return ds

def expandDigits(digit_stack):
    ''' Takes a stack of digit vectors probabilities, shape=[N, 10],
        returns a single vector of probabilities, shape=[10**N].
    '''
    digit_list = switchFirstTwo(tf.unstack(digit_stack))
    meshes = tf.meshgrid(*digit_list)
    product = tf.foldr(tf.multiply, meshes)
    number = tf.reshape(product, [-1])
    return number


def decodeNumber(net, keep_prob):
    ''' Decodes activation map into a 3 digit number.
        Uses a separate output for whether the input contains any digits or not.
    '''
    with slim.arg_scope([slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.truncated_normal_initializer(0.0, 0.1),
                        weights_regularizer=slim.l2_regularizer(0.0005)):
        net = slim.flatten(net)
        net = slim.fully_connected(net, 1024, scope='fc1')
        net = slim.dropout(net, keep_prob, scope='dropout1')

        ignore = slim.fully_connected(net, 2, activation_fn=None, scope='ignore')
        digit1 = slim.fully_connected(net, 10, activation_fn=None, scope='digit1')
        digit2 = slim.fully_connected(net, 10, activation_fn=None, scope='digit2')
        digit3 = slim.fully_connected(net, 10, activation_fn=None, scope='digit3')

        # Note: this adds redundancy but is necessary
        # in order to require the exact year
        # instead of giving credit for partially correct year transcriptions.
        stacked_digits = tf.stack([digit1, digit2, digit3], axis=1)
        numbers = tf.map_fn(expandDigits, stacked_digits)

        return ignore, numbers


def lenet5(image, keep_prob):
    ''' Lenet-5 with dropout

        Reaches 0.97 test acc on MNIST in 1000 minibatches of 50.
    '''
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


def refLenet5(image, keep_prob):
    ''' Refactored Lenet-5 with dropout

        Reaches 0.975 test acc on MNIST in 1000 minibatches of 50.
    '''
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
