
import tensorflow as tf
from tensorflow.contrib import slim

def addLayer(net, depth, name):
    net = slim.conv2d(net, depth, [3, 3], padding='SAME', scope='conva_' + name)
    net = slim.conv2d(net, depth, [3, 3], padding='SAME', scope='convb_' + name)
    net = slim.max_pool2d(net, [2, 2], scope='pool_' + name)
    return net

def deepEncoder(image):
    with slim.arg_scope([slim.conv2d],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                        weights_regularizer=slim.l2_regularizer(0.0005)):
        net = image
        net = slim.conv2d(net, 5, [3, 3], padding='VALID', scope='conv_0')
        net = slim.max_pool2d(net, [2, 2], scope='pool_0')

        net = slim.conv2d(net, 5, [3, 3], padding='VALID', scope='conv_1a')
        net = slim.conv2d(net, 10, [3, 3], padding='VALID', scope='conv_1b')
        net = slim.max_pool2d(net, [2, 2], scope='pool_1')

        net = slim.conv2d(net, 10, [3, 3], padding='VALID', scope='conv_2a')
        net = slim.conv2d(net, 10, [3, 3], padding='VALID', scope='conv_2b')
        # Receptive field of 24x24
        net = slim.max_pool2d(net, [2, 2], scope='pool_2')

        net = slim.conv2d(net, 10, [3, 3], padding='VALID', scope='conv_3a')
        net = slim.conv2d(net, 10, [3, 3], padding='VALID', scope='conv_3b')
        net = slim.max_pool2d(net, [4, 4], scope='pool_3a')
        net = slim.max_pool2d(net, [4, 4], scope='pool_3b')

        net = slim.conv2d(net, 20, [3, 3], padding='VALID', scope='conv_4')
        net = slim.max_pool2d(net, [4, 4], scope='pool_4')

        # net = addLayer(net, 16, '0')
        # net = addLayer(net, 32, '2')
        # net = addLayer(net, 64, '4')
        # # Now each unit has a receptive field of 36x36, enough to cover digits
        #
        # net = addLayer(net, 64, '7')
        # net = slim.max_pool2d(net, [4, 4], scope='pool_7x')
        # net = addLayer(net, 64, '8')
        # net = slim.max_pool2d(net, [4, 4], scope='pool_8x')
        # net = addLayer(net, 64, '9')
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
    ''' Takes a stack of N vectors of digit log probabilities,
        returns a single vector of number log probabilities with shape=[10**N].
    '''
    digit_list = switchFirstTwo(tf.unstack(digit_stack))
    meshes = tf.meshgrid(*digit_list)

    # Note: must add here instead of multiply
    # because probabilities are on log form.
    sum_log = tf.foldr(tf.add, meshes)
    number_log = tf.reshape(sum_log, [-1])
    return number_log


def decodeNumber(net, keep_prob):
    ''' Decodes activation map into a 3 digit number.
        Uses a separate output for whether the input contains any digits or not.
    '''
    with slim.arg_scope([slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                        weights_regularizer=slim.l2_regularizer(0.0005)):
        net = slim.flatten(net)
        net = slim.fully_connected(net, 1024, scope='fc1')
        net = slim.dropout(net, keep_prob, scope='dropout1')

        net = slim.fully_connected(net, 512, scope='fc2')
        net = slim.dropout(net, keep_prob, scope='dropout2')

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

        # return ignore, [digit1, digit2, digit3]


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
