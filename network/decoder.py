
import tensorflow as tf
from tensorflow.contrib import slim


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
        print('FLATTEN: ', net.get_shape())
        net = slim.fully_connected(net, 1024, scope='fc1')
        net = slim.dropout(net, keep_prob, scope='dropout1')

        # net = slim.fully_connected(net, 512, scope='fc2')
        # net = slim.dropout(net, keep_prob, scope='dropout2')

        # ignore = slim.fully_connected(net, 2, activation_fn=None, scope='ignore')
        digit1 = slim.fully_connected(net, 10, activation_fn=None, scope='digit1')
        digit2 = slim.fully_connected(net, 10, activation_fn=None, scope='digit2')
        digit3 = slim.fully_connected(net, 10, activation_fn=None, scope='digit3')

        # Note: this adds redundancy but is necessary
        # in order to require the exact year
        # instead of giving credit for partially correct year transcriptions.
        return [digit1, digit2, digit3]
        # numbers = tf.map_fn(expandDigits, stacked_digits)
        # return ignore, numbers
        # return numbers
