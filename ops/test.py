import loading.load_swe as swe
import network.run as run
import network.scoring as sc
import pretrain.mnist as mnist

import numpy as np
import tensorflow as tf

import math


def py_printCompare(min_expected, max_expected, output, accuracy, certainties):
    iterations = min(len(min_expected), 25)
    for i in range(iterations):
        if min_expected[i] <= output[i] and max_expected[i] >= output[i]:
            prefix = '   '
        else:
            prefix = ' X '
        upper = max_expected[i]
        if upper == min_expected[i]:
            upper = '    '
        print(prefix, min_expected[i], upper, '->', output[i])

    print('In total ' + str(len(min_expected)) + ' pairs evaluated...')
    print('Accuracy:', accuracy)
    return np.int32(len(min_expected))


def testOp(pretrain=True, debug=True):
    tf.get_variable_scope().reuse_variables()
    if pretrain:
        (batch_images, batch_years), num_batches = mnist.mnistBatch(False)
    else:
        batch_images, batch_years, _, num_batches = swe.sweBatch('test')

    year_log = run.runNetwork(batch_images, False)
    year_prob = tf.nn.softmax(year_log)
    pred = sc.predict(year_log)

    min_year = tf.squeeze(tf.slice(batch_years, [0,0], [-1,1]))
    max_year = tf.squeeze(tf.slice(batch_years, [0,1], [-1,1]))

    correct_lower = tf.less_equal(min_year, pred)
    correct_upper = tf.greater_equal(max_year, pred)

    correct_prediction = tf.logical_and(correct_lower, correct_upper)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    if debug:
        debug_pred = tf.py_func(py_printCompare, [min_year, max_year, pred, accuracy, 1], tf.int32, stateful=True)
        accuracy = tf.Print(accuracy, ['Compare', debug_pred], summarize=50)

    return accuracy, num_batches


def test(accuracy_op, num_batches):
    print('Will test for', num_batches, 'batches...')
    accs = 0
    for ai in range(num_batches):
        acc = accuracy_op.eval()
        accs = accs + acc
        if ai%10 == 0 and ai > 0:
            print('After batch ', ai, 'mean accuracy: ', accs/(ai + 1))

    print('After', num_batches, 'batches, accuracy: ', accs/num_batches)


def lazyTest(pretrain_mnist, num_batches=None, debug=True):
    ''' Lazy test operation.
        If specifies number of batches, will test that many.
        If None, then runs through the entire test set.
    '''
    accuracy_op, num_all_batches = testOp(pretrain_mnist, debug=debug)
    if not num_batches:
        num_batches = num_all_batches
    return lambda _: test(accuracy_op, num_batches=num_batches)
