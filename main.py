
import img_debug as debug
# import loading.load_iris as iris
import loading.load_swe as swe
import network.attend as att
import network.decoder as dec
import network.encoder as enc
import pretrain.mnist as mnist
import scoring as sc

import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline

import os
import time

# TODO: replace with FLAGS
model_dir = 'models'

MNIST_BATCH_SIZE = 50
IRIS_BATCH_SIZE = 1
SWE_BATCH_SIZE = 7
NUM_THREADS = 3

pretrain_mnist = True


def numParams():
    def varParams(var):
        prod = 1
        for dim in var.get_shape():
            prod = prod * dim.value
        return prod

    num_params = 0
    for var in tf.trainable_variables():
        num_params = num_params + varParams(var)
    return num_params


def printNumParams():
    print('# PARAMETERS: ', numParams())


def runNetwork(batch_images, train_mode):
    if train_mode:
        keep_prob = 0.5
    else:
        keep_prob = 1.0

    # Network
    print('Input: ', batch_images.get_shape())
    activation = enc.deepEncoder(batch_images)
    print('Batch embedding:', activation.get_shape())

    attention = att.attend(activation, keep_prob)
    print('Attention: ', attention.get_shape())

    # attention = tf.Print(attention, ['WRITE ATTENTION',
    #                                  debug.debugFirstImage(batch_images, 'input'),
    #                                  debug.debugAttention(attention)])

    attended = tf.reduce_sum(activation * attention, [1, 2])

    # DEBUG by writing image to file
    # TODO: debug attention model
    # first_image = tf.slice(batch_images, [0, 0, 0, 0], [1, -1, -1, -1])
    # first_attention = tf.slice(attention, [0, 0, 0, 0], [1, -1, -1, -1])
    # attended = tf.Print(attended, [debugImage(first_image)])

    print('Attend: ', attended.get_shape())
    year_log = dec.decodeNumber(attended, keep_prob)
    return year_log


def trainOp(pretrain=True):
    if pretrain:
        batch_images, batch_years = mnist.mnistBatch(MNIST_BATCH_SIZE, True)
        num_batches = int(55000/MNIST_BATCH_SIZE)
    else:
        # batch_images, batch_years = iris.irisQueue(iris_train, IRIS_BATCH_SIZE)
        batch_images, batch_years, _, num_batches = swe.sweBatch(SWE_BATCH_SIZE, True)
        print('Each epoch runs for', num_batches, 'batches, each with', SWE_BATCH_SIZE, 'images.')

    year_log = runNetwork(batch_images, True)
    error = sc.error(batch_years, year_log)
    train_step = tf.train.AdamOptimizer(1e-4).minimize(error)

    return train_step, num_batches

def py_printCompare(min_expected, max_expected, output, accuracy, certainties):
    iterations = min(len(min_expected), 25)
    for i in range(iterations):
        if min_expected[i] <= output[i] and max_expected[i] >= output[i]:
            prefix = '   '
        else:
            prefix = ' X '
        # print(prefix, expected[i], '->', output[i], '  ', certainties[i])
        upper = max_expected[i]
        if upper == min_expected[i]:
            upper = '    '
        print(prefix, min_expected[i], upper, '->', output[i])

    print('In total ' + str(len(min_expected)) + ' pairs evaluated...')
    print('Accuracy:', accuracy)
    # print('Mean certainty:', sum(certainties)/len(certainties))
    return np.int32(len(min_expected))

def testOp(pretrain=True, eval_size=SWE_BATCH_SIZE):
    tf.get_variable_scope().reuse_variables()
    if pretrain:
        batch_images, batch_years = mnist.mnistBatch(eval_size, False)
    else:
        # batch_images, batch_years = iris.irisQueue(iris_test, 3)
        batch_images, batch_years, _, _ = swe.sweBatch(eval_size, False)
    year_log = runNetwork(batch_images, False)
    year_prob = tf.nn.softmax(year_log)
    pred = sc.predict(year_log)

    min_year = tf.squeeze(tf.slice(batch_years, [0,0], [-1,1]))
    max_year = tf.squeeze(tf.slice(batch_years, [0,1], [-1,1]))
    # mid_year = tf.floordiv(min_year + max_year, 2)

    correct_lower = tf.less_equal(min_year, pred)
    correct_upper = tf.greater_equal(max_year, pred)

    correct_prediction = tf.logical_and(correct_lower, correct_upper)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # certainties = sc.certainty(year_prob, max_year)
    # print('DEBUG_CERT', certainties.get_shape())
    debug_pred = tf.py_func(py_printCompare, [min_year, max_year, pred, accuracy, 1], tf.int32, stateful=True)
    accuracy = tf.Print(accuracy, ['Compare', debug_pred], summarize=MNIST_BATCH_SIZE)

    return accuracy


def saveModel(saver, model_name, step):
    print('Saving ', model_name)
    save_name = os.path.join(model_dir, model_name)
    save_path = saver.save(sess, save_name, global_step=step, write_meta_graph=False)

def train(model_name):
    train_step, num_batches = trainOp(pretrain_mnist)
    saver = tf.train.Saver(max_to_keep=3)

    for i in range(num_batches):
        if (i%100 == 0):
            print('BATCH', i)

        train_step.run()

        if (i%100 == 99):
            saveModel(saver, model_name, i)
        if (i%100 == 99):
            accuracy.eval()
    saveModel(saver, model_name, i)



def loadModel(sess, model_name=None):
    ''' Restores network variables of the latest session
        OR loads the specified model, e.g. 'soft_attention_2-1050'.
    '''
    saver = tf.train.Saver()
    if model_name:
        version = os.path.join(model_dir, model_name)
    else:
        version = tf.train.latest_checkpoint(model_dir)

    print('Restoring model from:', version)
    saver.restore(sess, version)

def runTimeEstimate(sess):
    ''' Creates timeline-file for debugging expensive operations.
        Open file with chrome://tracing.
    '''
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    train_step, _ = trainOp(pretrain_mnist)

    sess.run(train_step, options=run_options, run_metadata=run_metadata)

    tl = timeline.Timeline(run_metadata.step_stats)
    ctf = tl.generate_chrome_trace_format()
    with open('data\\timeline.json', 'w') as f:
        f.write(ctf)
    print('WROTE TIMELINE')

def test(total_test=500, eval_size=SWE_BATCH_SIZE):
    accuracy = testOp(pretrain_mnist, eval_size=eval_size)

    accs = 0
    iterations = total_test/eval_size
    for ai in range(iterations):
        if ai%10 == 0 and ai > 0:
            print('Tested ', ai*eval_size, 'acc: ', accs/ai)
        acc = accuracy.eval()
        accs = accs + acc

    ai = ai + 1
    print('Tested ', ai*eval_size, 'acc: ', accs/ai)

def classifyOp(collection, batch_size=SWE_BATCH_SIZE):
    batch_images, _, batch_paths, num_batches = swe.classificationBatch(batch_size, collection)
    year_log = runNetwork(batch_images, False)
    return year_log, batch_paths, num_batches

def classify(sess, collection):
    year_log, batch_paths, num_batches = classifyOp(collection)
    print('YEAR_LOG', year_log[0].get_shape())

    classifications = []
    for i in range(num_batches):
        readout_logits, readout_paths = sess.run([year_log, batch_paths])
        for j in range(len(readout_paths)):
            digit1 = readout_logits[0][j]
            digit2 = readout_logits[1][j]
            digit3 = readout_logits[3][j]
            classification = (readout_paths[j], digit1, digit2, digit3)
            classifications.append(classification)

    return classifications

with tf.Session(config=tf.ConfigProto(
        intra_op_parallelism_threads=NUM_THREADS)) as sess:
    printNumParams()
    print('INIT VARIABLES!')
    sess.run(tf.global_variables_initializer())

    print('Start threads...')
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # loadModel(sess, model_name=None)
    # loadModel(sess, model_name='Swe_DEP_7-199')
    # loadModel(sess, model_name='DEP_pad_random_3-1099')

    print('System ready!')
    time_start = time.process_time()

    # epoch_start = 1
    # for i in range(10):
    #     epoch = epoch_start + i
    #     model_name = 'Single_digit_PRE_' + str(epoch)
    #     train(model_name)

    classifications = classify(sess, '1647578')
    print(classifications)
    time_end = time.process_time()

    # test()

    # writeReEncoded()
    # runTimeEstimate(sess)

    print('CPU time: ', time_end - time_start)

    coord.request_stop()
    coord.join(threads)
