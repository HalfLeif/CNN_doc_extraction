
import network.conv as conv
import preprocess.load_iris as iris
import preprocess.load_swe as swe
import preprocess.grayscale as gray
import pretrain.mnist_debug as debug
import pretrain.mnist as mnist
import scoring as sc

import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline

import os
import time

# TODO: replace with FLAGS
model_name = 'temp'
model_dir = 'models'

# TODO train over several OR split up list from same collection...
swe_train = '1647578'
swe_test = '1949331'

# Csv-file generated by load_iris:
# iris_train = 'data\\fr_train.csv'
# iris_test = 'data\\fr_test.csv'

# FIX_SHAPE = [900, 1500]
MNIST_BATCH_SIZE = 50
IRIS_BATCH_SIZE = 1
SWE_BATCH_SIZE = 1
NUM_THREADS = 3


def loadImage(jpg_path, ratio):
    image_content = tf.read_file(jpg_path)
    x = tf.image.decode_jpeg(image_content, channels=1, ratio=ratio, name='image')
    # x = tf.image.resize_images(x, FIX_SHAPE)
    x = tf.cast(x, tf.float32)
    x = gray.simpleRemap(x)
    # x = tf.cast(x, tf.uint8)
    # x, _ = gray.otsusGlobalThreshold(x)
    return x

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
    # activation = conv.deepEncoder(batch_images)
    activation = conv.minWidthEncoder(batch_images)
    # activation = conv.balancedWidthEncoder(batch_images)
    print('Batch embedding:', activation.get_shape())

    attention = conv.attend(activation, keep_prob)
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
    year_prob = conv.decodeNumber(attended, keep_prob)
    return year_prob

def irisQueue(iris_dir, batch_size):
    print('Load transcriptions')
    all_jpgs, all_years = iris.loadTranscriptions(iris_dir)
    jpgs = tf.constant(all_jpgs, tf.string)
    years = tf.constant(all_years, tf.int32)

    jpg_path, year = tf.train.slice_input_producer([jpgs, years], shuffle=True, capacity=25)
    # jpg_path = tf.Print(jpg_path, ['Load iris image: ', jpg_path], summarize=100)
    image = loadImage(jpg_path, ratio=2)

    batch_images, batch_years = tf.train.batch([image, year], batch_size=batch_size, capacity=2, num_threads=2, dynamic_pad=True)

    # Need to invert images after the dynamic padding.
    batch_images = 1 - batch_images

    print('Iris queue created')
    return batch_images, batch_years

def sweQueue(collection_name, batch_size):
    print('Load transcriptions')
    all_jpgs, all_years = swe.loadCollection(collection_name)
    jpgs = tf.constant(all_jpgs, tf.string)
    years = tf.constant(all_years, tf.int32)

    jpg_path, year = tf.train.slice_input_producer([jpgs, years], shuffle=True, capacity=25)
    jpg_path = tf.Print(jpg_path, ['Load swe image: ', jpg_path], summarize=100)
    # TODO: ratio 4 or 8?
    image = loadImage(jpg_path, ratio=8)

    batch_images, batch_years = tf.train.batch([image, year], batch_size=batch_size, capacity=4, num_threads=2, dynamic_pad=True)

    # Need to invert images after the dynamic padding.
    batch_images = 1 - batch_images

    batch_images = tf.Print(batch_images, ['DEBUG', debug.debugFirstImage(batch_images, 'SWE')])

    print('Swe queue created')
    return batch_images, batch_years


def trainOp(pretrain=True):
    if pretrain:
        batch_images, batch_years = mnist.mnistBatch(MNIST_BATCH_SIZE, True)
    else:
        # batch_images, batch_years = irisQueue(iris_train, IRIS_BATCH_SIZE)
        batch_images, batch_years = sweQueue(swe_train, SWE_BATCH_SIZE)

    remapped = tf.mod(batch_years, 1000) + 1000
    year_prob = runNetwork(batch_images, True)

    error = sc.error(remapped, year_prob)
    train_step = tf.train.AdamOptimizer(1e-4).minimize(error)
    return train_step

def py_printCompare(expected, output, accuracy, certainties):
    iterations = min(len(expected), 25)
    for i in range(iterations):
        if expected[i] == output[i]:
            prefix = '   '
        else:
            prefix = ' X '
        print(prefix, expected[i], '->', output[i], '  ', certainties[i])

    print('In total ' + str(len(expected)) + ' pairs evaluated...')
    print('Accuracy:', accuracy)
    print('Mean certainty:', sum(certainties)/len(certainties))
    return len(expected)

eval_batch_size = tf.placeholder_with_default(50, [], name='eval_batch_size')

def evalOp(pretrain=True):
    tf.get_variable_scope().reuse_variables()
    if pretrain:
        batch_images, batch_years = mnist.mnistBatch(eval_batch_size, False)
    else:
        # batch_images, batch_years = irisQueue(iris_test, 3)
        batch_images, batch_years = sweQueue(swe_test, SWE_BATCH_SIZE)
    year_prob = runNetwork(batch_images, False)
    year_prob = tf.nn.softmax(year_prob)
    pred = sc.predict(year_prob)
    certainties = sc.certainty(year_prob, batch_years)

    remapped = tf.mod(batch_years, 1000) + 1000
    correct_prediction = tf.equal(remapped, pred)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    debug_pred = tf.py_func(py_printCompare, [batch_years, pred, accuracy, certainties], tf.int32, stateful=True)
    accuracy = tf.Print(accuracy, ['Compare', debug_pred], summarize=MNIST_BATCH_SIZE)

    return accuracy

pretrain_mnist = False
train_step = trainOp(pretrain_mnist)
accuracy = evalOp(pretrain_mnist)

def saveModel(saver):
    print('Saving ', model_name)
    save_name = os.path.join(model_dir, model_name)
    save_path = saver.save(sess, save_name, global_step=i, write_meta_graph=False)

def train():
    saver = tf.train.Saver(max_to_keep=3)

    if pretrain_mnist:
        num_batches = int(55000/MNIST_BATCH_SIZE)
    else:
        # num_batches = int(10500/IRIS_BATCH_SIZE)
        num_batches = int(14330/SWE_BATCH_SIZE)

    for i in range(num_batches):
        if (i%100 == 0):
            print('BATCH', i)

        train_step.run()

        if (i%100 == 99):
            saveModel(saver)
        if (i%100 == 99):
            accuracy.eval()
    saveModel(saver)



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

    sess.run(train_step, options=run_options, run_metadata=run_metadata)

    tl = timeline.Timeline(run_metadata.step_stats)
    ctf = tl.generate_chrome_trace_format()
    with open('data\\timeline.json', 'w') as f:
        f.write(ctf)
    print('WROTE TIMELINE')

def evaluate():
    batch_size=10

    accs = 0
    for ai in range(100):
        if ai%10 == 0 and ai > 0:
            print('Tested ', ai*batch_size, 'acc: ', accs/ai)
        acc = accuracy.eval(feed_dict={eval_batch_size: batch_size})
        accs = accs + acc

    ai = ai + 1
    print('Tested ', ai*batch_size, 'acc: ', accs/ai)

with tf.Session(config=tf.ConfigProto(
        intra_op_parallelism_threads=NUM_THREADS)) as sess:
    printNumParams()
    print('INIT VARIABLES!')
    # sess.run(tf.global_variables_initializer())

    print('Start threads...')
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # loadModel(sess, model_name=None)
    # loadModel(sess, model_name='DEP_pad_random_4-1099')
    loadModel(sess, model_name='DEM_pad_random_25-1099')
    # loadModel(sess, model_name='DEB_pad_random_12-1099')

    print('System ready!')
    time_start = time.process_time()

    # evalsize = 100
    # model_name = 'swe_debug'
    train()
    # evaluate()

    # accuracy.eval(feed_dict={eval_batch_size: 1})

    # writeReEncoded()
    # runTimeEstimate(sess)

    time_end = time.process_time()
    print('CPU time: ', time_end - time_start)

    coord.request_stop()
    coord.join(threads)
