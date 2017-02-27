
import network.conv as conv
import preprocess.load_iris as iris
import preprocess.grayscale as gray
import pretrain.mnist as mnist
import scoring as sc

import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline

import os

# TODO: replace with FLAGS
model_name = 'temp'
model_path = os.path.join('models', model_name)

# Csv-file generated by load_iris:
iris_train = 'data\\fr_train.csv'
iris_test = 'data\\fr_test.csv'

# FIX_SHAPE = [900, 1500]
BATCH_SIZE = 50
NUM_THREADS = 4


def loadImage(jpg_path):
    image_content = tf.read_file(jpg_path)
    x = tf.image.decode_jpeg(image_content, channels=1, ratio=2, name='image')
    x = tf.image.resize_images(x, FIX_SHAPE)
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

# Preload labels
# all_jpgs, all_years = iris.loadTranscriptions(iris_train)
# jpgs = tf.constant(all_jpgs, tf.string)
# years = tf.constant(all_years, tf.int32)

# Input pipeline
# jpg_path, year = tf.train.slice_input_producer([jpgs, years], shuffle=True)
# image = loadImage(jpg_path)
# batch_images, batch_years = tf.train.batch([image, year], batch_size=BATCH_SIZE)

# DEBUG
# restore = tf.cast(image*255, tf.uint8)
# re_encoded = tf.image.encode_jpeg(restore)

def runNetwork(batch_images, train_mode):
    if train_mode:
        keep_prob = 0.5
    else:
        keep_prob = 1.0

    # Network
    activation = conv.deepEncoder(batch_images)
    print('Batch embedding:', activation.get_shape())

    attended = conv.attend(activation, keep_prob)
    print('Attend: ', attended.get_shape())
    year_prob = conv.decodeNumber(attended, keep_prob)
    return year_prob

def trainOp():
    batch_images, batch_years = mnist.mnistBatch(BATCH_SIZE, True)
    remapped = tf.mod(batch_years, 1000) + 1000
    year_prob = runNetwork(batch_images, True)

    error = sc.error(remapped, year_prob)
    train_step = tf.train.AdamOptimizer(1e-4).minimize(error)
    return train_step

def py_printCompare(expected, output):
    iterations = min(len(expected), 100)
    for i in range(iterations):
        print(' ', expected[i], '->', output[i])

    print('In total ' + str(len(expected)) + ' pairs evaluated...')
    return 0

def evalOp():
    tf.get_variable_scope().reuse_variables()

    batch_images, batch_years = mnist.mnistBatch(500, False)
    remapped = tf.mod(batch_years, 1000) + 1000
    year_prob = runNetwork(batch_images, False)
    pred = sc.predict(year_prob)

    correct_prediction = tf.equal(remapped, pred)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    debug_pred = tf.py_func(py_printCompare, [batch_years, pred], tf.int32, stateful=True)
    accuracy = tf.Print(accuracy, ['Compare', debug_pred], summarize=BATCH_SIZE)
    return accuracy

train_step = trainOp()
accuracy = evalOp()

def writeReEncoded():
    b = re_encoded.eval()
    with open('data\\temp.jpg', 'wb+') as f:
        f.write(b)
    print('WROTE RE-ENCODED IMAGE')

def train():
    # num_batches = int(len(all_jpgs)/BATCH_SIZE)
    num_batches = int(55000/BATCH_SIZE)
    for i in range(num_batches):
        print('BATCH', i)

        if (i%100 == 0):
            acc = accuracy.eval()
            print('Accuracy: ', acc)

        train_step.run()

        if (i%50 == 49 and i > 0):
            savepath = saver.save(sess, model_path, global_step=i)

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

with tf.Session(config=tf.ConfigProto(
        intra_op_parallelism_threads=NUM_THREADS)) as sess:
    printNumParams()
    print('INIT VARIABLES!')
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver(max_to_keep=5)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    train()
    # writeReEncoded()
    # runTimeEstimate(sess)

    coord.request_stop()
    coord.join(threads)
