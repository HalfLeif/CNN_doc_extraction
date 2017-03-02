
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as mnist_data

import pretrain.mnist_debug as debug

def mnistData(train_mode):
    mnist = mnist_data.read_data_sets('MNIST_data', one_hot=False)
    if train_mode:
        return mnist.train.images, mnist.train.labels
    else:
        return mnist.test.images, mnist.test.labels

def stackMnist(imgs, n):
    imgs = tf.reshape(imgs, [n, 28, 28, 1])
    image_list = tf.unstack(imgs)
    wide_image = tf.concat(image_list, axis=1)
    return wide_image

def fragmentImage(image):
    ''' Takes image of shape [784]. Creates fragments of 14x14'''
    image = tf.reshape(image, [28, 28])
    fragments = tf.split(image, 4, axis=0)
    fragments = tf.stack(fragments)
    return fragments

def noiseImage(digit_queue, n):
    ''' Creates an image of noise with shape [28, n*28].'''
    n_images, years = tf.train.batch(digit_queue, batch_size=n)

    n_images = tf.map_fn(fragmentImage, n_images)
    n_images = tf.reshape(n_images, [n*4, 7, 28])
    n_images = tf.random_shuffle(n_images)
    n_images = stackMnist(n_images, n)

    # n_images = tf.Print(n_images, ['WRITE NOISE', debug.debugImage(n_images, years)])
    return n_images

def mnistSample(train_mode):
    ''' Returns two tensors: a four digit image and its label.
        MNIST images uses 1.0 for ink and 0.0 for background.
    '''
    imgs, labels = mnistData(train_mode)
    shuffled = tf.train.slice_input_producer([imgs, labels], shuffle=True, seed=None)
    four_images, four_labels = tf.train.batch(shuffled, batch_size=4)

    wide_image = stackMnist(four_images, 4)

    noise_width = 10
    noise = noiseImage(shuffled, noise_width)
    left, right = tf.split(noise, 2, axis=1)

    parts = [left, wide_image, right]
    parts = list(map(lambda p: tf.pad(p, [[6,6],[4,4],[0,0]]), parts))
    wide_image = tf.concat(parts, axis=1)

    four_labels = tf.cast(four_labels, tf.int32)
    year = tf.reduce_sum(four_labels * tf.constant([1000, 100, 10, 1], tf.int32))

    # year = tf.Print(year, ['WRITE IMG', debug.debugImage(wide_image, year)])

    return wide_image, year

def mnistBatch(batch_size, train_mode):
    wide_image, year = mnistSample(train_mode)
    return tf.train.batch([wide_image, year], batch_size=batch_size)
