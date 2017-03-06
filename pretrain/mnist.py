import numpy as np
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

def padNoise(image, noise_width, queue):
    noise = noiseImage(queue, noise_width)
    left, right = tf.split(noise, 2, axis=1)

    parts = [left, image, right]
    parts = list(map(lambda p: tf.pad(p, [[6,6],[4,4],[0,0]]), parts))
    wide_image = tf.concat(parts, axis=1)
    return wide_image


def padRandom(image):
    ''' Places the image at a random position in a rectangle
        with shape `crop_shape`.'''

    add_top = 28
    add_side = 56
    pad_all = tf.pad(image, [[add_top,add_top],[add_side,add_side],[0,0]])

    shape = image.get_shape()
    crop_shape = [int(shape[0])+add_top, int(shape[1])+add_side, int(shape[2])]
    return tf.random_crop(pad_all, crop_shape)


def addDotNoise(image):
    shape = image.get_shape()
    dots = tf.random_uniform(shape, minval=0.6, maxval=1.0, dtype=tf.float32)

    black_ink = 1 - image
    black_ink = black_ink * dots
    return 1 - black_ink

def ones(imgs, labels):
    ''' Returns list of images whose label is 1.'''
    return np.array([imgs[i] for i in range(len(labels)) if labels[i] == 1])

def mnistSample(train_mode):
    ''' Returns two tensors: a four digit image and its label.
        MNIST images uses 1.0 for ink and 0.0 for background.
    '''
    imgs, labels = mnistData(train_mode)
    shuffled = tf.train.slice_input_producer([imgs, labels], shuffle=True, seed=None)
    three_images, three_labels = tf.train.batch(shuffled, batch_size=3)

    one_imgs = ones(imgs, labels)
    ones_queue = tf.train.input_producer(one_imgs, shuffle=True, seed=None)
    one_img = ones_queue.dequeue()
    one_img = tf.expand_dims(one_img, 0)

    four_images = tf.concat([one_img, three_images], axis=0)
    wide_image = stackMnist(four_images, 4)

    noise_width = 10
    # wide_image = padNoise(wide_image, noise_width, shuffled)
    # wide_image = padEmpty(wide_image, noise_width)
    wide_image = padRandom(wide_image)
    wide_image = addDotNoise(wide_image)

    three_labels = tf.cast(three_labels, tf.int32)
    year = 1000 + tf.reduce_sum(three_labels * tf.constant([100, 10, 1], tf.int32))

    year = tf.Print(year, ['WRITE IMG', debug.debugImage(wide_image, year)])

    return wide_image, year

def mnistBatch(batch_size, train_mode):
    wide_image, year = mnistSample(train_mode)
    return tf.train.batch([wide_image, year], batch_size=batch_size)
