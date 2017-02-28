
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# TODO REMOVE:
import sys

def mnistData(train_mode):
    mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
    if train_mode:
        return mnist.train.images, mnist.train.labels
    else:
        return mnist.test.images, mnist.test.labels

def mnistSample(train_mode):
    ''' Returns two tensors: a four digit image and its label.'''
    imgs, labels = mnistData(train_mode)
    shuffled = tf.train.slice_input_producer([imgs, labels], shuffle=True, seed=None)
    four_images, four_labels = tf.train.batch(shuffled, batch_size=4)
    print('FOUR LABELS', four_labels.get_shape())

    four_images = tf.reshape(four_images, [4, 28, 28])
    image_list = tf.unstack(four_images)
    stacked = tf.stack(image_list, axis=1)
    wide_image = tf.map_fn(lambda row: tf.reshape(row, [4*28, 1]), stacked)

    four_labels = tf.cast(four_labels, tf.int32)
    year = tf.reduce_sum(four_labels * tf.constant([1000, 100, 10, 1], tf.int32))
    return wide_image, year

def mnistBatch(batch_size, train_mode):
    wide_image, year = mnistSample(train_mode)
    return tf.train.batch([wide_image, year], batch_size=batch_size)

def printImage(image):
    for i in range(len(image)):
        for j in range(len(image[0])):
            pixel = image[i][j][0]
            out = '_'
            if pixel > 0.7:
                out = '#'
            elif pixel > 0.3:
                out = '/'
            sys.stdout.write(out)
        sys.stdout.write('\n')

if __name__ == '__main__':
    pixels, year = mnistBatch(5)
    print(pixels.get_shape())
    print(year.get_shape())

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        xs, ys = sess.run([pixels, year])
        print(ys)
        for x in xs:
            printImage(x)

        coord.request_stop()
        coord.join(threads)
