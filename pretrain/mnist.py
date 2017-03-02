
import tensorflow as tf
# import tensorflow.contrib.learn.python.learn.datasets.mnist as mnist_data
import tensorflow.examples.tutorials.mnist.input_data as mnist_data

def mnistData(train_mode):
    mnist = mnist_data.read_data_sets('MNIST_data', one_hot=False)
    if train_mode:
        return mnist.train.images, mnist.train.labels
    else:
        return mnist.test.images, mnist.test.labels

def mnistSample(train_mode):
    ''' Returns two tensors: a four digit image and its label.
        MNIST images uses 1.0 for ink and 0.0 for background.
    '''
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
