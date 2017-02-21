
import network.conv as conv
import preprocess.load_iris as iris
import preprocess.grayscale as gray
import scoring as sc

import tensorflow as tf

# TODO: replace with FLAGS
model_path = 'models'
train_data_path = 'D:Data\\french_parish_train'
test_data_path = 'D:Data\\french_parish_test'


# Input variables
image = tf.placeholder(tf.float32, shape=[325, 260, 1], name='image')
year = tf.placeholder(tf.int32, shape=[], name='year')
keep_prob = tf.placeholder(tf.float32)

batch = tf.expand_dims(image, 0)

# Network
activation = conv.encodeImage(batch)
decision_prob, year_prob = conv.decodeNumber(activation, keep_prob)
error = sc.error(year, decision_prob, year_prob)
train_step = tf.train.AdamOptimizer(1e-4).minimize(error)

# TODO: batch accuracy


with tf.Session() as sess:
    print('INIT VARIABLES!')
    sess.run(tf.global_variables_initializer())

    print('Start generator')
    gen = iris.inputNames(train_data_path)
    # all_files = list(gen)
    # filename_queue = tf.train.string_input_producer(all_files, shuffle=True, seed=1)

    for i in range(10):
        # TODO increase range...
        # TODO batch training
        # basename = tf.as_string(filename_queue.dequeue())
        basename = gen.__next__()
        jpg_path, y = iris.loadPair(train_data_path, basename)
        print('Y: ', y)

        image_content = tf.read_file(jpg_path)
        x_tensor = tf.image.decode_jpeg(image_content, channels=1, name='image')
        x_tensor = tf.image.resize_images(x_tensor, [325, 260])
        x = tf.cast(x_tensor, tf.uint8).eval()
        x, _ = gray.otsusGlobalThreshold(x)

        train_step.run(feed_dict={image: x.eval(), year: y, keep_prob: 0.5})
