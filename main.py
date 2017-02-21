
import network.conv as conv
import preprocess.load_iris as iris
import preprocess.grayscale as gray
import scoring as sc

import tensorflow as tf

# TODO: replace with FLAGS
model_path = 'models'
train_data_path = 'D:Data\\french_parish_train'
test_data_path = 'D:Data\\french_parish_test'

FIX_SHAPE = [1500, 950]

# Input variables
image = tf.placeholder(tf.float32, shape=[1500, 950, 1], name='image')
year = tf.placeholder(tf.int32, shape=[], name='year')
keep_prob = tf.placeholder(tf.float32)

batch = tf.expand_dims(image, 0)

# Network
activation = conv.deepEncoder(batch)
decision_prob, year_prob = conv.decodeNumber(activation, keep_prob)
error = sc.error(year, decision_prob, year_prob)
train_step = tf.train.AdamOptimizer(1e-4).minimize(error)

# TODO: batch accuracy


print('FIND FILES')
all_files = iris.inputNames(train_data_path)
filename_queue = tf.train.string_input_producer(all_files, shuffle=True, seed=1)

with tf.Session() as sess:
    print('INIT VARIABLES!')
    sess.run(tf.global_variables_initializer())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(10):
        pairname_tensor = filename_queue.dequeue()
        pairname = pairname_tensor.eval().decode('utf-8')
        jpg_path, y = iris.loadPair(train_data_path, pairname)
        print(pairname, 'Year:', y)

        image_content = tf.read_file(jpg_path)
        x_tensor = tf.image.decode_jpeg(image_content, channels=1, ratio=2, name='image')
        x_tensor = tf.image.resize_images(x_tensor, FIX_SHAPE)
        x = tf.cast(x_tensor, tf.uint8)
        x, _ = gray.otsusGlobalThreshold(x)

        train_step.run(feed_dict={image: x.eval(), year: y, keep_prob: 0.5})

    coord.request_stop()
    coord.join(threads)
