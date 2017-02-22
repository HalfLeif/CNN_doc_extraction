
import network.conv as conv
import preprocess.load_iris as iris
import preprocess.grayscale as gray
import scoring as sc

import tensorflow as tf

import os

# TODO: replace with FLAGS
model_name = 'deep_cnn'
model_path = os.path.join('models', model_name)
train_data_path = 'D:Data\\french_parish_train'
test_data_path = 'D:Data\\french_parish_test'

FIX_SHAPE = [1500, 950]
BATCH_SIZE = 15

# Input variables
images = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 1500, 950, 1], name='images')
years = tf.placeholder(tf.int32, shape=[BATCH_SIZE], name='years')
keep_prob = tf.placeholder(tf.float32)

# Network
activation = conv.deepEncoder(images)
print('Batch embedding:', activation.get_shape())
decision_prob, year_prob = conv.decodeNumber(activation, keep_prob)
error = sc.error(years, decision_prob, year_prob)
train_step = tf.train.AdamOptimizer(1e-4).minimize(error)

def prediction(decision_prob, year_prob):
    decision = tf.argmax(decision_prob, axis=1)
    year = tf.argmax(year_prob, axis=1)

    # decision is either 0 or 1, so returns -1 or `year`.
    return tf.cast(decision*year + (decision-1), tf.int32)

correct_prediction = tf.equal(years, prediction(decision_prob, year_prob))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


print('FIND FILES')
all_files = iris.inputNames(train_data_path)
filename_queue = tf.train.string_input_producer(all_files, shuffle=True, seed=1)

def loadImage(jpg_path):
    image_content = tf.read_file(jpg_path)
    x = tf.image.decode_jpeg(image_content, channels=1, ratio=2, name='image')
    x = tf.image.resize_images(x, FIX_SHAPE)
    x = tf.cast(x, tf.uint8)
    x, _ = gray.otsusGlobalThreshold(x)
    return x.eval()

with tf.Session() as sess:
    print('INIT VARIABLES!')
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver(max_to_keep=5)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(1000):
        print('BATCH', i)

        pairname_tensor = filename_queue.dequeue_many(BATCH_SIZE)
        pairnames = [p.decode('utf-8') for p in pairname_tensor.eval()]

        jpg_paths = iris.getJpgPaths(train_data_path, pairnames)
        xs = list(map(loadImage, jpg_paths))
        ys = list(iris.parseXmlFiles(train_data_path, pairnames))

        if (i%50 == 0):
            acc = accuracy.eval(feed_dict={images: xs, years: ys, keep_prob: 1.0})
            print('Accuracy: ', acc)

        train_step.run(feed_dict={images: xs, years: ys, keep_prob: 0.5})
        savepath = saver.save(sess, model_path, global_step=i)

    coord.request_stop()
    coord.join(threads)
