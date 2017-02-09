import tensorflow as tf
import sys, os



filename_queue = tf.train.string_input_producer(
    tf.train.match_filenames_once("./data/*.jpg"))

image_reader = tf.WholeFileReader()

_, image_file = image_reader.read(filename_queue)

# Grayscale image: channels=1
# Can downscale with: ratio=2/4/8
image = tf.image.decode_jpeg(image_file, channels=1, ratio=2)


def printImage(image_tensor):
    rows = len(image_tensor)
    cols = len(image_tensor[0])

    # print(image_tensor[1][1])

    for i in xrange(rows):
        for j in xrange(cols):
            cell = image_tensor[i][j][0]
            if cell < 64:
                out = '#'
            elif cell < 128:
                out = '='
            elif cell < 196:
                out = '-'
            else:
                out = '.'
            sys.stdout.write(out)
        sys.stdout.write('\n')


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("Session started")

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    print("Evaluate...")
    # print(image.get_shape().as_list())
    # image_tensor, image_shape = sess.run([image, image.get_shape().as_list()])

    result = sess.run([image])
    print("Evaluated")
    image_tensor = result[0]
    # print(image.get_shape().as_list())




    coord.request_stop()
    coord.join(threads)
