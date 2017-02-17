
import tensorflow as tf

'''Crops edges by 10%'''
def cropEdges(image):
    shape = tf.to_float(tf.shape(image))

    height = shape[0]
    width = shape[1]
    depth = shape[2]

    begin = tf.to_int32([0.1*height, 0.1*width, 0])
    size = tf.to_int32([0.8*height, 0.8*width, depth])
    return tf.slice(image, begin, size)
