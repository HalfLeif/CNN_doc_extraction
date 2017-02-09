import numpy as np
import tensorflow as tf

''' Input: image tensor of shape [width, height, channels]
           and global threshold.
    Output: boolean image tensor of shape [width, height].
            Elements are true for black pixels.

    Uses global thresholding
'''
def binarizeGlobal(image, threshold):
    # TODO: reshape into one lower dimension.
    return tf.less(image, threshold)

'''Sum over one_hot constants. Works but is too slow!'''
def slowHistogram(arr):
    one_hots = tf.one_hot(array, 256, on_value=1, off_value=0)
    histogram = tf.reduce_sum(one_hots, 0)
    return histogram

def addHistogram(histogram, ix):
    vec = tf.one_hot(ix, 256, on_value=val, off_value=0)
    return tf.add(histogram, vec)

''' Numpy function for aggregating a histogram of the pixel counts.

    Returns a numpy array of 256 elements with the count for each value.
'''
def pyHistogram(pixelValues, counts):
    print('call received!')
    histogram = np.zeros((256,), dtype=np.int32)

    for i in xrange(len(pixelValues)):
        intensity = pixelValues[i]
        histogram[intensity] = counts[i]

    return histogram

def otsusGlobalThreshold(image):
    array = tf.reshape(image, shape=[-1])
    pixelValues, _, counts = tf.unique_with_counts(array)
    histogram = tf.py_func(pyHistogram, [pixelValues, counts], tf.int32, stateful=True)

    # TODO

    return histogram
