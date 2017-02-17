import numpy as np
import tensorflow as tf

import image_transform as it

''' Input: image tensor of shape [width, height, channels]
           and global threshold.
    Output: boolean image tensor of shape [width, height, channels].
            Elements are true for black pixels.

    Uses global thresholding
'''
def binarizeGlobal(image, threshold):
    return tf.less(image, threshold)

''' Numpy function for aggregating a histogram of the pixel counts.

    Returns a numpy array of 256 elements with the count for each value.
'''
def pyHistogram(pixelValues, counts):
    histogram = np.zeros((256,), dtype=np.int32)

    for i in range(len(pixelValues)):
        intensity = pixelValues[i]
        histogram[intensity] = counts[i]

    return histogram

''' Given an intensity histogram, computes Otsu's global threshold.'''
def pyOtsus(histogram):
    total = sum(histogram)

    # Maintains class count.
    # Divide by total count above to get class probability.
    w_black = 0
    w_white = total

    # Maintains weighted count.
    # Divide by class count above to get expectation.
    mu_black = 0
    mu_white = 0
    for i in range(len(histogram)):
        mu_white = mu_white + i * histogram[i]

    highest_var = 0
    best_t = 0
    for t in range(len(histogram)):
        # Update counts
        w_diff = histogram[t]
        w_black = w_black + w_diff
        w_white = w_white - w_diff

        mu_black = mu_black + t * w_diff
        mu_white = mu_white - t * w_diff

        # Compute variance
        if w_black == 0 or w_white == 0:
            continue
        diff = mu_black/w_black - mu_white/w_white
        sigma2 = (w_black/total) * (w_white/total) * diff * diff
        if sigma2 > highest_var:
            highest_var = sigma2
            best_t = t

    return np.uint8(best_t)

def pyOtsusGlobalThreshold(pixelValues, counts):
    histogram = pyHistogram(pixelValues, counts)
    return pyOtsus(histogram)

def otsusGlobalThreshold(image):
    cropped = it.cropEdges(image)
    array = tf.reshape(cropped, shape=[-1])
    pixelValues, _, counts = tf.unique_with_counts(array)
    threshold = tf.py_func(pyOtsusGlobalThreshold, [pixelValues, counts], tf.uint8, stateful=False)

    binarized = binarizeGlobal(image, threshold)
    return (binarized, threshold)
