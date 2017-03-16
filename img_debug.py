import tensorflow as tf
import numpy as np

import os
import sys

def py_WriteImage(re_encoded, label):
    tempdir = os.path.join('data', 'temp_imgs')
    os.makedirs(tempdir, exist_ok=True)

    filename = os.path.join(tempdir, 'temp_'+str(label)+'.jpg')
    with open(filename, 'wb+') as f:
        f.write(re_encoded)
    print('WROTE RE-ENCODED IMAGE to ', filename)
    return np.int32(0)

def debugImage(image, label):
    image = tf.squeeze(image)
    image = tf.expand_dims(image, -1)
    restore = tf.cast(image*255, tf.uint8)
    re_encoded = tf.image.encode_jpeg(restore)
    write_op = tf.py_func(py_WriteImage, [re_encoded, label], tf.int32, stateful=True)
    return write_op

def debugFirstImage(batch_images, label):
    image = tf.slice(batch_images, [0, 0, 0, 0], [1, -1, -1, -1])
    return debugImage(image, label)

def py_rebuildBottomAttention(attention):
    ''' Sums attention over their receptive field in the size of the
        original image. The receptive field and step sizes
        depend on the number of layers in the network.'''
    rec_field_y = 68
    rec_field_x = 132
    step_y = 2**5
    step_x = 2**5

    len_y = len(attention)
    len_x = len(attention[0])

    out_y = rec_field_y + step_y * (len_y - 1)
    out_x = rec_field_x + step_x * (len_x - 1)

    out = np.zeros((out_y, out_x), dtype=np.float32)

    for i in range(len_y):
        start_y = i*step_y
        stop_y = start_y + rec_field_y
        for j in range(len_x):
            start_x = j*step_x
            stop_x = start_x + rec_field_x

            out[start_y:stop_y, start_x:stop_x] += attention[i,j]
    return out

def py_printMatrix(matrix):
    for row in matrix:
        for elem in row:
            sys.stdout.write("{:.4f}".format(elem))
            # sys.stdout.write("{:6.4f}".format(elem*255))
            sys.stdout.write(' ')
        sys.stdout.write('\n')
    return np.int32(0)

def debugAttention(attention):
    first_attention = tf.slice(attention, [0, 0, 0, 0], [1, -1, -1, -1])
    first_attention = tf.squeeze(first_attention)
    rebuild_attention = tf.py_func(py_rebuildBottomAttention, [first_attention], tf.float32, stateful=False)
    # print_op = tf.py_func(py_printMatrix, [first_attention], tf.int32, stateful=True)
    # print_op = tf.Print(print_op, [debugImage(first_attention, 'attention')])
    return debugImage(rebuild_attention, 'rebuilt')

if __name__ == '__main__':
    # x = np.arange(9)
    # x = np.reshape(x, (3,3))
    x = np.zeros((3,3), dtype=np.float32) + 1
    print(x)
    # x[1:2,:] = 1
    # print(x)
    y = py_rebuildBottomAttention(x)
    print(y)
