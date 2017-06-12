import util.img_debug as debug
import network.attend as att
import network.decoder as dec
import network.encoder as enc

import tensorflow as tf


def runNetwork(batch_images, train_mode):
    if train_mode:
        keep_prob = 0.5
    else:
        keep_prob = 1.0

    # Network
    print('Input: ', batch_images.get_shape())
    activation = enc.deepEncoder(batch_images)
    print('Batch embedding:', activation.get_shape())

    attention = att.attend(activation, keep_prob)
    print('Attention: ', attention.get_shape())

    # Can debug attention:
    # attention = tf.Print(attention, ['WRITE ATTENTION',
    #                                  debug.debugFirstImage(batch_images, 'input'),
    #                                  debug.debugAttention(attention)])

    attended = tf.reduce_sum(activation * attention, [1, 2])

    print('Attend: ', attended.get_shape())
    year_log = dec.decodeNumber(attended, keep_prob)
    return year_log
