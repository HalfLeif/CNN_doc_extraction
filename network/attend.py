
import tensorflow as tf
from tensorflow.contrib import slim

def attend_vector(feature_vector, keep_prob):
    ''' Returns a scalar in range [0-1] for this activation unit.'''
    with slim.arg_scope([slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                        weights_regularizer=slim.l2_regularizer(0.0005)):

        net = tf.expand_dims(feature_vector, 0)
        net = slim.fully_connected(net, 256, scope='attend_1')
        net = slim.dropout(net, keep_prob, scope='att_dropout1')

        # net = slim.fully_connected(net, 64, scope='attend_2')
        # net = slim.dropout(net, keep_prob, scope='att_dropout2')

        attention = slim.fully_connected(net, 1, activation_fn=None, scope='attend_out')
        return tf.squeeze(attention)

def attend_image(activation_3d, keep_prob):
    ''' Takes an activation map with units of depth D,
        returns the normalized attention weights for each batch item.'''
    atts = tf.map_fn(lambda vectors: tf.map_fn(lambda vec: attend_vector(vec, keep_prob), vectors), activation_3d)

    # Softmax on attention over all dimensions:
    atts = tf.exp(atts)
    atts = atts / tf.reduce_sum(atts)
    return atts

def attend(net, keep_prob):
    ''' Computes soft attention for all activations in this batch.'''
    atts = tf.map_fn(lambda img: attend_image(img, keep_prob), net)
    return tf.expand_dims(atts, -1)

def hardAttend(net, keep_prob):
    ''' Computes hard attention by maximum instead of sampling.
        Interface identical as for soft attention above.
    '''
    attention = attend(net, keep_prob)
    [batch_size, height, width, depth] = tf.unstack(tf.shape(net))

    flat_att = tf.reshape(attention, [batch_size, height*width])
    max_att = tf.argmax(flat_att, axis=1)
    att_vector = tf.one_hot(max_att, height*width, dtype=tf.float32)
    att_matrix = tf.reshape(att_vector, [batch_size, height, width, 1])

    return att_matrix
