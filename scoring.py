
import tensorflow as tf

def encodeLabel(years):
    ''' Takes a python list of years [1000-1999],
        returns whether the list is empty
        tupled with maximum year in form of onehot [0-999].'''

    has_number = tf.greater(len(years), 0)
    maxyear = tf.reduce_max(tf.constant(years, tf.int32))
    number = maxyear - 1000
    return has_number, tf.one_hot(number, 1000)


def error(label, decision_prob, year_prob):
    '''Error function to minimize.'''
    has_number, year_label = encodeLabel(label)

    has_number_as_int = tf.cast(has_number, tf.int32)
    decision_label = tf.one_hot(has_number_as_int, 2)
    decision_error = tf.nn.softmax_cross_entropy_with_logits(decision_prob,
                                                             decision_label)

    year_entropy = tf.nn.softmax_cross_entropy_with_logits(year_prob,
                                                           year_label)
    year_error = tf.cond(has_number, lambda: year_entropy,
                         lambda: tf.constant(0, dtype=tf.float32))

    return decision_error + year_error
