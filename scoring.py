
import tensorflow as tf

def error(year, decision_prob, year_prob):
    ''' Error function to minimize.
        Label year is an integer in range [1000-1999] OR non-positive.
        Supports batches.
    '''
    has_number = tf.greater(year, 0)
    has_number_as_int = tf.cast(has_number, tf.int32)
    decision_label = tf.one_hot(has_number_as_int, 2)
    decision_error = tf.nn.softmax_cross_entropy_with_logits(decision_prob,
                                                             decision_label)

    # If year < 1000, then one_hot makes a zero vector
    # and the entropy evaluates to 0.
    year_label = tf.one_hot(year - 1000, 1000)
    year_error = tf.nn.softmax_cross_entropy_with_logits(year_prob,
                                                         year_label)

    return decision_error + year_error
