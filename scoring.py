
import tensorflow as tf

def encodeYear(year):
    ''' Given digit, encodes year in vector representation'''
    # ones = tf.mod(year, 10)
    # year = tf.floordiv(year, 10)
    # tens = tf.mod(year, 10)
    # year = tf.floordiv(year, 10)
    # hundreds = tf.mod(year, 10)
    #
    # return [tf.one_hot(d, 10) for d in [hundreds, tens, ones]]
    number = tf.mod(year, 1000)
    return tf.one_hot(number, 1000)


def predict(year_prob):
    ''' Given network output, computes prediction.'''
    # decision = tf.argmax(decision_prob, axis=1)
    year = 1000 + tf.argmax(year_prob, axis=1)
    # digits = [tf.argmax(prob, axis=1) for prob in year_prob]

    # Combine scalar values to a full year
    # year = 1000 + sum([x*y for x,y in zip([100, 10, 1], digits)])

    # decision is either 0 or 1, so returns -1 or `year`.
    # return tf.cast(decision*year + (decision-1), tf.int32)
    return tf.cast(year, tf.int32)


def error(year, year_prob):
    ''' Error function to minimize.
        Label year is an integer in range [1000-1999] OR non-positive.
        Supports batches.
    '''
    # has_number = tf.greater(year, 0)
    # has_number_as_int = tf.cast(has_number, tf.int32)

    # decision_label = tf.one_hot(has_number_as_int, 2)

    # decision_label = tf.Print(decision_label, [year, has_number_as_int, decision_label], summarize=100)
    # decision_error = tf.nn.softmax_cross_entropy_with_logits(decision_prob,
    #                                                          decision_label)

    # Experiment with independent digit learning instead of combined...
    # encoded_year = encodeYear(year)
    # compare_years = [tf.nn.softmax_cross_entropy_with_logits(x, y)
    #                  for x, y in zip(year_prob, encoded_year)]
    # year_error = sum(compare_years)

    # If year < 1000, then one_hot makes a zero vector
    # and the entropy evaluates to 0.
    year_label = tf.one_hot(tf.mod(year, 1000), 1000)
    year_error = tf.nn.softmax_cross_entropy_with_logits(year_prob,
                                                         year_label)

    return year_error

    # total_err = tf.reduce_mean(decision_error + year_error)
    # return total_err
