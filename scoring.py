
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

def certainty(year_prob, years):
    ''' Returns probability of the correct label.
        Note: does not return the highest probability.'''
    indices = tf.mod(years, 1000)
    hots = tf.one_hot(indices, 1000, dtype=tf.float32, axis=-1)

    certainties = tf.reduce_sum(year_prob * hots, axis=1)
    return certainties

def clusterError(year_prob):
    ''' Returns cost for having multiple clusters.
        Takes into account distances between years.
        Expects input in batch mode.
        The error value will always be in the range [0, 1).'''
    largest_cluster = tf.argmax(year_prob, axis=1)
    largest_cluster = tf.cast(largest_cluster, dtype=tf.float32)

    [batch_size, y_range] = year_prob.get_shape()
    y_range = tf.cast(y_range, tf.float32)

    indices = tf.range(y_range, dtype=tf.float32)
    diff = tf.map_fn(lambda pos: indices - pos, largest_cluster)
    weights = tf.square(diff/y_range)
    return 2 * tf.reduce_sum(weights * year_prob, axis=-1)

def error(year, year_prob):
    ''' Error function to minimize.
        Label year is an integer in range [1000-1999].
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
    year = tf.mod(year, 1000)
    year_label = tf.one_hot(year, 1000)
    year_error = tf.nn.softmax_cross_entropy_with_logits(logits=year_prob,
                                                         labels=year_label)

    return year_error + clusterError(year_prob)

    # total_err = tf.reduce_mean(decision_error + year_error)
    # return total_err
