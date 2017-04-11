
import tensorflow as tf

def encodeYear(year):
    ''' Given digit, encodes year in vector representation'''
    number = tf.mod(year, 1000)
    return tf.one_hot(number, 1000)

def encodeYearAsDigits(year):
    ones = tf.mod(year, 10)
    year = tf.floordiv(year, 10)
    tens = tf.mod(year, 10)
    year = tf.floordiv(year, 10)
    hundreds = tf.mod(year, 10)

    return [tf.one_hot(d, 10) for d in [hundreds, tens, ones]]


def predict(year_log):
    ''' Given network output, computes prediction.'''
    # year = 1000 + tf.argmax(year_prob, axis=1)
    digits = [tf.argmax(prob, axis=1) for prob in year_log]

    # Combine scalar values to a full year
    year = 1000 + sum([x*y for x,y in zip([100, 10, 1], digits)])

    return tf.cast(year, tf.int32)

def certainty(year_prob, years):
    ''' Returns probability of the correct label.
        Note: does not return the highest probability.'''
    years = tf.squeeze(years)
    indices = tf.mod(years, 1000)
    hots = tf.one_hot(indices, 1000, dtype=tf.float32, axis=-1)

    certainties = tf.reduce_sum(year_prob * hots, axis=1)
    return certainties

def clusterError(year_prob):
    ''' DEPRECATED.
        Returns cost for having multiple clusters.
        Takes into account distances between years.
        Expects input in batch mode.
        The error value will always be in the range [0, 1).

        This method has several problems:
        1. For a uniform probability vector, it has strong bias for
           different years. That means that some year labels will have
           greater influence than others on the training.
        2. For a probability vector x=[0.5, 0, 0, 0, ... 0, 0.5], the cluster
           error is minimized in the middle, not at the edges as expected!
        '''
    largest_cluster = tf.argmax(year_prob, axis=1)
    largest_cluster = tf.cast(largest_cluster, dtype=tf.float32)

    [batch_size, y_range] = year_prob.get_shape()
    y_range = tf.cast(y_range, tf.float32)

    indices = tf.range(y_range, dtype=tf.float32)
    diff = tf.map_fn(lambda pos: indices - pos, largest_cluster)
    weights = tf.square(diff/y_range)
    return 2 * tf.reduce_sum(weights * year_prob, axis=-1)

def error(year_pair, year_log):
    ''' Error function to minimize.
        Label year is an integer in range [1000-1999].
        Supports batches.
    '''
    min_year = tf.slice(year_pair, [0,0], [-1,1])
    max_year = tf.slice(year_pair, [0,1], [-1,1])
    year = tf.floordiv(min_year + max_year, 2)

    encoded_year = encodeYearAsDigits(year)
    compare_years = [tf.nn.softmax_cross_entropy_with_logits(logits=x, labels=y)
                     for x, y in zip(year_log, encoded_year)]
    year_error = sum(compare_years)

    # year = tf.mod(year, 1000)
    # year_label = tf.one_hot(year, 1000)
    # year_error = tf.nn.softmax_cross_entropy_with_logits(logits=year_log,
    #                                                      labels=year_label)
    #
    # year_prob = tf.nn.softmax(year_log)
    return year_error
    # return year_error + clusterError(year_prob)
