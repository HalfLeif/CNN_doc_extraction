
import tensorflow as tf

def replicateAlong(tensor, repetitions, axis):
    ''' Replicates tensor along given dimension.
        For example [1, 10, 1] -> [1, 10, 4].'''
    many = [tensor for i in range(repetitions)]
    return tf.concat(many, axis=axis)
