import gflags
import tensorflow as tf

import os

gflags.DEFINE_string('model_dir', 'models', 'Directory for saving and loading models')


def saveModel(saver, model_name, step):
    print('Saving ', model_name)
    save_name = os.path.join(gflags.FLAGS.model_dir, model_name)
    save_path = saver.save(sess, save_name, global_step=step, write_meta_graph=False)


def loadModel(sess, model_name=None):
    ''' Restores network variables of the latest session
        OR loads the specified model, e.g. 'soft_attention_2-1050'.
    '''
    saver = tf.train.Saver()
    if model_name:
        version = os.path.join(gflags.FLAGS.model_dir, model_name)
    else:
        version = tf.train.latest_checkpoint(gflags.FLAGS.model_dir)

    print('Restoring model from:', version)
    saver.restore(sess, version)

def lazyLoadModel(model_name=None):
    return lambda sess: loadModel(sess, model_name=model_name)
