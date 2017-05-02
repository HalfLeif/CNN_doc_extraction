
import ops.train

import tensorflow as tf
from tensorflow.python.client import timeline

import os


def numParams():
    def varParams(var):
        prod = 1
        for dim in var.get_shape():
            prod = prod * dim.value
        return prod

    num_params = 0
    for var in tf.trainable_variables():
        num_params = num_params + varParams(var)
    return num_params


def lazyPrintNumParams():
    return lambda _: print('# PARAMETERS:', numParams())


def runTimeEstimate(sess):
    ''' Creates timeline-file for debugging expensive operations.
        Open file with chrome://tracing.
    '''
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()

    train_step, _ = ops.train.trainOp(True)
    sess.run(train_step, options=run_options, run_metadata=run_metadata)

    tl = timeline.Timeline(run_metadata.step_stats)
    ctf = tl.generate_chrome_trace_format()

    filename = os.path.join('data', 'timeline.json')
    with open(filename, 'w+') as f:
        f.write(ctf)
    print('Wrote timeline to', filename)

def lazyTimeEstimate():
    return lambda sess: runTimeEstimate(sess)
