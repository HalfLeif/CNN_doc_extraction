''' See README '''

import ops.classify
import ops.model_io
import ops.time
import ops.time_estimate
import ops.test
import ops.train

import gflags
import tensorflow as tf

import sys


gflags.DEFINE_integer('NUM_THREADS', 3, 'Number of threads created for tensorflow.', lower_bound=1)


def runLambdas(lambdas):
    ''' Runs the lambdas in order in this Session.

        All necessary computation nodes for running the lambdas must already
        be added to the current computation graph!
    '''
    with tf.Session(config=tf.ConfigProto(
            intra_op_parallelism_threads=gflags.FLAGS.NUM_THREADS)) as sess:
        print('INIT VARIABLES!')
        sess.run(tf.global_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        print('System ready!')
        for f in lambdas:
            f(sess)

        coord.request_stop()
        coord.join(threads)


def main(argv):
    gflags.FLAGS(argv)

    # Whether to train on 4-digit MNIST or not.
    pretrain_mnist = False

    # When each lambda is created, it adds the necessary nodes to the current
    # computation graph.
    # When a lambda is executed, it runs the relevant computation nodes.
    model_name = 'Swe_DEP3_ind_digits7-2299'
    lambdas = [
        ops.time_estimate.lazyPrintNumParams(),
        ops.model_io.lazyLoadModel(model_name),
        ops.time.lazyStart(),
        # ops.train.lazyTrain('TEMP', pretrain_mnist),
        # ops.test.lazyTest(pretrain_mnist, num_batches=None, debug=False),
        ops.classify.lazyClassify('test', output_name='test_' + model_name),
        ops.classify.lazyClassify('eval', output_name='eval_' + model_name),
        # ops.time_estimate.lazyTimeEstimate(),
        ops.time.lazyClock()
    ]
    runLambdas(lambdas)


if __name__ == '__main__':
    main(sys.argv)
