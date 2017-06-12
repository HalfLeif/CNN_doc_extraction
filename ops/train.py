import loading.load_swe as swe
import network.run as run
import network.scoring as sc
import ops.model_io as io
import pretrain.mnist as mnist

import tensorflow as tf


def trainOp(pretrain=True):
    if pretrain:
        (batch_images, batch_years), num_batches = mnist.mnistBatch(True)
    else:
        batch_images, batch_years, _, num_batches = swe.sweBatch('train')

    year_log = run.runNetwork(batch_images, True)
    error = sc.error(batch_years, year_log)
    train_step = tf.train.AdamOptimizer(1e-4).minimize(error)

    return train_step, num_batches


def train(model_name, train_step, n_train_batches):
    saver = tf.train.Saver(max_to_keep=3)

    for i in range(n_train_batches):
        if (i%100 == 0):
            print('BATCH', i)

        train_step.run()

        if (i%100 == 99):
            io.saveModel(saver, model_name, i)
        if (i%100 == 99):
            accuracy.eval()
    io.saveModel(saver, model_name, i)

def lazyTrain(model_name, pretrain_mnist):
    train_step, n_train_batches = trainOp(pretrain_mnist)
    return lambda _: train(model_name, train_step, n_train_batches)
