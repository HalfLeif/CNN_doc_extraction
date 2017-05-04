import loading.load_swe as swe
import network.run as run
import util.file

import numpy as np
import tensorflow as tf

import os


def classifyOp(dataset):
    batch_images, _, batch_paths, num_batches = swe.sweBatch(dataset, shuffle=False)
    year_log = run.runNetwork(batch_images, False)
    return year_log, batch_paths, num_batches


def classify(sess, year_log, batch_paths, num_batches):
    for i in range(num_batches):
        readout_logits, readout_paths = sess.run([year_log, batch_paths])
        for j in range(len(readout_paths)):
            digit1 = readout_logits[0][j]
            digit2 = readout_logits[1][j]
            digit3 = readout_logits[2][j]
            yield readout_paths[j], digit1, digit2, digit3

def formatArray(arr):
    return np.array2string(arr, separator=', ', precision=12,
                           max_line_width=float('inf'))

def saveClassifications(output_name, classifications):
    directory = os.path.join('data', 'classification')
    util.file.mkdirs(directory)

    filename = os.path.join(directory, output_name+'.csv')
    with open(filename, 'w+') as f:
        for img_path, digit1, digit2, digit3 in classifications:
            line = ' | '.join([img_path.decode('utf-8'),
                               formatArray(digit1),
                               formatArray(digit2),
                               formatArray(digit3)])
            f.write(line)
            f.write('\n')
    print('Wrote file', filename)


def lazyClassify(dataset, output_name=None):
    year_log, batch_paths, num_batches = classifyOp(dataset)

    if not output_name:
        output_name = str(dataset)

    return lambda sess: saveClassifications(
            output_name, classify(sess, year_log, batch_paths, num_batches))
