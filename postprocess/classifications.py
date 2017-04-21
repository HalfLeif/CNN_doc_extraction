import numpy as np

import re

def parseArray(arr_str):
    elems = arr_str.split(',')

    elems[0] = elems[0].lstrip(' [')
    last = len(elems) - 1
    elems[last] = elems[last].rstrip(']\n ')

    numbers = []
    for elem in elems:
        if elem:
            numbers.append(float(elem))
    return np.array(numbers)


def loadClassifications(filename):
    with open(filename, 'r') as read_file:
        pattern = re.compile('([A-Z0-9-]*)\\.jpg')
        for line in read_file:
            [img_path, logits_1, logits_2, logits_3] = line.split(' | ')
            img_id = pattern.search(img_path).group(1)
            yield (img_id,
                parseArray(logits_1),
                parseArray(logits_2),
                parseArray(logits_3))

def prediction(logits_1, logits_2, logits_3):
    digits = [np.argmax(logits) for logits in [logits_1, logits_2, logits_3]]
    return 1000 + np.sum(np.multiply(digits, [100, 10, 1]))

def buildPredictionDict(filename):
    predictions = {}
    for img_id, d1, d2, d3 in loadClassifications(filename):
        pred = prediction(d1, d2, d3)
        predictions[img_id] = pred
    return predictions
