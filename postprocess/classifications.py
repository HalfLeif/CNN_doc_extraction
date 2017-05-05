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


def softmax(x):
    ''' Computes the softmax value for each label score in x.
        We subtract max(x) for numerical stability
        but it does not affect the mathematical result, see:
        http://stackoverflow.com/questions/34968722/softmax-function-python
    '''
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def expandLogits(logits_list):
    digit_probabilities = [softmax(dlog) for dlog in logits_list]
    return digit_probabilities

def confidence(digits, digit_probabilities):
    product = 1.0
    for i in range(3):
        digit = digits[i]
        product *= digit_probabilities[i][digit]
    return product

def prediction(logits_list):
    digits = [np.argmax(logits) for logits in logits_list]
    pred_year = 1000 + np.sum(np.multiply(digits, [100, 10, 1]))

    probabilities = expandLogits(logits_list)
    conf = confidence(digits, probabilities)
    return pred_year, conf

def buildPredictionDict(filename):
    predictions = {}
    for img_id, d1, d2, d3 in loadClassifications(filename):
        predictions[img_id] = prediction([d1, d2, d3])
    return predictions

def buildLogitsDict(filename):
    logits = {}
    for img_id, *digits in loadClassifications(filename):
        logits[img_id] = digits
    return logits
