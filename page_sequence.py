
import loading.load_swe as swe
import postprocess.jump_distribution as jp
import util.dict_util as du

import numpy as np

import ast
import os
import re


def organizeToBooks(filename):
    print('# Organizing', filename)
    page_dict = swe.buildImageDict(filename)

    organized_books = []
    page_sequence = []
    prev_book_id = None
    for year_str, book_id, image_id in du.traverseSorted(page_dict):
        year_list = ast.literal_eval(year_str)

        if prev_book_id == book_id:
            page = (image_id, year_list)
            page_sequence.append(page)
        else:
            if page_sequence:
                yield prev_book_id, page_sequence
            page_sequence = []
            prev_book_id = book_id

    return organized_books


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


def loadClassifications(collection):
    filename = os.path.join('data', 'classification', collection+'.csv')
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

def buildPredictionDict():
    predictions = {}
    for img_id, d1, d2, d3 in loadClassifications('1647578'):
        pred = prediction(d1, d2, d3)
        predictions[img_id] = pred
    return predictions

def printBooks():
    page_index_dir = os.path.join('data', 'labels_index', 'page_index')
    filename = os.path.join(page_index_dir, '1647578.csv')
    organized = organizeToBooks(filename)

    predictions = buildPredictionDict()

    for book_id, page_seq in organized:
        print(book_id, len(page_seq))
        for image_id, year_seq in page_seq:
            if image_id in predictions:
                print('\t',image_id, predictions[image_id], year_seq)


if __name__ == '__main__':
    # printBooks()
    distribution, denominator = jp.loadDistribution(os.path.join('data', 'jump_distribution.csv'))
    print(denominator)
    print(distribution)
