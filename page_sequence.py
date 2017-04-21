
import loading.load_swe as swe

import numpy as np

import ast
import os
import re


def buildImageDict(filename):
    page_dict = {}
    with open(filename, 'r', newline='\n') as csvfile:
        for line in csvfile:
            fields = line.split(' | ')
            image_id = fields[0]
            years = fields[1]
            sort_value = fields[2]
            book_id = '-'.join(fields[3:]).rstrip('\n')

            page_dict[sort_value] = (years, book_id, image_id)
    return page_dict

def traverseSorted(dictionary):
    for key in sorted(dictionary.keys()):
        yield dictionary[key]

def increment(distribution, key, incr):
    if key in distribution:
        distribution[key] += incr
    else:
        distribution[key] = incr

def addExample(distribution, prev, current):
    num_combinations = len(prev) * len(current)
    for y1 in prev:
        for y2 in current:
            diff = y2 - y1
            increment(distribution, diff, 1.0/num_combinations)

def isSameBook(book_id1, book_id2):
    return book_id1 == book_id2

def analyzeCollection(filename, distribution, denominator):
    print('# Analyzing', filename)
    page_dict = buildImageDict(filename)

    prev_year_list = []
    prev_book_id = None
    for year_str, book_id, image_id in traverseSorted(page_dict):
        year_list = ast.literal_eval(year_str)

        if isSameBook(prev_book_id, book_id):
            addExample(distribution, prev_year_list, year_list)
            denominator += 1.0
        else:
            prev_book_id = book_id

        prev_year_list = year_list

    return distribution, denominator

def printDistribution():
    page_index_dir = os.path.join('data', 'labels_index', 'page_index')
    distribution = {}
    denominator = 0.0

    # for filename in os.listdir(page_index_dir):
    for filename in swe.swe_train_collections:
    # for filename in ['1647578']:
        path = os.path.join(page_index_dir, filename+'.csv')
        distribution, denominator = analyzeCollection(path, distribution, denominator)

    print(denominator)
    for key in sorted(distribution.keys()):
        print(key, distribution[key])

def organizeToBooks(filename):
    print('# Organizing', filename)
    page_dict = buildImageDict(filename)

    organized_books = []
    page_sequence = []
    prev_book_id = None
    for year_str, book_id, image_id in traverseSorted(page_dict):
        year_list = ast.literal_eval(year_str)

        if isSameBook(prev_book_id, book_id):
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


    # numbers = ast.literal_eval(arr_str)

    try:
        for elem in elems:
            if elem:
                numbers.append(float(elem))
        return np.array(numbers)
    except:
        print(arr_str)
        print(elems)
        raise


def loadClassifications(collection):
    filename = os.path.join('data', 'classification', collection+'.csv')
    with open(filename, 'r') as read_file:
        pattern = re.compile('([A-Z0-9-]*)\\.jpg')
        for line in read_file:
            try:
                [img_path, logits_1, logits_2, logits_3] = line.split(' | ')
                img_id = pattern.search(img_path).group(1)
                yield (img_id,
                    parseArray(logits_1),
                    parseArray(logits_2),
                    parseArray(logits_3))
            except ValueError:
                print(line)
                # print(logits_1)
                # print(logits_2)
                # print(logits_3)
                raise

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
    printBooks()
