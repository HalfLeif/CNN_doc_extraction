
import loading.load_swe as swe

import ast
import os

def buildImageDict(filename):
    page_dict = {}
    with open(filename, 'r', newline='\n') as csvfile:
        for line in csvfile:
            fields = line.split(' | ')
            image_id = fields[0]
            years = fields[1]
            page_nbr = fields[2]
            book = fields[3:]

            page_dict[page_nbr] = (years, book, image_id)
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

def isSameBook(book1, book2):
    if not book1 or not book2:
        return False

    # All books have 3 fields: Parish, series, volume
    for field in range(3):
        if book1[field] != book2[field]:
            return False

    return True

def analyzeCollection(filename, distribution, denominator):
    print('Analyzing', filename)
    page_dict = buildImageDict(filename)

    prev_year_list = []
    prev_book = None
    for year_str, book, image_id in traverseSorted(page_dict):
        year_list = ast.literal_eval(year_str)

        if isSameBook(prev_book, book):
            addExample(distribution, prev_year_list, year_list)
            denominator += 1.0
        else:
            prev_book = book

        prev_year_list = year_list

    return distribution, denominator

if __name__ == '__main__':
    page_index_dir = os.path.join('data', 'labels_index', 'page_index')
    distribution = {}
    denominator = 0.0

    # for filename in os.listdir(page_index_dir):
    for filename in swe.swe_train_collections:
        path = os.path.join(page_index_dir, filename+'.csv')
        distribution, denominator = analyzeCollection(path, distribution, denominator)

    print(denominator)
    for key in sorted(distribution.keys()):
        print(key, distribution[key])
