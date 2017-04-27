import postprocess.books as books
import postprocess.classifications as cs
import postprocess.jump_distribution as jp
import postprocess.optimize_book as ob
import util.dict_util as du

import os
import statistics as st


def printBooks(collection):
    page_index_dir = os.path.join('data', 'labels_index', 'page_index')
    filename = os.path.join(page_index_dir, collection+'.csv')
    organized = books.organizeToBooks(filename)

    cs_file = os.path.join('data', 'classification', collection+'.csv')
    predictions = cs.buildPredictionDict(cs_file)

    for book_id, page_seq in organized:
        print(book_id, len(page_seq))
        for image_id, year_seq in page_seq:
            if image_id in predictions:
                print('\t', image_id, predictions[image_id], year_seq)
            else:
                print('\t', image_id, '____', year_seq)

def yearDiff(year, year_list):
    low = min(year_list)
    high = max(year_list)
    if year < low:
        return low - year
    elif year > high:
        return year - high
    else:
        return 0

def listAcc(diffs):
    return float(diffs.count(0)) / len(diffs)

def listMedian(diffs):
    return st.median(diffs)

def printStats(diffs):
    print('Acc', listAcc(diffs))
    print('Med', listMedian(diffs))

def optimizeBooks(collection):
    page_index_dir = os.path.join('data', 'labels_index', 'page_index')
    filename = os.path.join(page_index_dir, collection+'.csv')
    organized = books.organizeToBooks(filename)

    cs_file = os.path.join('data', 'classification', collection+'.csv')
    logits_dict = cs.buildLogitsDict(cs_file)

    jump_file = os.path.join('data', 'jump_distribution.csv')
    distr = jp.buildObj(1899 - 1690 + 1, laplace=0.5, filename=jump_file)

    orig_diffs = []
    opt_diffs = []

    for book_id, page_seq in organized:
        sequence, original = ob.optimizeBook(page_seq, distr, logits_dict)
        i = 0
        print(book_id, len(page_seq))
        for image_id, year_labels in page_seq:
            if image_id in logits_dict:
                print('\t', image_id, original[i], sequence[i], year_labels)
                orig_diffs.append(yearDiff(original[i], year_labels))
                opt_diffs.append(yearDiff(sequence[i], year_labels))
                i += 1
            else:
                print('\t', image_id, '____', '____', year_labels)

    print('# Original predictions')
    printStats(orig_diffs)

    print('# Post-processed')
    printStats(opt_diffs)

if __name__ == '__main__':
    # printBooks('1647578')
    optimizeBooks('1647578')
    # cs_file = os.path.join('data', 'classification', '1647578.csv')
    # ls = cs.buildLogitsDict(cs_file)
    # du.printDict(ls)
