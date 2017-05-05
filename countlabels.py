
import loading.load_swe as swe
import postprocess.books as books
import util.dict_util as du

import gflags

# from collections import Counter
import sys

# def printYearCounts(years):
    # for y, count in Counter(years).items():
        # print(y, count)

def countYears(collections):
    organized = books.organizeToBooks(collections)
    counts = {}
    for book_id, page_seq in organized:
        for _, year_list in page_seq:
            for year in year_list:
                du.increment(counts, year, 1)
    return counts

if __name__ == '__main__':
    gflags.FLAGS(sys.argv)
    counts = countYears(swe.swe_train_collections)
    du.printDictSorted(counts)
