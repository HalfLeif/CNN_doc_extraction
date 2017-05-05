
import loading.load_swe as swe
import postprocess.books as books
import util.dict_util as du

import gflags

import sys


def countYears(collections):
    organized = books.organizeToBooks(collections)
    counts = {}
    for book_id, page_seq in organized:
        for _, year_list in page_seq:
            for year in year_list:
                du.increment(counts, year, 1.0/len(year_list))
    return counts

if __name__ == '__main__':
    gflags.FLAGS(sys.argv)
    # counts = countYears(swe.swe_eval_only)
    counts = countYears(swe.swe_train_collections)
    print('# Average:', sum(counts.values()) / len(counts) )
    du.printDictSorted(counts)
