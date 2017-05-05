
import loading.load_swe as swe

import gflags

from collections import Counter
import sys

def printYearCounts(years):
    for y, count in Counter(years).items():
        print(y, count)


if __name__ == '__main__':
    gflags.FLAGS(sys.argv)
    _, years = swe.loadTrainingSet();
    printYearCounts(years)
