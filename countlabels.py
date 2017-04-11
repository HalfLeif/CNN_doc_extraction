
import loading.load_swe as swe

from collections import Counter

def printYearCounts(years):
    for y, count in Counter(years).items():
        print(y, count)


if __name__ == '__main__':
    _, years = swe.loadTrainingSet();
    printYearCounts(years)
