
import loading.load_swe as swe

from collections import Counter

def printYearCounts(years):
    for y, count in Counter(years).items():
        print(y, count)

def isConsecutive(yearlist):
    last = None
    for year in yearlist:
        if last and last != year - 1:
            return False
        last = year
    return True

def verifyConsecutive(years):
    errors = 0
    for yearlist in years:
        if not isConsecutive(yearlist):
            print(yearlist)
            errors = errors + 1
    print('There are', errors, 'inconsecutive year labels out of', len(years))

if __name__ == '__main__':
    _, years = swe.loadTrainingSet();
    verifyConsecutive(years)
    _, years = swe.loadTestSet();
    verifyConsecutive(years)
    _, years = swe.loadEvalSet();
    verifyConsecutive(years)
