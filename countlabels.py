
import loading.load_swe as swe

from collections import Counter

if __name__ == '__main__':
    _, years = swe.loadTrainingSet();
    for y, count in Counter(years).items():
        print(y, count)
