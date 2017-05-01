import loading.load_swe as swe
import util.dict_util as du

import ast
import os

class JumpDistribution:
    ''' Deprecated in favor of ConditionalJumpDistribution.
    '''
    def __init__(self, distribution, denominator):
        self.distribution = distribution
        self.denominator = denominator
        self.year_range = 0
        self.laplace = 0.0

    def setSmoothing(self, laplace, n_output_values):
        if laplace and not n_output_values:
            raise ValueError('Error! If uses Laplace smoothing, then must specify how many valid output values there are.')

        self.laplace = laplace
        self.year_range = n_output_values

    def probability(self, diff):
        ''' Returns the unnormalized probability of doing a jump with this
            year difference between two consecutive pages.

            It is unnormalized because part of the distribution is truncated to
            the valid range. However, because the distribution is very sharp
            around 0, the effect should be negligible except for the very edges
            of the valid range.

            Uses laplace smoothing for the probabilities.
        '''
        count = self.distribution.get(diff, 0)
        return (count + self.laplace) / (
                self.denominator + self.laplace * self.year_range)

    def printSelf(self):
        print(self.denominator)
        for key in sorted(self.distribution.keys()):
            print(key, self.distribution[key])


def addExample(distribution, prev, current):
    num_combinations = len(prev) * len(current)
    for y1 in prev:
        for y2 in current:
            diff = y2 - y1
            du.increment(distribution, diff, 1.0/num_combinations)


def addCollection(filename, distribution, denominator):
    print('# Analyzing', filename)
    page_dict = swe.buildImageDict(filename)

    prev_year_list = []
    prev_book_id = None
    for year_str, book_id, image_id in du.traverseSorted(page_dict):
        year_list = ast.literal_eval(year_str)

        if prev_book_id == book_id:
            addExample(distribution, prev_year_list, year_list)
            denominator += 1.0
        else:
            prev_book_id = book_id

        prev_year_list = year_list

    return distribution, denominator


def buildDistribution():
    page_index_dir = os.path.join('data', 'labels_index', 'page_index')
    distribution = {}
    denominator = 0.0

    for filename in swe.swe_train_collections:
        path = os.path.join(page_index_dir, filename+'.csv')
        distribution, denominator = addCollection(path, distribution, denominator)

    return JumpDistribution(distribution, denominator)


def loadDistribution(filename):
    denominator = None
    distribution = {}
    for line in open(filename, 'r'):
        if line.startswith('#'):
            continue
        if denominator == None:
            denominator = float(line)
        else:
            [diff, count] = line.split(' ')
            distribution[int(diff)] = float(count)
    return JumpDistribution(distribution, denominator)
