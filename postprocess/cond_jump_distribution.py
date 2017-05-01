import loading.load_swe as swe
import util.dict_util as du

import ast
import os

def jumpList(from_years, to_years):
    ''' Counts how many times each jump size is performed and
        then normalizes the counts.
    '''
    jump_list = {}
    denominator = len(from_years) * len(to_years)
    for i in from_years:
        for j in to_years:
            key = j-i
            du.increment(jump_list, key, 1.0/denominator)
    return jump_list

class JumpsAccumulator:
    def __init__(self):
        ''' Aggregates counts of jumps in sequences with memory of one step.

            Maintains 4 types of counts:
                'total': total count
                diff:    count of all jumps with length `diff`
                ('total', prev): count of all jumps followed by the jump `prev`
                (diff, prev):    count of jumps `diff` followed by `prev`
        '''
        self.counts = {}
        self.newBook()

    def newBook(self):
        self.prev_years = []
        self.prev_jump = {}

    def feedYearList(self, year_list):
        jump_list = jumpList(self.prev_years, year_list)
        self.addJump(jump_list)
        self.prev_years = year_list

    def addJump(self, jump_list):
        ''' Updates all four types of counts.
        '''
        if jump_list:
            du.increment(self.counts, 'total', 1.0)

            for j in self.prev_jump:
                du.increment(self.counts, ('total', j), self.prev_jump[j])

            for k in jump_list:
                du.increment(self.counts, k, jump_list[k])

            for j in self.prev_jump:
                for k in jump_list:
                    incr = self.prev_jump[j] * jump_list[k]
                    du.increment(self.counts, (k, j), incr)

        self.prev_jump = jump_list

threshold = 0.3

class ConditionalJumpDistribution:
    ''' Similar to JumpDistribution with the extension that it also
        handles probabilities conditioned on the previous jump size.

        However, decreases accuracy in practice due to that with our
        statistics, P(0|0) > P(0). So the model is even more likely to get
        stuck at some year for the entire book.
    '''
    def __init__(self, counts):
        self.counts = {}
        for key in counts:
            if counts[key] > threshold:
                # Save memory by ignoring very small counts.
                # Reduces the number of cache misses.
                self.counts[key] = counts[key]
        self.setSmoothing(0.0, 0.0)

    def setSmoothing(self, laplace, n_output_values):
        if laplace and not n_output_values:
            raise ValueError('Error! If uses Laplace smoothing, then must specify how many valid output values there are.')

        self.laplace = laplace
        self.n_output_values = n_output_values

    def condProb(self, curr, prev):
        ''' P(k|j) = N(k, j) / N(*, j)
            but with laplace smoothing.
        '''
        nominator = self.counts.get( (curr, prev), 0.0)
        denominator = self.counts.get( ('total', prev), 0.0)
        return (nominator + self.laplace) / (
                denominator + self.laplace * self.n_output_values)

    def marginalProb(self, curr):
        ''' P(k) = sum_j P(k,j) = N(k,*) / N(*,*)
            For each j, N(k,j) add `laplace` once,
            so N(k,*) should add `laplace` * `n_output_values`.
        '''
        nominator = self.counts.get(curr, 0.0)
        denominator = self.counts.get('total', 0.0)
        return (nominator + self.laplace * self.n_output_values) / (
                denominator + self.laplace * self.n_output_values * self.n_output_values)

    def printSelf(self):
        for key in self.counts.keys():
            print(repr(key), ':', self.counts[key])


def addCollection(filename, distribution):
    print('# Analyzing', filename)
    page_dict = swe.buildImageDict(filename)

    prev_book_id = None
    for year_str, book_id, image_id in du.traverseSorted(page_dict):
        year_list = ast.literal_eval(year_str)

        if prev_book_id == book_id:
            distribution.feedYearList(year_list)
        else:
            distribution.newBook()
            prev_book_id = book_id

    return distribution

def buildDistribution():
    page_index_dir = os.path.join('data', 'labels_index', 'page_index')
    accumulator = JumpsAccumulator()

    for filename in swe.swe_train_collections:
        path = os.path.join(page_index_dir, filename+'.csv')
        addCollection(path, accumulator)

    return ConditionalJumpDistribution(accumulator.counts)

def loadDistribution(filename):
    counts = {}
    for line in open(filename, 'r'):
        if line.startswith('#'):
            continue
        [key_str, value_str] = line.split(':')
        try:
            key = ast.literal_eval(key_str)
        except:
            print(key_str)
            raise
        counts[key] = float(value_str)
    return ConditionalJumpDistribution(counts)
