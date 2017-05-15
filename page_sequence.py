import loading.load_swe as swe
import postprocess.books as books
import postprocess.classifications as cs
import postprocess.jump_distribution as jd
import postprocess.cond_jump_distribution as cjd
import postprocess.optimize_book as ob
import util.dict_util as du

import os
import statistics as st


classification_dir = os.path.join('data', 'classification')


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

def listThreshold(diffs, threshold):
    ''' Returns the fraction of elements that are below the threshold.'''
    below = len([e for e in diffs if e < threshold])
    return float(below) / len(diffs)

def printStats(diffs):
    print('# Prec', listAcc(diffs))
    print('# Med ', st.median(diffs))
    print('# Mean', st.mean(diffs))
    print('# Below', listThreshold(diffs, 5))

def filterList(elems, confs, threshold):
    above = []
    for i in range(len(elems)):
        if confs[i] > threshold:
            above.append(elems[i])
    return above

def printBooks(collections, predictions_filename, debug=True):
    ''' Prints classifications in order of the books.
        Also prints some overall metric stats.
    '''
    organized = books.organizeToBooks(collections)

    cs_file = os.path.join(classification_dir, predictions_filename)
    predictions = cs.buildPredictionDict(cs_file)

    diffs = []
    confs = []
    for book_id, page_seq in organized:
        if debug:
            print(book_id, len(page_seq))

        for image_id, year_labels in page_seq:
            if image_id in predictions:
                pred_year, conf = predictions[image_id]
                diffs.append(yearDiff(pred_year, year_labels))
                confs.append(conf)
            else:
                pred_year = '____'
                conf = ''

            if debug:
                print('\t', image_id, pred_year, conf, year_labels)

    print('# Predictions from', predictions_filename)

    num_thresholds = 5
    max_threshold = 0.4
    for i in range(num_thresholds):
        threshold = i * max_threshold / (num_thresholds - 1)
        print('# Confidence threshold:', threshold)
        filtered = filterList(diffs, confs, threshold)
        if filtered:
            printStats(filtered)
        print('Cov ', float(len(filtered))/len(diffs))
    # printStats(diffs)


def optimizeBooks(collections, predictions_filename):
    ''' Prints classifications and post-processing in order of the books.
        Also prints some overall metric stats.
    '''
    organized = books.organizeToBooks(collections)

    cs_file = os.path.join(classification_dir, predictions_filename)
    logits_dict = cs.buildLogitsDict(cs_file)

    # Conditional jump distribution:
    # jump_file = os.path.join('data', 'cond_jumps_thresh.csv')
    # distr = cjd.loadDistribution(jump_file)

    # Unconditional jump distribution:
    jump_file = os.path.join('data', 'jump_distribution.csv')
    distr = jd.loadDistribution(jump_file)

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
    # printBooks(swe.swe_eval_only, 'eval_Swe_DEP3_ind_digits6-3417.csv', False)
    # printBooks(swe.swe_train_collections, 'test_Swe_DEP3_ind_digits6-3417.csv', False)
    # optimizeBooks(swe.swe_eval_only, 'eval_SweDEP4_multiyear_4-3417.csv')
    optimizeBooks(['1647578'], '1647578.csv')
