
import numpy as np

min_label = 1650
max_label = 1899
num_labels = max_label - min_label + 1


def softmax(x):
    ''' Computes the softmax value for each label score in x.
        We subtract max(x) for numerical stability
        but it does not affect the mathematical result, see:
        http://stackoverflow.com/questions/34968722/softmax-function-python
    '''
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def expandLogits(logits):
    digit_probabilities = [softmax(dlog) for dlog in logits]
    return digit_probabilities


def yearToDigits(year):
    q, digit3 = divmod(year, 10)
    q, digit2 = divmod(q, 10)
    _, digit1 = divmod(q, 10)
    return (digit1, digit2, digit3)


def labelProbabilities(digit_probabilities):
    ''' Returns unnormalized label probabilities for min_label to max_label
        as a numpy array.'''
    label_probabilities = np.zeros(num_labels)
    for index in range(num_labels):
        label_digits = yearToDigits(min_label + index)
        product = 1.0
        for digit_pos in range(3):
            digit = label_digits[digit_pos]
            product *= digit_probabilities[digit_pos][digit]
        label_probabilities[index] = product
    return label_probabilities


def maximizeStep(max_probabilities_arr, jump_distribution, to_index):
    best_index = 0
    highest_prob = -1
    for from_index in range(num_labels):
        jump_prob = jump_distribution.probability(to_index - from_index)
        prob = jump_prob * max_probabilities_arr[from_index]
        if prob > highest_prob:
            highest_prob = prob
            best_index = from_index

    return highest_prob, best_index

def backtrack(backtrack_matrix, last_label):
    ''' Returns the sequence indicated by the backtracking.'''
    sequence = []
    for backtrack_arr in reversed(backtrack_matrix):
        sequence.append(last_label)
        last_label = backtrack_arr[last_label]
    sequence.reverse()
    return sequence

def toYear(indices):
    return [min_label + index for index in indices]

def optimizeBook(page_seq, jump_distribution, logits_dict):
    ''' Calculates maximum probability path using the Viterbi algorithm
        over this sequence of pages.
        page_seq: sequence of (image_id, label)
        jump_distribution: dictionary of jump distance to probability
        logits: dictionary of image_id to three logits, one for each number
    '''
    original_predictions = []
    backtrack_matrix = []
    # max_probabilities_matrix = []

    for image_id, label in page_seq:
        if not image_id in logits_dict:
            # TODO handle better?
            continue

        logits = logits_dict[image_id]
        digit_probabilities = expandLogits(logits)
        label_probabilities = labelProbabilities(digit_probabilities)
        original_predictions.append(np.argmax(label_probabilities))
        # TODO do I need to normalize probabilities?
        # argmax should work correct anyway?

        backtrack_arr = np.zeros(num_labels, dtype=np.int32)

        if backtrack_matrix:
            max_probabilities_arr = np.zeros(num_labels)

            for to_index in range(num_labels):
                prob, from_index = maximizeStep(prev_max_arr, jump_distribution, to_index)
                max_probabilities_arr[to_index] = prob
                backtrack_arr[to_index] = from_index
        else:
            # First iteration!
            max_probabilities_arr = np.ones(num_labels)

        max_probabilities_arr = max_probabilities_arr * label_probabilities

        prev_max_arr = max_probabilities_arr
        backtrack_matrix.append(backtrack_arr)
        # max_probabilities_matrix.append(max_probabilities_arr)

    last_label = np.argmax(max_probabilities_arr)
    sequence = backtrack(backtrack_matrix, last_label)
    return toYear(sequence), toYear(original_predictions)
