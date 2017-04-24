import postprocess.books as books
import postprocess.classifications as cs
import postprocess.jump_distribution as jp
import postprocess.optimize_book as ob
import util.dict_util as du

import os


def printBooks(collection):
    page_index_dir = os.path.join('data', 'labels_index', 'page_index')
    filename = os.path.join(page_index_dir, collection+'.csv')
    organized = books.organizeToBooks(filename)

    cs_file = os.path.join('data', 'classification', collection+'.csv')
    predictions = cs.buildPredictionDict(cs_file)

    for book_id, page_seq in organized:
        print(book_id, len(page_seq))
        for image_id, year_seq in page_seq:
            if image_id in predictions:
                print('\t',image_id, predictions[image_id], year_seq)


def optimizeBooks(collection):
    page_index_dir = os.path.join('data', 'labels_index', 'page_index')
    filename = os.path.join(page_index_dir, collection+'.csv')
    organized = books.organizeToBooks(filename)

    cs_file = os.path.join('data', 'classification', collection+'.csv')
    logits_dict = cs.buildLogitsDict(cs_file)

    jump_file = os.path.join('data', 'jump_distribution.csv')
    distr = jp.buildObj(1899 - 1690 + 1, laplace=0.5, filename=jump_file)

    for book_id, page_seq in organized:
        sequence, original = ob.optimizeBook(page_seq, distr, logits_dict)
        i = 0
        print(book_id, len(page_seq))
        for image_id, year_labels in page_seq:
            if image_id in logits_dict:
                print('\t',image_id, original[i], sequence[i], year_labels)
                i += 1

if __name__ == '__main__':
    # printBooks('1647578')
    optimizeBooks('1647578')
    # cs_file = os.path.join('data', 'classification', '1647578.csv')
    # ls = cs.buildLogitsDict(cs_file)
    # du.printDict(ls)
