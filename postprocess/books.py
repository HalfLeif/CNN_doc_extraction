import loading.load_swe as swe
import util.dict_util as du

import ast

def organizeToBooks(filename):
    ''' Organizes labels into a list of books.
        Each book consists of (book_id, page_list)
        where each page = (image_id, year_list).
    '''
    print('# Organizing', filename)
    page_dict = swe.buildImageDict(filename)

    organized_books = []
    page_sequence = []
    prev_book_id = None
    for year_str, book_id, image_id in du.traverseSorted(page_dict):
        year_list = ast.literal_eval(year_str)

        if prev_book_id == book_id:
            page = (image_id, year_list)
            page_sequence.append(page)
        else:
            if page_sequence:
                yield prev_book_id, page_sequence
            page_sequence = []
            prev_book_id = book_id

    return organized_books
