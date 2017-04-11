
import os
import sys

def analyzeCollection(filename):
    print('Analyzing', filename)
    page_dict = {}
    with open(filename, 'r', newline='\n') as csvfile:
        for line in csvfile:
            fields = line.split(' | ')
            image_id = fields[0]
            years = fields[1]
            page_nbr = int(fields[2])
            book_name = ' | '.join(fields[3:])

            page_dict[page_nbr] = (years, book_name, image_id)

    for key in sorted(page_dict.keys()):
        print(key, page_dict[key])

if __name__ == '__main__':
    page_index_dir = os.path.join('data', 'labels_index', 'page_index')
    for filename in os.listdir(page_index_dir):
        path = os.path.join(page_index_dir, filename)
        analyzeCollection(path)
        break
