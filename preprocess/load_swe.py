
import ast
# import csv
import os

# TODO: replace with FLAGS
records_dir = "/home/leif/sweden/records"
labels_dir = "/home/leif/labels"
# labels_dir = "/Users/HalfLeif/labels"
# records_dir = "/Users/HalfLeif/sweden/records"


def loadCollection(collection_name):
    image_files = []
    years = []
    labels_file = os.path.join(labels_dir, collection_name + '.csv')
    with open(labels_file, 'r', newline='\n') as csvfile:
        for line in csvfile:
            fields = line.split(', ', maxsplit=1)
            image_name = fields[0]
            year_str = fields[1]
            year_list = ast.literal_eval(year_str)

            # Skip images which have no year:
            if not year_list:
                continue

            image_path = os.path.join(records_dir, collection_name, 'Images', image_name + '.jpg')
            image_files.append(image_path)
            years.append(max(year_list))
    return image_files, years

if __name__ == '__main__':
    # print(loadCollection("1930273"))
    print(loadCollection("1647578"))
