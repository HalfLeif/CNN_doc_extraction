
import ast
import os

# TODO: replace with FLAGS
records_dir = "/home/leif/sweden/records"
labels_dir = "/home/leif/labels"
# labels_dir = "/Users/HalfLeif/labels"
# records_dir = "/Users/HalfLeif/sweden/records"

swe_train_collections = ['1647578', '1647598', '1647693', '1930273']
swe_eval_only = ['1930243', '1949331']

def loadTrainingSet():
    imgs = []
    years = []
    for collection in swe_train_collections:
        coll_imgs, coll_years = loadCollection(collection, train=True)
        imgs += coll_imgs
        years += coll_years
    return imgs, years

def loadTestSet():
    imgs = []
    years = []
    for collection in swe_train_collections:
        coll_imgs, coll_years = loadCollection(collection, train=False)
        imgs += coll_imgs
        years += coll_years
    return imgs, years

def loadEvalSet():
    imgs = []
    years = []
    for collection in swe_eval_only:
        train_imgs, train_years = loadCollection(collection, train=True)
        test_imgs, test_years = loadCollection(collection, train=False)
        imgs += train_imgs + test_imgs
        years += train_years + test_years
    return imgs, years

def loadCollection(collection_name, train=True):
    image_files = []
    years = []
    if train:
        subdir = 'train'
    else:
        subdir = 'test'
    labels_file = os.path.join(labels_dir, subdir, collection_name + '.csv')

    if not os.path.exists(labels_file):
        print('Could not find file with labels: ', labels_file)
        return [], []

    missing_files = 0
    missing_example = ''
    with open(labels_file, 'r', newline='\n') as csvfile:
        for line in csvfile:
            fields = line.split(' | ')
            image_name = fields[0]
            year_str = fields[1]
            year_list = ast.literal_eval(year_str)

            # Skip images which have no year:
            if not year_list:
                continue

            image_path = os.path.join(records_dir, collection_name, 'Images', image_name + '.jpg')
            if os.path.exists(image_path):
                image_files.append(image_path)
                years.append(max(year_list))
            else:
                missing_files += 1
                missing_example = image_name
    if missing_files > 0:
        print('Collection', collection_name, 'is missing', missing_files, 'images, for example', missing_example)
    return image_files, years

if __name__ == '__main__':
    print(loadTestSet())
