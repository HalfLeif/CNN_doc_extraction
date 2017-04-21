
import loading.load_image as img

import tensorflow as tf

import ast
import os

# TODO: replace with FLAGS
records_dir = "/home/leif/sweden/records"
labels_dir = "/home/leif/labels"
# labels_dir = "/Users/HalfLeif/labels"
# records_dir = "/Users/HalfLeif/sweden/records"

swe_train_collections = ['1647578', '1647598', '1647693', '1930273']
swe_eval_only = ['1930243', '1949331']

def makeQueue(batch_size, all_jpgs, all_years, shuffle=True):
    num_batches = int(len(all_years)/batch_size)

    jpgs = tf.constant(all_jpgs, tf.string)
    years = tf.constant(all_years, tf.int32)

    jpg_path, year = tf.train.slice_input_producer([jpgs, years], shuffle=shuffle, capacity=25)
    # jpg_path = tf.Print(jpg_path, ['Load swe image: ', jpg_path], summarize=100)
    # TODO: ratio 4 or 8?
    image = img.loadImage(jpg_path, ratio=8)

    batch_images, batch_years, batch_paths = tf.train.batch(
            [image, year, jpg_path], batch_size=batch_size,
            capacity=4, num_threads=2,
            dynamic_pad=True, allow_smaller_final_batch=True)

    # Need to invert images after the dynamic padding.
    batch_images = 1 - batch_images

    # batch_images = tf.Print(batch_images, ['DEBUG', debug.debugFirstImage(batch_images, 'SWE')])

    print('Swe queue created')
    return batch_images, batch_years, batch_paths, num_batches

def sweBatch(batch_size, train_mode):
    print('Load transcriptions')
    if train_mode:
        all_jpgs, all_years = loadTrainingSet()
    else:
        all_jpgs, all_years = loadTestSet()

    return makeQueue(batch_size, all_jpgs, all_years, shuffle=True)


def classificationBatch(batch_size, collection):
    train_imgs, train_years = loadCollection(collection, train=True)
    test_imgs, test_years = loadCollection(collection, train=False)
    all_imgs = train_imgs + test_imgs
    all_years = train_years + test_years

    return makeQueue(batch_size, all_imgs, all_years, shuffle=False)

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
            image_id = fields[0]
            year_str = fields[1]
            year_list = ast.literal_eval(year_str)

            # Skip images which have no year:
            if not year_list:
                continue

            image_path = os.path.join(records_dir, collection_name, 'Images', image_id + '.jpg')
            if os.path.exists(image_path):
                image_files.append(image_path)
                pair = (min(year_list), max(year_list))
                years.append(pair)
            else:
                missing_files += 1
                missing_example = image_id
    if missing_files > 0:
        print('Collection', collection_name, 'is missing', missing_files, 'images, for example', missing_example)
    return image_files, years


def buildImageDict(filename):
    ''' Opens collection label file and builds dictionary of
        sort_value to (label, book_id, image_id).'''
    page_dict = {}
    with open(filename, 'r', newline='\n') as csvfile:
        for line in csvfile:
            fields = line.split(' | ')
            image_id = fields[0]
            years = fields[1]
            sort_value = fields[2]
            book_id = '-'.join(fields[3:]).rstrip('\n')

            page_dict[sort_value] = (years, book_id, image_id)
    return page_dict


if __name__ == '__main__':
    imgs, years = loadTrainingSet()
    print('MIN', min(years))
    print('MAX', max(years))
    print('#Labels', len(years))
