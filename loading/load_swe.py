
import loading.load_image as img

import gflags
import tensorflow as tf

import ast
import os


gflags.DEFINE_integer('SWE_BATCH_SIZE', 10, 'Number of training examples for MNIST per batch.', lower_bound=2)

gflags.DEFINE_string('records_dir', '/home/leif/sweden/records',
        'Directory containing the SWE image collections.')

gflags.DEFINE_string('labels_dir', '/home/leif/labels',
        'Directory containing the SWE label csvs.')


swe_train_collections = ['1647578', '1647598', '1647693', '1930273']
swe_eval_only = ['1930243', '1949331']


def makeQueue(all_jpgs, all_years, shuffle=True):
    batch_size = gflags.FLAGS.SWE_BATCH_SIZE
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


def sweBatch(dataset, shuffle=True):
    all_jpgs, all_years = loadDataset(dataset)
    return makeQueue(all_jpgs, all_years, shuffle=shuffle)


def loadDataset(dataset):
    print('Load transcriptions for', dataset)
    if dataset == 'train':
        return loadTrainingSet()
    elif dataset == 'test':
        return loadTestSet()
    elif dataset == 'eval':
        return loadEvalSet()
    else:
        # List of collections, load both test and train data for each.
        return loadTrainTest(dataset)


def loadTrainingSet():
    return loadCollections(swe_train_collections, train=True)


def loadTestSet():
    return loadCollections(swe_train_collections, train=False)


def loadEvalSet():
    return loadTrainTest(swe_eval_only)


def loadTrainTest(collection_names):
    tr_jpgs, tr_years = loadCollections(collection_names, train=True)
    ts_jpgs, ts_years = loadCollections(collection_names, train=False)
    return tr_jpgs + ts_jpgs, tr_years + ts_years


def loadCollections(collection_names, train):
    imgs = []
    years = []
    for collection in collection_names:
        coll_imgs, coll_years = loadCollection(collection, train=train)
        imgs += coll_imgs
        years += coll_years
    return imgs, years


def loadCollection(collection_name, train=True):
    image_files = []
    years = []
    if train:
        subdir = 'train'
    else:
        subdir = 'test'

    labels_file = os.path.join(gflags.FLAGS.labels_dir,
            subdir, collection_name + '.csv')

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

            image_path = os.path.join(gflags.FLAGS.records_dir,
                    collection_name, 'Images', image_id + '.jpg')
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
