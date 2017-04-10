
import loading.load_image as img

import tensorflow as tf

import ast
import os

# TODO: replace with FLAGS

root_dir = 'D:\\Data'
# root_dir = '/home/leif'
# root_dir = '/Users/HalfLeif'

records_dir = os.path.join(root_dir, 'sweden', 'records')
labels_dir = os.path.join(root_dir, 'labels')

swe_train_collections = ['1647578', '1647598', '1647693', '1930273']
swe_eval_only = ['1930243', '1949331']


def sweBatch(batch_size, train_mode):
    print('Load transcriptions')
    if train_mode:
        all_jpgs, all_years = loadTrainingSet()
    else:
        all_jpgs, all_years = loadTestSet()

    num_batches = int(len(all_years)/batch_size)

    jpgs = tf.constant(all_jpgs, tf.string)
    years = tf.constant(all_years, tf.int32)

    jpg_path, year = tf.train.slice_input_producer([jpgs, years], shuffle=True, capacity=25)
    # jpg_path = tf.Print(jpg_path, ['Load swe image: ', jpg_path], summarize=100)
    # TODO: ratio 4 or 8?
    image = img.loadImage(jpg_path, ratio=8)

    batch_images, batch_years = tf.train.batch([image, year], batch_size=batch_size, capacity=4, num_threads=2, dynamic_pad=True)

    # Need to invert images after the dynamic padding.
    batch_images = 1 - batch_images

    # batch_images = tf.Print(batch_images, ['DEBUG', debug.debugFirstImage(batch_images, 'SWE')])

    print('Swe queue created')
    return batch_images, batch_years, num_batches


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
    imgs, years = loadTrainingSet()
    print('MIN', min(years))
    print('MAX', max(years))
    print('#Labels', len(years))
