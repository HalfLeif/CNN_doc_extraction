
import preprocess.grayscale as gray

import tensorflow as tf

def loadImage(jpg_path, ratio):
    image_content = tf.read_file(jpg_path)
    x = tf.image.decode_jpeg(image_content, channels=1, ratio=ratio, name='image')
    x = tf.cast(x, tf.float32)
    x = gray.simpleRemap(x)
    return x
