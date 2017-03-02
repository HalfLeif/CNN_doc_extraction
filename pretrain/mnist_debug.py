import tensorflow as tf

def py_WriteImage(re_encoded, label):
    filename ='data\\temp_imgs\\temp_'+str(label)+'.jpg'
    with open(filename, 'wb+') as f:
        f.write(re_encoded)
    print('WROTE RE-ENCODED IMAGE to ', filename)
    return 0

def debugImage(image, label):
    image = tf.squeeze(image)
    image = tf.expand_dims(image, -1)
    restore = tf.cast(image*255, tf.uint8)
    re_encoded = tf.image.encode_jpeg(restore)
    write_op = tf.py_func(py_WriteImage, [re_encoded, label], tf.int32, stateful=True)
    return write_op
