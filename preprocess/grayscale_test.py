import grayscale as gray

import tensorflow as tf

class ScoringTest(tf.test.TestCase):

    def testSimpleRemap(self):
        with self.test_session():
            pixels = [3, 7, 5]
            expect = tf.constant([0, 1, 0.5], tf.float32)
            remapped = gray.simpleRemap(pixels)
            self.assertAllEqual(expect.eval(), remapped.eval())

if __name__ == '__main__':
    tf.test.main()
