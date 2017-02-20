
import conv

import tensorflow as tf

class DecodeTest(tf.test.TestCase):

    def testDigitsToNumber_decimal(self):
        with self.test_session():
            d1 = tf.one_hot(1, 10)
            d2 = tf.one_hot(2, 10)
            d3 = tf.one_hot(3, 10)
            self.assertAllEqual(d1.eval(), [0, 1, 0, 0, 0, 0, 0, 0, 0, 0])

            number_prob = conv.digitsToNumber(d1, d2, d3)
            number = tf.argmax(number_prob, axis=0)
            self.assertAllEqual(number.eval(), 123)

    def testDigitsToNumber_binary(self):
        with self.test_session():
            d1 = tf.constant([0, 1], dtype=tf.float32)
            d2 = tf.constant([0.5, 0.5], dtype=tf.float32)
            d3 = tf.constant([1, 0], dtype=tf.float32)

            number_prob = conv.digitsToNumber(d1, d2, d3)
            # Either 100 (4) or 110 (6):
            self.assertAllEqual(number_prob.eval(), [0, 0, 0, 0, 0.5, 0, 0.5, 0])


if __name__ == '__main__':
    tf.test.main()
