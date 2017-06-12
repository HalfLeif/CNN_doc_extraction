
import decoder

import tensorflow as tf

def stackDigits(digit_list, base):
    onehots = map(lambda d: tf.one_hot(d, base), digit_list)
    return tf.stack(list(onehots))

def expandTest(digit_list, base):
    stack = stackDigits(digit_list, base)
    number_prob = decoder.expandDigits(stack)
    number = tf.argmax(number_prob, axis=0)
    return number

class DecodeTest(tf.test.TestCase):

    def testDigitsToNumber_decimal(self):
        with self.test_session():
            number = expandTest([1, 2], 10)
            self.assertAllEqual(number.eval(), 12)

            number = expandTest([1, 2, 3], 10)
            self.assertAllEqual(number.eval(), 123)

    def testDigitsToNumber_binary(self):
        with self.test_session():
            number = expandTest([0, 0, 0, 0, 1], 2)
            self.assertAllEqual(number.eval(), 1)

            number = expandTest([0, 0, 0, 1, 0], 2)
            self.assertAllEqual(number.eval(), 2)

            number = expandTest([0, 0, 1, 0, 0], 2)
            self.assertAllEqual(number.eval(), 4)

            number = expandTest([0, 1, 0, 0, 0], 2)
            self.assertAllEqual(number.eval(), 8)

            number = expandTest([1, 0, 0, 0, 0], 2)
            self.assertAllEqual(number.eval(), 16)

            number = expandTest([1, 0, 0, 0, 1], 2)
            self.assertAllEqual(number.eval(), 17)


if __name__ == '__main__':
    tf.test.main()
