import scoring as sc

import tensorflow as tf

class ScoringTest(tf.test.TestCase):

    def testEncodeLabel_single(self):
        with self.test_session():
            has, year = sc.encodeLabel([1881])
            expect = tf.one_hot(881, 1000)
            self.assertAllEqual(year.eval(), expect.eval())
            self.assertTrue(has.eval())

    def testEncodeLabel_multi_max(self):
        with self.test_session():
            has, year = sc.encodeLabel([1881, 1882])
            expect = tf.one_hot(882, 1000)
            self.assertAllEqual(year.eval(), expect.eval())
            self.assertTrue(has.eval())

    def testEncodeLabel_none(self):
        with self.test_session():
            has, year = sc.encodeLabel([])
            expect = tf.one_hot(-1, 1000)
            self.assertAllEqual(year.eval(), expect.eval())
            self.assertFalse(has.eval())

    def testError_hasNumber(self):
        with self.test_session():
            decision = tf.constant([0, 1], tf.float32)
            prediction = tf.one_hot(881, 1000)
            perfect = sc.error([1881], decision, prediction).eval()
            wrong_class = sc.error([1882], decision, prediction).eval()

            decision = tf.constant([0.5, 0.5], tf.float32)
            empty_or_correct = sc.error([1881], decision, prediction).eval()
            empty_or_wrong = sc.error([1882], decision, prediction).eval()

            self.assertLess(perfect, empty_or_correct)
            self.assertLess(empty_or_correct, wrong_class)
            self.assertLess(wrong_class, empty_or_wrong)

            decision = tf.constant([1, 0], tf.float32)
            prediction = tf.one_hot(881, 1000)
            false_negative = sc.error([1881], decision, prediction).eval()

            self.assertLess(empty_or_correct, false_negative)

    def testError_emptyYear(self):
        with self.test_session():
            decision = tf.constant([1, 0], tf.float32)
            prediction = tf.one_hot(881, 1000)
            true_negative = sc.error([], decision, prediction).eval()

            decision = tf.constant([0.5, 0.5], tf.float32)
            uncertain = sc.error([], decision, prediction).eval()

            decision = tf.constant([0, 1], tf.float32)
            false_positive = sc.error([], decision, prediction).eval()

            self.assertLess(true_negative, uncertain)
            self.assertLess(uncertain, false_positive)


if __name__ == '__main__':
    tf.test.main()
