import scoring as sc

import tensorflow as tf

class ScoringTest(tf.test.TestCase):

    # def digitEqual(self, digit, expected_digit):
    #     expected = tf.one_hot(expected_digit, 10)
    #     self.assertAllEqual(digit.eval(), expected.eval())
    #
    # def testEncodeLabel(self):
    #     with self.test_session():
    #         label = sc.encodeYear(1891)
    #         self.assertAllEqual(len(label), 3)
    #         self.digitEqual(label[0], 8)
    #         self.digitEqual(label[1], 9)
    #         self.digitEqual(label[2], 1)
    #
    # def testPrediction(self):
    #     with self.test_session():
    #         year_prob = sc.encodeYear(1881)
    #         decision = tf.constant([0,1], tf.float32)
    #
    #         # Expand dims to create a batch with one unit
    #         year_prob = [tf.expand_dims(d, axis=0) for d in year_prob]
    #         decision = tf.expand_dims(decision, axis=0)
    #
    #         year = sc.predict(decision, year_prob)
    #         self.assertAllEqual(year.eval(), [1881])
    #
    #         decision = tf.constant([0.6,0.4], tf.float32)
    #         decision = tf.expand_dims(decision, axis=0)
    #         year = sc.predict(decision, year_prob)
    #         self.assertAllEqual(year.eval(), [-1])


    def testClusterError(self):
        with self.test_session():
            single = sc.clusterError(tf.constant([[0, 0, 1, 0]], tf.float32)).eval()
            cluster = sc.clusterError(tf.constant([[0, 0, 0.5, 0.5]], tf.float32)).eval()
            double = sc.clusterError(tf.constant([[0.5, 0, 0.5, 0]], tf.float32)).eval()
            max_error = sc.clusterError(tf.constant([[0.5, 0, 0, 0, 0, 0, 0, 0.5]], tf.float32)).eval()

            noisy_single = sc.clusterError(tf.constant([[0.1, 0.1, 0.7, 0.1]], tf.float32)).eval()

            self.assertLess(single, noisy_single)
            self.assertLess(single, cluster)
            self.assertLess(cluster, double)
            self.assertLess(double, max_error)
            self.assertLess(noisy_single, double)

            xs = sc.clusterError(tf.constant([[0, 0, 1, 0], [0, 1, 0, 0], [0, 0.5, 0.5, 0]], tf.float32)).eval()
            self.assertAllEqual(xs[0], xs[1])
            self.assertLess(xs[0], xs[2])

    # def testError_hasNumber(self):
    #     with self.test_session():
    #         decision = tf.constant([0, 1], tf.float32)
    #         prediction = sc.encodeYear(1881)
    #         perfect = sc.error(1881, decision, prediction).eval()
    #         wrong_class = sc.error(1882, decision, prediction).eval()
    #
    #         decision = tf.constant([0.5, 0.5], tf.float32)
    #         empty_or_correct = sc.error(1881, decision, prediction).eval()
    #         empty_or_wrong = sc.error(1882, decision, prediction).eval()
    #
    #         self.assertLess(perfect, empty_or_correct)
    #         self.assertLess(empty_or_correct, wrong_class)
    #         self.assertLess(wrong_class, empty_or_wrong)
    #
    #         decision = tf.constant([1, 0], tf.float32)
    #         prediction = sc.encodeYear(1881)
    #         false_negative = sc.error(1881, decision, prediction).eval()
    #
    #         self.assertLess(empty_or_correct, false_negative)
    #
    # def testError_emptyYear(self):
    #     with self.test_session():
    #         decision = tf.constant([1, 0], tf.float32)
    #         prediction = sc.encodeYear(881)
    #         true_negative = sc.error(-1, decision, prediction).eval()
    #
    #         decision = tf.constant([0.5, 0.5], tf.float32)
    #         uncertain = sc.error(-1, decision, prediction).eval()
    #
    #         decision = tf.constant([0, 1], tf.float32)
    #         false_positive = sc.error(-1, decision, prediction).eval()
    #
    #         self.assertLess(true_negative, uncertain)
    #         self.assertLess(uncertain, false_positive)


if __name__ == '__main__':
    tf.test.main()
