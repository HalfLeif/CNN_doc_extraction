
import tensor_transformations as ts

import numpy as np
import tensorflow as tf


class TensorTransformationsTest(tf.test.TestCase):

    def testReplicate(self):
        with self.test_session():
            vector = tf.range(3)
            vector = tf.reshape(vector, [1, 3, 1])
            vector_eval = vector.eval()

            first = ts.replicateAlong(vector, 2, axis=0)
            self.assertAllEqual(first.get_shape(), [2, 3, 1])
            first = first.eval()
            self.assertAllEqual(first[0], vector_eval[0])
            self.assertAllEqual(first[1], vector_eval[0])

            second = ts.replicateAlong(vector, 2, axis=1)
            self.assertAllEqual(second.get_shape(), [1, 6, 1])
            second = second.eval()
            self.assertAllEqual(second[0,0:3], vector_eval[0])
            self.assertAllEqual(second[0,3:], vector_eval[0])

            third = ts.replicateAlong(vector, 2, axis=-1)
            self.assertAllEqual(third.get_shape(), [1, 3, 2])
            third = third.eval()
            for i in range(3):
                self.assertAllEqual(third[0][i], np.array([i, i]))


if __name__ == '__main__':
    tf.test.main()
