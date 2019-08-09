import tensorflow as tf
import numpy as np
import os
from matplotlib import pyplot as plt
tf.set_random_seed(777)  # for reproducibility


# Launch graph
def predict_medication(test_data, camera_degree=220):
    tf.set_random_seed(777)  # for reproducibility

    # 피쳐의 개수에 따라서 shape의 숫자를 바꿔준다.
    # placeholders for a tensor that will be always fed.
    if camera_degree == 220:
        # 각도에 따른 모델 변경
        save_file = 'static/ml_models/model_logistic.ckpt'

        X = tf.placeholder(tf.float32, shape=[None, 14])
        Y = tf.placeholder(tf.float32, shape=[None, 1])

        W = tf.Variable(tf.random_normal([14, 1]), name='weight')
        b = tf.Variable(tf.random_normal([1]), name='bias')
    else:
        # 각도에 따른 모델 변경
        save_file = 'static/ml_models/model_logistic.ckpt'

        X = tf.placeholder(tf.float32, shape=[None, 12])
        Y = tf.placeholder(tf.float32, shape=[None, 1])

        W = tf.Variable(tf.random_normal([12, 1]), name='weight')
        b = tf.Variable(tf.random_normal([1]), name='bias')


    # Hypothesis using sigmoid: tf.div(1., 1. + tf.exp(-tf.matmul(X, W)))
    hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

    # cost/loss function
    cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) *
                           tf.log(1 - hypothesis))

    train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

    # Accuracy computation
    # True if hypothesis>0.5 else False
    predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

    SAVER_DIR = "model"
    saver = tf.train.Saver()
    checkpoint_path = os.path.join(SAVER_DIR, "model")

    with tf.Session() as sess:
        # Initialize TensorFlow variables
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        # saver.save(sess, save_file)
        saver.restore(sess, save_file)

        result, hy = sess.run([predicted, hypothesis],
                              feed_dict={X: [test_data]})
        print(result, hy)
        # saver.save(sess, save_file)
        result = result.flatten()
        result = int(result[0])
        hy = hy.flatten()
        hy = float(hy[0])
    return result, hy


if __name__ == '__main__':
    predict_medication([1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0.03551, 0.06142])
