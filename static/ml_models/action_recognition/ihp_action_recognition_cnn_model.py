from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np


tf.enable_eager_execution()


def get_prediction(feature, camera_degree):
    model = tf.keras.models.load_model(f'static/ml_models/action_recognition/inhand_action_recognition_cnn_model_{camera_degree}.hdf5')
    features = np.array([feature], dtype=np.float32)

    result = model.predict_classes(features, verbose=0)
    predictions = model.predict(features)
    hypothesis = tf.nn.softmax(predictions)
    hypothesis = float(hypothesis[:, 1:])
    return result, hypothesis


if __name__ == '__main__':
    result, hy = get_prediction([1., 1., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0.19276, 0.07373], 120)
