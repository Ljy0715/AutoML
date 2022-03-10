# import autosklearn.classification
# from sklearn import datasets
# from sklearn import model_selection
# from sklearn import metrics
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
# import keras.datasets.cifar10
# import keras.datasets.mnist
# from autosklearn.classification import AutoSklearnClassifier
import numpy as np
import tensorflow as tf
import autokeras as ak
from tensorflow.python.client import device_lib
#
# print("版本:", tf.__version__)
# print("型号:", device_lib.list_local_devices())

# print(tf.test.is_gpu_available())
print(tf.config.list_physical_devices('GPU'))
print("loading CIFAR10 dataset...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
print(x_train.shape)  # (60000, 28, 28)
print(y_train.shape)  # (60000,)
print(y_train[:3])  # array([7, 2, 1], dtype=uint8)

# Initialize the image classifier.
clf = ak.ImageClassifier(overwrite=True, max_trials=1)
# Feed the image classifier with training data.
clf.fit(x_train, y_train, epochs=10)


# Predict with the best model.
predicted_y = clf.predict(x_test)
print(predicted_y)


# Evaluate the best model with testing data.
print(clf.evaluate(x_test, y_test))
