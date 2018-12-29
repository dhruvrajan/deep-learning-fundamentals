import sys

import numpy as np
from keras.callbacks import TensorBoard
from keras.layers.core import Dense, Activation
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import logging
import time
from sklearn.datasets import load_iris


def one_hot_encode(array):
    """One-Hot encode a numpy array of (say) strings"""
    uniques, ids = np.unique(array, return_inverse=True)
    return np_utils.to_categorical(ids, len(uniques))


def prepare_data_set():
    """Return (x-train, x-test, y-train, y-test)"""
    # data = load_iris()
    # features = data.iloc[:, 0:4].values
    # labels = data.iloc[:, 4].values
    iris_data = load_iris()
    features = iris_data.data
    labels = iris_data.target

    one_hot = one_hot_encode(labels)
    return train_test_split(features, one_hot, test_size=0.3)


def main(_):
    x_train, x_test, y_train, y_test = prepare_data_set()

    model = Sequential()

    model.add(Dense(16, input_shape=(4,)))
    model.add(Activation('sigmoid'))

    # model.add(Dense(5))
    # model.add(Activation('tanh'))

    model.add(Dense(3))
    model.add(Activation('softmax'))
    optimizer = Adam(lr=0.001)
    start = time.time()
    model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                  metrics=["accuracy"])

    tensorboard = TensorBoard(log_dir="tmp/keras_iris_logs_{}".format(time.time()))
    print("tensorboard at:" + tensorboard.log_dir)
    model.fit(x_train, y_train, epochs=100, validation_split=0.1, batch_size=1, verbose=1,
              callbacks=[tensorboard])

    loss, accuracy = model.evaluate(x_test, y_test, verbose=1)

    logging.info("Time = {:.2f} Loss = {:.2f} Accuracy = {:.2f}".format(time.time() - start, loss, accuracy))
    print("Tensorboard output at: " + tensorboard.log_dir)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    main(sys.argv)
