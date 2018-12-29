import time

import tensorflow as tf
from keras import backend as K
mnist = tf.keras.datasets.mnist



(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

img_rows, img_cols = 28, 28

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(10, kernel_size=5, input_shape=input_shape),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Activation("relu"),
    tf.keras.layers.Conv2D(20, kernel_size=5),
    tf.keras.layers.SpatialDropout2D(0.2),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Activation("relu"),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(50, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

start = time.time()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
loss, acc = model.evaluate(x_test, y_test)

print("Test_loss: ", loss)
print("Test acc: ", acc)
print("Time taken: ", time.time() - start)