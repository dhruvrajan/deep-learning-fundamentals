import time

# import tensorflow as tf
from keras import backend as K
from keras.callbacks import TensorBoard
import keras

class A:
    def __init__(self):

        import tensorflow as tf
        self.keras = keras
        self.nn = tf.nn
tf = A()

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
    tf.keras.layers.MaxPool2D(2),
    tf.keras.layers.Activation("relu"),
    tf.keras.layers.Conv2D(20, kernel_size=5),
    tf.keras.layers.SpatialDropout2D(0.2),
    tf.keras.layers.MaxPool2D(2),
    tf.keras.layers.Activation("relu"),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(50, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

start = time.time()

tensorboard = TensorBoard(log_dir="tmp/keras_mnist_logs_{}".format(time.time()))
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
print("Tensorboard Output at: ", tensorboard.log_dir)

model.fit(x_train, y_train, epochs=10, callbacks=[tensorboard], batch_size=32)
loss, acc = model.evaluate(x_test, y_test, batch_size=32)

print("Test_loss: ", loss)
print("Test acc: ", acc)
print("Time taken: ", time.time() - start)
print("Tensorboard Output at: ", tensorboard.log_dir)