import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from datetime import datetime

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)

housing = fetch_california_housing()

m, n = housing.data.shape
scaler = StandardScaler()
scaled_housing_data = scaler.fit_transform(housing.data, housing.target)

scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]
target = housing.target.reshape(-1, 1)

batch_size = 10
n_batches = int(np.ceil(int(len(scaled_housing_data_plus_bias))))


def fetch_batch(epoch, batch_index, batch_size=batch_size):
    np.random.seed(epoch * n_batches + batch_index)  # not shown in the book
    indices = np.random.randint(m, size=batch_size)  # not shown
    X_batch = scaled_housing_data_plus_bias[indices] # not shown
    y_batch = housing.target.reshape(-1, 1)[indices] # not shown
    return X_batch, y_batch
# def fetch_batch(batch_idx):
#     start = min(len(scaled_housing_data_plus_bias) - 1, batch_idx * batch_size)
#     stop = min(len(scaled_housing_data_plus_bias) - 1, start + batch_size)
#     return scaled_housing_data_plus_bias[start:stop + 1], target[start:stop + 1]


n_epochs = 1000
learning_rate = 0.01

X = tf.placeholder(dtype=tf.float32, name="X")
y = tf.placeholder(dtype=tf.float32, name="y")

theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")

# not using autodiff

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)
# gradients = 2 / m * tf.matmul(tf.transpose(X), error)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

mse_summary = tf.summary.scalar("MSE", mse)
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        for batch_idx in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_idx)
            _, loss = sess.run([training_op, mse], feed_dict={X: X_batch, y: y_batch})
        if epoch % 100 == 0:
            import os

            print(os.getcwd())
            saver.save(sess, "tmp/my_model.ckpt")
            print("Epoch", epoch, "MSE = ", loss)

    best_theta = sess.run(theta)

with tf.Session() as sess:
    saver.restore(sess, "tmp/my_model.ckpt")
