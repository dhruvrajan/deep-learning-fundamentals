import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()

m, n = housing.data.shape
scaler = StandardScaler()
scaled_housing_data = scaler.fit_transform(housing.data, housing.target)

scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]

batch_size = m
n_batches = int(np.ceil(m / batch_size))


dataset = tf.data.Dataset.from_tensor_slices((scaled_housing_data_plus_bias.astype(np.float32), housing.target.reshape(-1, 1).astype(np.float32)))
# batched_dataset_iterator = dataset.batch(batch_size=batch_size).make_initializable_iterator()
# next_element = batched_dataset_iterator.get_next()

####
# X, y = next_element

####

n_epochs = 1000
dataset.cache()
dataset = dataset.repeat(n_epochs)

iter = dataset.make_one_shot_iterator()
next_val = iter.get_next()


X, y = next_val

learning_rate = 0.01

# X = tf.placeholder(tf.float32, name="X")
# y = tf.placeholder(tf.float32, name="y")


# X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
# y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")

theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, dtype=tf.float32), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")

# using autodiff
# gradients = tf.gradients(mse, [theta])[0]
# print(tf.shape(gradients))

# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
# training_op = optimizer.minimize(mse)
# not using autodiff
gradients = 2/m * tf.matmul(tf.transpose(X), error)

training_op = tf.assign(theta, theta - learning_rate * gradients)

# training_op = tf.assign(theta, theta - learning_rate * gradients)
init = tf.global_variables_initializer()

# dataset = tf.data.Dataset.from_tensor_slices(raw_data)
# dataset = dataset.batch(MINI_BATCH)
# dataset = dataset.cache()
# dataset = dataset.repeat(EPOCHS)
# iterator = dataset.make_one_shot_iterator()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):

        epoch_loss = 0
        for i in range(m):
            # X_batch, y_batch = sess.run(next_val)
            # y_batch = y_batch.reshape(-1, 1)
            _, loss = sess.run([training_op, mse])
            epoch_loss += loss

        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE = ", epoch_loss)

        # sess.run(batched_dataset_iterator.initializer)
        # epoch_loss = 0
        # num_session_runs = 0
        # while True:
        # num_session_runs += 1
        # try:
        #     # X_batch, y_batch = sess.run(next_element)
        #     _, loss = sess.run([training_op, mse]) #, feed_dict={X: X_batch, y: y_batch})
        #     epoch_loss += loss
        # except tf.errors.OutOfRangeError:
        #     break

        # print("Session runs: {}".format(num_session_runs))
        # if epoch % 100 == 0:
        #     print("Epoch", epoch, "MSE = ", epoch_loss)