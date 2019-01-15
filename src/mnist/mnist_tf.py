import tensorflow as tf
from datetime import datetime
import time

from argparse import ArgumentParser
from tensorflow.examples.tutorials.mnist import input_data

def mnist_args(parser):
    now  = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir  = "tmp/logs"
    logdir = "{}/tf_sentiment_{}/".format(root_logdir, now)

    time_label = str(time.time())
    prefix = "tf_sentiment_logs"
    parser.add_argument("--input_data", type=str, default="data/mnist/processed",
                        help="input batch size for training (default: 64)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="input batch size for training (default: 64)")
    parser.add_argument("--val_batch_size", type=int, default=64,
                        help="input batch size for validation (default: 1000)")
    parser.add_argument("--test_batch_size", type=int, default=1,
                        help="input batch size for test (default: 1000)")
    parser.add_argument("--epochs", type=int, default=20,
                        help="number of epochs to train (default: 10)")
    parser.add_argument("--lr", type=float, default=0.0005,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--embedding_size", type=int, default=100,
                        help="word embedding vector size")
    parser.add_argument("--freeze", type=bool, default=False,
                        help="freeze embedding layer (default: True)")
    parser.add_argument("--log_interval", type=int, default=10,
                        help="how many batches to wait before logging training status")
    parser.add_argument("--checkpoint_interval", type=int, default=10,
                        help="how many epochs between checkpointing the model")
    parser.add_argument("--n_checkpoints", type=int, default=1,
                        help="how many checkpointed models to save")
    parser.add_argument("--log_dir", type=str, default=logdir,
                        help="log directory for Tensorboard log output")



def main(args):
    mnist_data = input_data.read_data_sets(args.input_data)
    print("loaded mnist_data")

    n_steps = 28
    n_inputs = 28
    n_neurons = 150
    n_outputs = 10


    X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
    y = tf.placeholder(tf.int32, [None])

    basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
    outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)

    log_probs = tf.layers.dense(states, n_outputs)
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=log_probs)

    loss = tf.reduce_mean(xentropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    training_op = optimizer.minimize(loss)

    correct = tf.nn.in_top_k(log_probs, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    init = tf.global_variables_initializer()

    X_train = mnist_data.train.images.reshape((-1, n_steps, n_inputs))
    y_train = mnist_data.train.labels


    X_test = mnist_data.test.images.reshape((-1, n_steps, n_inputs))
    y_test = mnist_data.test.labels

    n_epochs = 100
    batch_size = 150

    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):
            for iteration in range(mnist_data.train.num_examples // batch_size):
                X_batch, y_batch = mnist_data.train.next_batch(batch_size)
                X_batch = X_batch.reshape((-1, n_steps, n_inputs))
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

            acc_train = accuracy.eval(feed_dict={X: X_train, y: y_train})
            acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
            print(epoch, "Train accuracy:", acc_train, "Test accuracy", acc_test
                  )

if __name__ == '__main__':
    parser = ArgumentParser()
    mnist_args(parser)
    args = parser.parse_args()

    main(args)