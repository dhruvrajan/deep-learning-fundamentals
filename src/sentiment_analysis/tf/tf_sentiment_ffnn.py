import time

from argparse import ArgumentParser
from datetime import datetime

from sentiment_analysis.sentiment_data import SentimentDataset, load_sentiment_data
from sentiment_analysis.tf.tf_models import *


def sentiment_args(parser):
    now  = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir  = "tmp/logs"
    logdir = "{}/tf_sentiment_{}/".format(root_logdir, now)

    time_label = str(time.time())
    prefix = "tf_sentiment_logs"
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




def train_ffnn(train_data: SentimentDataset, dev_data: SentimentDataset, word_vectors):
    t = time.time()
    X_train, y_train, input_lengths_train = train_data.create_numpy_dataset()
    X_val, y_val, input_lengths_dev = dev_data.create_numpy_dataset(pad_to=X_train.shape[1])

    input_features = X_train.shape[1]

    with tf.name_scope("data"):
        X = tf.placeholder(tf.int32, shape=(None, input_features), name="X")
        y = tf.placeholder(tf.int32, shape=(None), name="y")

    # log_probs = ffnn_sentiment_model(X, word_vectors, 8, 4, 1, 2, freeze=args.freeze)
    log_probs = rnn_sentiment_model(X, 15, 2, word_vectors)

    with tf.name_scope("loss"):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=log_probs)
        loss = tf.reduce_mean(xentropy, name="loss")
        train_loss_summary = tf.summary.scalar("train_loss", loss)
        val_loss_summary = tf.summary.scalar("val_loss", loss)

    with tf.name_scope("train"):
        optimizer = tf.train.AdamOptimizer(learning_rate=args.lr)
        training_op = optimizer.minimize(loss)

    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(log_probs, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        train_accuracy_summary = tf.summary.scalar("train_accuracy", accuracy)
        val_accuracy_summary = tf.summary.scalar("val_accuracy", accuracy)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    num_batches = X_train.shape[0] // args.batch_size

    def fetch_batch(batch_id):
        start = batch_id * args.batch_size
        return X_train[start: start + args.batch_size], y_train[start: start + args.batch_size]

    file_writer = tf.summary.FileWriter(args.log_dir, tf.get_default_graph())
    with tf.Session() as sess:
        init.run()
        for epoch in range(args.epochs):
            for batch in range(num_batches):
                X_batch, y_batch = fetch_batch(batch)
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

            acc_train = accuracy.eval(feed_dict={X: X_train, y: y_train})
            acc_dev = accuracy.eval(feed_dict={X: X_val, y: y_val})

            train_loss_summary_str = train_loss_summary.eval(feed_dict={X: X_train, y: y_train})
            train_accuracy_summary_str = train_accuracy_summary.eval(feed_dict={X: X_train, y: y_train})
            val_loss_summary_str = val_loss_summary.eval(feed_dict={X: X_val, y: y_val})
            val_accuracy_summary_str = val_accuracy_summary.eval(feed_dict={X: X_val, y: y_val})
            file_writer.add_summary(train_loss_summary_str, epoch)
            file_writer.add_summary(train_accuracy_summary_str, epoch)
            file_writer.add_summary(val_loss_summary_str, epoch)
            file_writer.add_summary(val_accuracy_summary_str, epoch)

            print("Epoch #{}: acc_train: {}, acc_dev: {}".format(epoch, acc_train, acc_dev))

        save_path = saver.save(sess, "./tf_sentiment_{}.ckpt".format(t))

    file_writer.close()

    print("loaded train tensor", X_train.shape, "and dev tensor", X_val.shape)


def main(args):
    train_data, dev_data, word_vectors = load_sentiment_data(args.embedding_size)
    train_ffnn(train_data, dev_data, word_vectors)


if __name__ == '__main__':
    # Get Arguments
    parser = ArgumentParser()
    sentiment_args(parser)
    args = parser.parse_args()

    start = time.time()
    main(args)
