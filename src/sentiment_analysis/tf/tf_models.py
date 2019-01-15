import tensorflow as tf
import numpy as np

def embedding_layer(X, word_vectors, trainable=False, name="embedding_layer"):
    with tf.name_scope(name):
        embeddings = tf.Variable(word_vectors.vectors, trainable=trainable)
        return tf.nn.embedding_lookup(embeddings, X)


def xavier_normal(in_dim, out_dim, name="xavier_init"):
    stddev = 2 / np.sqrt(in_dim + out_dim)
    return tf.truncated_normal((in_dim, out_dim), stddev=stddev)


def linear_layer(X, out_dim, name, activation=None):
    with tf.name_scope(name):
        in_dim = int(X.get_shape()[1])

        # linear layer definition
        W = tf.Variable(xavier_normal(in_dim, out_dim), name="kernel")
        b = tf.Variable(tf.zeros([out_dim]), name="bias")
        Z = tf.matmul(tf.cast(X, tf.float32), W) + b

        return activation(Z) if activation else Z

def rnn_sentiment_model(X, lstm_hid, out, word_vectors, name="rnn_sentiment_model"):
    with tf.name_scope(name):
        rnn_cell = tf.contrib.rnn.BasicRNNCell(num_units=lstm_hid)
        embeddings = embedding_layer(X, word_vectors, trainable=False)
        outputs, states = tf.nn.dynamic_rnn(rnn_cell, embeddings, dtype=tf.float64)
        log_probs = linear_layer(states, out, name="fc_layer", activation=tf.nn.log_softmax)
        return log_probs



def ffnn_sentiment_model(X, word_vectors, hid1, hid2, hid3, out, freeze, name="ffnn_sentiment_model"):
    with tf.name_scope(name):
        embeddings = embedding_layer(X, word_vectors, trainable=not freeze)
        averaged_embeddings = tf.reduce_mean(embeddings, axis=1)
        hidden1 = linear_layer(averaged_embeddings, hid1, name="hidden1", activation=tf.nn.tanh)
        hidden2 = linear_layer(hidden1, hid2, name="hidden2", activation=tf.nn.sigmoid)
        hidden3 = linear_layer(hidden2, hid3, name="hidden3", activation=tf.nn.tanh)
        output = linear_layer(hidden3, out, name="output_layer", activation=tf.nn.log_softmax)
        return output