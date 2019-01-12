import time

from argparse import ArgumentParser
from keras import Sequential
from keras.callbacks import TensorBoard
from keras.initializers import Constant
from keras.layers import *
from sentiment_analysis.sentiment_data import load_sentiment_data

def sentiment_args(parser):
    parser.add_argument("--batch_size", type=int, default=64,
                        help="input batch size for training (default: 64)")
    parser.add_argument("--val_batch_size", type=int, default=64,
                        help="input batch size for validation (default: 1000)")
    parser.add_argument("--test_batch_size", type=int, default=1,
                        help="input batch size for test (default: 1000)")
    parser.add_argument("--epochs", type=int, default=20,
                        help="number of epochs to train (default: 10)")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--embedding_dim", type=int, default=100,
                        help="word embedding vector size")
    parser.add_argument("--freeze", type=bool, default=False,
                        help="freeze embedding layer (default: True)")
    parser.add_argument("--log_interval", type=int, default=10,
                        help="how many batches to wait before logging training status")
    parser.add_argument("--checkpoint_interval", type=int, default=10,
                        help="how many epochs between checkpointing the model")
    parser.add_argument("--n_checkpoints", type=int, default=1,
                        help="how many checkpointed models to save")
    parser.add_argument("--log_dir", type=str, default="tmp/logs/" + prefix + "_" + time_label,
                        help="log directory for Tensorboard log output")
    parser.add_argument("--run_id", type=str, default=prefix + "_" + time_label,
                        help="identifier for the run")




def main(args):
    train_data, val_data, word_vectors = load_sentiment_data(args.embedding_dim)
    X_train, y_train, input_lengths_train = train_data.create_numpy_dataset()
    X_val, y_val, input_lengths_val = val_data.create_numpy_dataset(pad_to=X_train.shape[1])

    model = Sequential([
        Embedding(
            len(word_vectors.word_indexer),
            args.embedding_dim,
            embeddings_initializer=Constant(word_vectors.vectors),
            input_length=X_train.shape[1],
            trainable=False
        ),
        Bidirectional(LSTM(50)),
        # Dense(5, activation='sigmoid'),
        Dense(3, activation='sigmoid'),
        Dense(1, activation='sigmoid')
    ])

    print(model.summary())

    start = time.time()
    tensorboard = TensorBoard(log_dir="tmp/logs/keras_sentiment_logs_{}".format(time.time()))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_val, y_val), callbacks=[tensorboard])
    loss, acc = model.evaluate(X_val, y_val)

    print("")
    print("Test_loss: ", loss)
    print("Dev acc: ", acc)
    print("Time taken: ", time.time() - start)
    print("Tensorboard Output at: ", tensorboard.log_dir)






if __name__ == '__main__':
    time_label = str(time.time())
    prefix = "pytorch_sentiment_logs"

    # Get Arguments
    parser = ArgumentParser()
    sentiment_args(parser)
    args = parser.parse_args()
    main(args)

