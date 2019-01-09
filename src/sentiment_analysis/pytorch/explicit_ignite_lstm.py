import time

import torch
import torch.nn as nn
from argparse import ArgumentParser
from ignite.engine import Engine
from torch.optim import Adam

from sentiment_analysis.models import SimpleLSTM
from sentiment_analysis.sentiment_data import load_sentiment_data
from sentiment_analysis.sentiment_trainer import train_sentiment_model


def sentiment_args(parser):
    parser.add_argument("--batch_size", type=int, default=1,
                        help="input batch size for training (default: 64)")
    parser.add_argument("--val_batch_size", type=int, default=1,
                        help="input batch size for validation (default: 1000)")
    parser.add_argument("--test_batch_size", type=int, default=1,
                        help="input batch size for test (default: 1000)")
    parser.add_argument("--epochs", type=int, default=20,
                        help="number of epochs to train (default: 10)")
    parser.add_argument("--lr", type=float, default=0.0001,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--embedding_size", type=int, default=300,
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


def create_lstm_trainer(model: nn.Module, optimizer, loss_function, args):
    def _training_step(engine, batch):
        model.train()
        # model.zero_grad()
        optimizer.zero_grad()

        X, y, input_lengths = batch
        # model.hidden_state = model.init_hidden(model.hidden_dim, X.shape[0])
        model.hidden = model.init_hidden()
        log_probs = model(X, input_lengths)
        loss = loss_function(log_probs, y)

        loss.backward()
        optimizer.step()
        return loss.item()

    return Engine(_training_step)


def create_lstm_evaluator(model: nn.Module, metrics: dict, args):
    def _inference_step(engine, batch):
        model.eval()
        with torch.no_grad():
            X, y, input_lengths = batch

            log_probs = model(X, input_lengths)
            return log_probs, y

    engine = Engine(_inference_step)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def main(args):
    print("TensorboardX output at: ", args.log_dir)
    train_data, test_data, word_vectors = load_sentiment_data(args.embedding_size)

    model = SimpleLSTM(word_vectors, args.embedding_size, 10, 2, 0.2, False, args)
    loss_function = nn.NLLLoss()
    optimizer = Adam(model.parameters(), args.lr)

    train_sentiment_model(model, loss_function, optimizer,
                          create_lstm_trainer, create_lstm_evaluator,
                          train_data, test_data, args)

    print("TensorboardX output at: ", args.log_dir)


if __name__ == "__main__":
    time_label = str(time.time())
    prefix = "pytorch_sentiment_logs"

    # Get Arguments
    parser = ArgumentParser()
    sentiment_args(parser)
    args = parser.parse_args()

    main(args)
