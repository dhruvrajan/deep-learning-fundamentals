import time

import torch.nn as nn
from argparse import ArgumentParser
from tensorboardX import SummaryWriter
from torch import optim
from torch.utils.data import DataLoader

from sentiment_analysis.models import SimpleFFNN, SimpleLSTM
from sentiment_analysis.sentiment_data import load_sentiment_data, SentimentDataset, WordVectors


def create_summary_writer(model, data_loader, log_dir):
    writer = SummaryWriter(log_dir=log_dir)

    data_loader_iter = iter(data_loader)
    x, y, _ = next(data_loader_iter)
    try:
        writer.add_graph(model, x)
    except Exception as e:
        print("Failed to save model graph: {}".format(e))
    return writer


def evaluate_sentiment_model(trained_model, tensors, msg, args):
    data_loader = DataLoader(tensors, args.val_batch_size)

    total = 0
    correct = 0
    for X_batch, y_batch, _ in data_loader:
        if X_batch.shape[0] == args.batch_size:
            output_probabilities = trained_model(X_batch)
            predicted = output_probabilities.argmax(dim=1)

            for i in range(predicted.shape[0]):
                if predicted[i].item() == y_batch[i].item():
                    correct += 1

                total += 1

    print("model accuracy ({}): {} / {} = {}".format(msg, correct, total, correct / total))
    return total, correct


def train_simple_ffnn(train_data: SentimentDataset, test_data, word_vectors: WordVectors, args):
    train_tensors = train_data.create_tensor_dataset()
    test_tensors = test_data.create_tensor_dataset()

    data_loader = DataLoader(train_tensors, batch_size=args.batch_size, shuffle=True)

    embedding_size = word_vectors.vectors.shape[1]
    # model = SimpleLSTM(word_vectors, embedding_size, 10, 2, 0.2, False, args)
    model = SimpleFFNN(word_vectors, embedding_size, 8, 4, 1, 2, freeze=args.freeze)

    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    summary_writer = create_summary_writer(model, data_loader, args.log_dir)

    for epoch in range(args.epochs):
        epoch_loss = 0
        for i_batch, (X_batch, y_batch, input_lengths) in enumerate(data_loader):
            model.zero_grad()
            log_probs = model(X_batch)

            loss = loss_function(log_probs, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print("Epoch {}, loss={}".format(epoch, epoch_loss))
        summary_writer.add_scalar("training/loss", epoch_loss, epoch)

        total, correct = evaluate_sentiment_model(model, train_tensors, "train", args)
        summary_writer.add_scalar("train/accuracy", correct / total, epoch)

        total, correct = evaluate_sentiment_model(model, test_tensors, "dev", args)
        summary_writer.add_scalar("dev/accuracy", correct / total, epoch)

    run_summary = {
        "model_summary": model.summary,
        "optimizer_summary": {
            "type": "Adam",
            "learning_rate": args.lr
        },
        "run_params": {
            "epochs": args.epochs,
            "train_batch_size": args.batch_size

        }
    }

    for key in run_summary:
        print(key, str(run_summary[key]))
        summary_writer.add_text(key, str(run_summary[key]))

    summary_writer.close()

    return model


def train_simple_lstm(train_data: SentimentDataset, test_data, word_vectors: WordVectors, args):
    train_tensors = train_data.create_tensor_dataset(args.batch_size)
    test_tensors = test_data.create_tensor_dataset(args.val_batch_size)

    data_loader = DataLoader(train_tensors, batch_size=args.batch_size, shuffle=True)

    embedding_size = word_vectors.vectors.shape[1]
    model = SimpleLSTM(word_vectors, embedding_size, 10, 2, 0.2, False, args)

    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    summary_writer = create_summary_writer(model, data_loader, args.log_dir)

    for epoch in range(args.epochs):
        epoch_loss = 0

        for i_batch, (X_batch, y_batch, input_lengths) in enumerate(data_loader):
            if X_batch.shape[0] == args.batch_size:
                model.zero_grad()
                model.hidden_state = model.init_hidden(model.hidden_size)
                log_probs = model(X_batch)

                loss = loss_function(log_probs, y_batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

        print("Epoch {}, loss={}".format(epoch, epoch_loss))
        summary_writer.add_scalar("train/loss", epoch_loss, epoch)

        total, correct = evaluate_sentiment_model(model, train_tensors, "train", args)
        summary_writer.add_scalar("train/accuracy", correct / total, epoch)

        total, correct = evaluate_sentiment_model(model, test_tensors, "dev", args)
        summary_writer.add_scalar("dev/accuracy", correct / total, epoch)


    run_summary = {
        "model_summary": model.summary,
        "optimizer_summary": {
            "type": "Adam",
            "learning_rate": args.lr
        },
        "run_params": {
            "epochs": args.epochs,
            "train_batch_size": args.batch_size

        }
    }

    for key in run_summary:
        print(key, str(run_summary[key]))

        summary_writer.add_text(key, str(run_summary[key]))
    summary_writer.close()
    return model


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--val_batch_size', type=int, default=64,
                        help='input batch size for validation (default: 1000)')
    parser.add_argument('--test_batch_size', type=int, default=1,
                        help='input batch size for test (default: 1000)')
    parser.add_argument('--epochs', type=int, default=20,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--freeze', type=bool, default=False,
                        help='freeze embedding layer (default: True)')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='how many batches to wait before logging training status')
    parser.add_argument("--log_dir", type=str, default="tmp/pytorch_sentiment_logs_" + str(time.time()),
                        help="log directory for Tensorboard log output")

    args = parser.parse_args()
    print("TensorboardX output at: ", args.log_dir)
    train_data, test_data, word_vectors = load_sentiment_data(300)
    trained_model = train_simple_lstm(train_data, test_data, word_vectors, args)
    print("TensorboardX output at: ", args.log_dir)
