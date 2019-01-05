import torch.nn as nn
from argparse import ArgumentParser
from torch import optim
from torch.utils.data import DataLoader

from sentiment_analysis.models import SimpleFFNN
from sentiment_analysis.sentiment_utils import load_sentiment_data, SentimentDataset, WordVectors

def evaluate_simple_ffnn(trained_model, tensors, msg, args):
    data_loader = DataLoader(tensors, args.val_batch_size)

    total = 0
    correct = 0
    for X_batch, y_batch in data_loader:
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
    model = SimpleFFNN(word_vectors, embedding_size, 256, 128, 64, 2, freeze=args.freeze)
    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

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
        evaluate_simple_ffnn(model, train_tensors, "train", args)
        evaluate_simple_ffnn(model, test_tensors, "dev", args)

    return model


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--val_batch_size', type=int, default=1,
                        help='input batch size for validation (default: 1000)')
    parser.add_argument('--test_batch_size', type=int, default=1,
                        help='input batch size for test (default: 1000)')
    parser.add_argument('--epochs', type=int, default=20,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--freeze', type=bool, default=False,
                        help='freeze embedding layer (default: True)')

    args = parser.parse_args()
    train_data, test_data, word_vectors = load_sentiment_data(300)
    trained_model = train_simple_ffnn(train_data, test_data, word_vectors, args)