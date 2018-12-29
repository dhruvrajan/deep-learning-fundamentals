import time

import logging
from argparse import ArgumentParser
import torch
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
import torch.nn.functional as F
from torch.optim import SGD, Adam
from sklearn.datasets import load_iris
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
import numpy as np
from keras.utils import np_utils


def one_hot_encode(array):
    """One-Hot encode a numpy array of (say) strings"""
    uniques, ids = np.unique(array, return_inverse=True)
    return np_utils.to_categorical(ids, len(uniques))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4, 16)
        self.fc2 = nn.Linear(16, 3)

    def forward(self, x):
        return F.log_softmax(self.fc2(F.sigmoid(self.fc1(x))), dim=1)


def get_data_loaders(train_batch_size, val_batch_size, test_batch_size):
    iris_data = load_iris()
    # iris_data.target = one_hot_encode(iris_data.target)

    X_train, X_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size=0.33)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1)



    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)),
                              batch_size=train_batch_size)
    val_loader = DataLoader(TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val)), batch_size=val_batch_size)
    test_loader = DataLoader(TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test)),
                             batch_size=test_batch_size)

    return train_loader, val_loader, test_loader


def create_summary_writer(model, data_loader, log_dir):
    writer = SummaryWriter(log_dir=log_dir)

    data_loader_iter = iter(data_loader)
    x, y = next(data_loader_iter)
    try:
        writer.add_graph(model, x)
    except Exception as e:
        print("Failed to save model graph: {}".format(e))
    return writer


def run(train_batch_size, val_batch_size, test_batch_size, epochs, lr, momentum, log_interval, log_dir):
    train_loader, val_loader, test_loader = get_data_loaders(train_batch_size, val_batch_size, test_batch_size)
    model = Net()
    writer = create_summary_writer(model, train_loader, log_dir)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    optimizer = Adam(model.parameters(), lr=lr)
    loss = F.cross_entropy

    trainer = create_supervised_trainer(model, optimizer, loss, device=device)
    evaluator = create_supervised_evaluator(model, device=device, metrics={
        'accuracy': Accuracy(),
        'loss': Loss(loss)
    })

    start = time.time()

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter = (engine.state.iteration - 1) % len(train_loader) + 1
        if iter % log_interval == 0:
            print("Epoch[{}] Iteration[{}/{}] Loss: {:.2f}"
                  .format(engine.state.epoch, iter, len(train_loader), engine.state.output))
            writer.add_scalar("training/loss", engine.state.output, engine.state.iteration)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_loss = metrics['loss']

        print("Epoch: [{}] Training   Avg accuracy: {:.2f} Avg loss: {:.2f}"
              .format(engine.state.epoch, avg_accuracy, avg_loss))
        writer.add_scalar("training/avg_loss", avg_loss, engine.state.epoch)
        writer.add_scalar("training/avg_accuracy", avg_accuracy, engine.state.epoch)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_loss = metrics['loss']
        print("Epoch: [{}] Validation Avg accuracy: {:.2f} Avg loss: {:.2f}"
              .format(engine.state.epoch, avg_accuracy, avg_loss))
        writer.add_scalar("validation/avg_loss", avg_loss, engine.state.epoch)
        writer.add_scalar("validation/avg_accuracy", avg_accuracy, engine.state.epoch)

    @trainer.on(Events.COMPLETED)
    def log_validation_results(engine):
        evaluator.run(test_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_loss = metrics['loss']
        # print("Test Results - Avg accuracy: {:.2f} Avgloss: {:.2f}"
        print(
            "Time = {:.2f} Loss = {:.2f} Accuracy = {:.2f}".format(time.time() - start, avg_loss, avg_accuracy))
        writer.add_scalar("test/avg_loss", avg_loss)
        writer.add_scalar("test/avg_accuracy", avg_accuracy)

    trainer.run(train_loader, max_epochs=epochs)
    print("TensorBoardX output at: "  + log_dir)
    writer.close()


if __name__ == '__main__':
    # logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--val_batch_size', type=int, default=1,
                        help='input batch size for validation (default: 1000)')
    parser.add_argument('--test_batch_size', type=int, default=1,
                        help='input batch size for test (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='how many batches to wait before logging training status')
    parser.add_argument("--log_dir", type=str, default="tmp/pytorch_iris_logs_" + str(time.time()),
                        help="log directory for Tensorboard log output")

    args = parser.parse_args()
    print(args.log_dir)

    run(args.batch_size, args.val_batch_size, args.test_batch_size, args.epochs, args.lr, args.momentum,
        args.log_interval, args.log_dir)
