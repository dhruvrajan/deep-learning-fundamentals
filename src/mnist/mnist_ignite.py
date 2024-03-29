from __future__ import print_function

import time

from argparse import ArgumentParser
import torch
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
from torch.optim import SGD, Adam
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize

try:
    from tensorboardX import SummaryWriter
except ImportError:
    raise RuntimeError("No tensorboardX package is found. Please install with the command: \npip install tensorboardX")

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(p=0.2)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)


def get_data_loaders(train_batch_size, val_batch_size):
    data_transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

    train_loader = DataLoader(MNIST(download=True, root=".", transform=data_transform, train=True),
                              batch_size=train_batch_size, shuffle=True)

    val_loader = DataLoader(MNIST(download=False, root=".", transform=data_transform, train=False),
                            batch_size=val_batch_size, shuffle=False)
    return train_loader, val_loader


def create_summary_writer(model, data_loader, log_dir):
    writer = SummaryWriter(log_dir=log_dir)
    data_loader_iter = iter(data_loader)
    x, y = next(data_loader_iter)
    try:
        writer.add_graph(model, x)
    except Exception as e:
        print("Failed to save model graph: {}".format(e))
    return writer


def run(train_batch_size, val_batch_size, epochs, lr, momentum, log_interval, log_dir):
    print("TensorboardX Output at:", log_dir)
    train_loader, val_loader = get_data_loaders(train_batch_size, val_batch_size)
    model = Net()
    writer = create_summary_writer(model, train_loader, log_dir)
    device = 'cpu'

    if torch.cuda.is_available():
        device = 'cuda'

    optimizer = Adam(model.parameters(), lr=lr)
    trainer = create_supervised_trainer(model, optimizer, F.cross_entropy, device=device)
    evaluator = create_supervised_evaluator(model,
                                            metrics={'accuracy': Accuracy(),
                                                     'nll': Loss(F.nll_loss)},
                                            device=device)

    start = time.time()

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter = (engine.state.iteration - 1) % len(train_loader) + 1
        if iter % log_interval == 0:
            print("Epoch[{}] Iteration[{}/{}] Loss: {:.4f}"
                  "".format(engine.state.epoch, iter, len(train_loader), engine.state.output))
            writer.add_scalar("training/loss", engine.state.output, engine.state.iteration)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_nll = metrics['nll']
        print("Training Results - Epoch: {}  Avg accuracy: {:.4f} Avg loss: {:.4f}"
              .format(engine.state.epoch, avg_accuracy, avg_nll))
        writer.add_scalar("training/avg_loss", avg_nll, engine.state.epoch)
        writer.add_scalar("training/avg_accuracy", avg_accuracy, engine.state.epoch)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_nll = metrics['nll']
        print("Validation Results - Epoch: {}  Avg accuracy: {:.4f} Avg loss: {:.4f}"
              .format(engine.state.epoch, avg_accuracy, avg_nll))
        writer.add_scalar("valdation/avg_loss", avg_nll, engine.state.epoch)
        writer.add_scalar("valdation/avg_accuracy", avg_accuracy, engine.state.epoch)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_nll = metrics['nll']
        print("Final Validation Results - Time: {} Epoch: {}  Avg accuracy: {:.4f} Avg loss: {:.4f}"
              .format(time.time() - start, engine.state.epoch, avg_accuracy, avg_nll))
        writer.add_scalar("final_valdation/avg_loss", avg_nll, engine.state.epoch)
        writer.add_scalar("final_valdation/avg_accuracy", avg_accuracy, engine.state.epoch)

    # kick everything off
    trainer.run(train_loader, max_epochs=epochs)

    writer.close()
    print("TensorboardX Output at:", log_dir)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--val_batch_size', type=int, default=1000,
                        help='input batch size for validation (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='how many batches to wait before logging training status')
    parser.add_argument("--log_dir", type=str, default="tmp/pytorch_mnist_logs_" + str(time.time()),
                        help="log directory for Tensorboard log output")

    args = parser.parse_args()

    run(args.batch_size, args.val_batch_size, args.epochs, args.lr, args.momentum,
        args.log_interval, args.log_dir)
