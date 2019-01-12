import time

import torch
import torch.nn.functional as F
from argparse import ArgumentParser
from ignite.engine import create_supervised_trainer, create_supervised_evaluator, Events
from ignite.metrics import Accuracy, Loss
from tensorboardX import SummaryWriter
from torch.optim import Adam
from torch.utils.data import DataLoader

from sentiment_analysis.pytorch.models import SimpleLSTM
from sentiment_analysis.sentiment_data import load_sentiment_data


def create_summary_writer(model, data_loader, log_dir):
    writer = SummaryWriter(log_dir=log_dir)

    data_loader_iter = iter(data_loader)
    x, y = next(data_loader_iter)
    try:
        writer.add_graph(model, x)
    except Exception as e:
        print("Failed to save model graph: {}".format(e))
    return writer


def train_ffnn_ignite(train_data, val_data, word_vectors, args):
    train_tensors = train_data.create_tensor_dataset()
    val_tensors = val_data.create_tensor_dataset()

    train_loader = DataLoader(train_tensors, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_tensors, batch_size=args.val_batch_size, shuffle=False)

    embedding_size = word_vectors.vectors.shape[1]
    # model = SimpleFFNN(word_vectors, embedding_size, 8, 4, 1, 2, freeze=args.freeze)
    model = SimpleLSTM(word_vectors, embedding_size, 10, 2, lstm_dropout=0.2, bidirectional=False, freeze=args.freeze)

    writer = create_summary_writer(model, train_loader, args.log_dir)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    optimizer = Adam(model.parameters(), lr=args.lr)
    loss = F.cross_entropy

    trainer = create_supervised_trainer(model, optimizer, loss, device=device)
    evaluator = create_supervised_evaluator(model, device=device, metrics={
        'accuracy': Accuracy(),
        'loss': Loss(loss)
    })

    start = time.time()

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
        print("Epoch: [{}] Validation Avg accuracy: {:.2f} Avg loss: {:.2f}" .format(engine.state.epoch, avg_accuracy, avg_loss))
        # print("Epoch: [{}] accuracy: (train: {:.2f}, val: {:.2f}) loss: (train: {:.2f}, val: {:.2f})" .format(engine.state.epoch, avg_accuracy, avg_accuracy2, avg_loss, avg_loss2))
        writer.add_scalar("validation/avg_loss", avg_loss, engine.state.epoch)
        writer.add_scalar("validation/avg_accuracy", avg_accuracy, engine.state.epoch)

    @trainer.on(Events.COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_loss = metrics['loss']
        # print("Test Results - Avg accuracy: {:.2f} Avgloss: {:.2f}"
        print("Time = {:.2f} Loss = {:.2f} Accuracy = {:.2f}".format(time.time() - start, avg_loss, avg_accuracy))
        writer.add_scalar("test/avg_loss", avg_loss)
        writer.add_scalar("test/avg_accuracy", avg_accuracy)

    trainer.run(train_loader, max_epochs=args.epochs)
    print("TensorBoardX output at: "  + args.log_dir)
    writer.close()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=100,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--val_batch_size', type=int, default=1,
                        help='input batch size for validation (default: 1000)')
    parser.add_argument('--epochs', type=int, default=20,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.0005)')
    parser.add_argument('--freeze', type=bool, default=True,
                        help='freeze embedding layer (default: True)')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='how many batches to wait before logging training status')
    parser.add_argument("--log_dir", type=str, default="tmp/pytorch_sentiment_logs_" + str(time.time()),
                        help="log directory for Tensorboard log output")

    args = parser.parse_args()
    train_data, test_data, word_vectors = load_sentiment_data(300)
    train_ffnn_ignite(train_data, test_data, word_vectors, args)