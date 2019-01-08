import time

import torch
import torch.nn as nn
from argparse import ArgumentParser
from ignite.engine import Engine, Events
from ignite.metrics import Accuracy, Loss
from tensorboardX import SummaryWriter
from torch import optim
from torch.utils.data import DataLoader

from sentiment_analysis.models import SimpleLSTM
from sentiment_analysis.sentiment_utils import load_sentiment_data, SentimentDataset, WordVectors


def create_summary_writer(model, data_loader, log_dir):
    writer = SummaryWriter(log_dir=log_dir)

    data_loader_iter = iter(data_loader)
    x, y, _ = next(data_loader_iter)
    try:
        writer.add_graph(model, x)
    except Exception as e:
        print("Failed to save model graph: {}".format(e))
    return writer


def create_trainer(model: nn.Module, optimizer, loss_function, args):
    def _training_step(engine, batch):
        model.train()
        model.zero_grad()
        optimizer.zero_grad()

        X, y, _ = batch
        model.hidden_state = model.init_hidden(model.hidden_size, X.shape[0])
        log_probs = model(X)
        loss = loss_function(log_probs, y)

        loss.backward()
        optimizer.step()
        return loss.item()

    return Engine(_training_step)


def create_evaluator(model: nn.Module, metrics: dict, args):
    def _inference_step(engine, batch):
        model.eval()
        with torch.no_grad():
            X, y, _ = batch

            # model.hidden_state = model.init_hidden(model.hidden_size, X.shape[0])

            log_probs = model(X)
            return log_probs, y

    engine = Engine(_inference_step)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def train_simple_lstm(
        train_data: SentimentDataset,
        test_data: SentimentDataset,
        word_vectors: WordVectors,
        args):
    train_tensors = train_data.create_tensor_dataset(args.batch_size)
    test_tensors = test_data.create_tensor_dataset(args.val_batch_size)

    train_loader = DataLoader(train_tensors, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(test_tensors, batch_size=args.val_batch_size, shuffle=True)

    model = SimpleLSTM(word_vectors, args.embedding_size, 10, 2, 0.2, False, args)

    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    summary_writer = create_summary_writer(model, train_loader, args.log_dir)
    # model_checkpoint = ModelCheckpoint("tmp/models", args.run_id, save_interval=args.checkpoint_interval, n_saved=args.n_checkpoints)
    trainer = create_trainer(model, optimizer, loss_function, args)
    evaluator = create_evaluator(model, {
        "accuracy": Accuracy(),
        "loss": Loss(loss_function)
    }, args)

    start = time.time()

    # trainer.add_event_handler(Events.EPOCH_COMPLETED, model_checkpoint, {"model": model})

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics["accuracy"]
        avg_loss = metrics["loss"]

        print("Epoch: [{}] Training   Avg accuracy: {:.2f} Avg loss: {:.2f}"
              .format(engine.state.epoch, avg_accuracy, avg_loss))
        summary_writer.add_scalar("training/avg_loss", avg_loss, engine.state.epoch)
        summary_writer.add_scalar("training/avg_accuracy", avg_accuracy, engine.state.epoch)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics["accuracy"]
        avg_loss = metrics["loss"]
        print("Epoch: [{}] Validation Avg accuracy: {:.2f} Avg loss: {:.2f}".format(engine.state.epoch, avg_accuracy,
                                                                                    avg_loss))
        summary_writer.add_scalar("validation/avg_loss", avg_loss, engine.state.epoch)
        summary_writer.add_scalar("validation/avg_accuracy", avg_accuracy, engine.state.epoch)

    @trainer.on(Events.COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics["accuracy"]
        avg_loss = metrics["loss"]
        print("Time = {:.2f} Loss = {:.2f} Accuracy = {:.2f}".format(time.time() - start, avg_loss, avg_accuracy))
        summary_writer.add_scalar("test/avg_loss", avg_loss)
        summary_writer.add_scalar("test/avg_accuracy", avg_accuracy)

    trainer.run(train_loader, max_epochs=args.epochs)

    run_summary = {
        "model_summary": model.summary,
        "optimizer_summary": {
            "type": "Adam",
            "learning_rate": args.lr
        },
        "run_params": {
            "epochs": args.epochs,
            "train_batch_size": args.batch_size

        },
        "args": args.__dict__
    }

    for key in run_summary:
        print(key, str(run_summary[key]))
        summary_writer.add_text(key, str(run_summary[key]))

    summary_writer.close()
    return model


if __name__ == "__main__":
    time_label = str(time.time())
    prefix = "pytorch_sentiment_logs"

    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64,
                        help="input batch size for training (default: 64)")
    parser.add_argument("--val_batch_size", type=int, default=64,
                        help="input batch size for validation (default: 1000)")
    parser.add_argument("--test_batch_size", type=int, default=1,
                        help="input batch size for test (default: 1000)")
    parser.add_argument("--epochs", type=int, default=20,
                        help="number of epochs to train (default: 10)")
    parser.add_argument("--lr", type=float, default=0.0001,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--embedding_size", type=int, default=300,
                        help="word embedding vector size")
    parser.add_argument("--freeze", type=bool, default=False    ,
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


    args = parser.parse_args()
    print("TensorboardX output at: ", args.log_dir)

    train_data, test_data, word_vectors = load_sentiment_data(args.embedding_size)
    trained_model = train_simple_lstm(train_data, test_data, word_vectors, args)
    print("TensorboardX output at: ", args.log_dir)
