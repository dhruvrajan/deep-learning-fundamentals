import time

import torch.nn as nn
from ignite.engine import Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Accuracy, Loss
from tensorboardX import SummaryWriter
from torch import optim
from torch.utils.data import DataLoader


def create_summary_writer(model, data_loader, log_dir):
    writer = SummaryWriter(log_dir=log_dir)

    data_loader_iter = iter(data_loader)
    x, y, _ = next(data_loader_iter)
    try:
        writer.add_graph(model, x)
    except Exception as e:
        print("Failed to save model graph: {}".format(e))
    return writer


def train_sentiment_model(model, loss_function, optimizer, create_trainer, create_evaluator, train_data, test_data, args):
    train_tensors = train_data.create_tensor_dataset(args.batch_size)
    test_tensors = test_data.create_tensor_dataset(args.val_batch_size)

    print(args.batch_size ,args.val_batch_size)

    train_loader = DataLoader(train_tensors, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(test_tensors, batch_size=args.val_batch_size, shuffle=True)

    summary_writer = create_summary_writer(model, train_loader, args.log_dir)
    model_checkpoint = ModelCheckpoint("tmp/models", args.run_id, save_interval=args.checkpoint_interval, n_saved=args.n_checkpoints)
    trainer = create_trainer(model, optimizer, loss_function, args)
    evaluator = create_evaluator(model, {
        "accuracy": Accuracy(),
        "loss": Loss(loss_function)
    }, args)

    start = time.time()

    trainer.add_event_handler(Events.EPOCH_COMPLETED, model_checkpoint, {"model": model})

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
