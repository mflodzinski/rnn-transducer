from logging import Logger
import time

from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch

from data import DataLoader
from model import Transducer
from optim import Optimizer
from tokenizer import CharTokenizer
from utils import AttrDict
from typing import Union
import utils


def train(
    epoch: int,
    config: AttrDict,
    model: Transducer,
    train_data: DataLoader,
    optimizer: Optimizer,
    logger: Logger,
    device: Union[str, torch.device],
    visualizer: SummaryWriter = None,
):
    model.train()
    start_epoch_time = time.process_time()
    total_loss = 0
    batch_steps = len(train_data)

    optimizer.epoch()

    for step, (inputs, inputs_length, targets, targets_length) in enumerate(train_data):
        inputs, inputs_length, targets, targets_length = (
            inputs.to(device),
            inputs_length.to(device),
            targets.to(device),
            targets_length.to(device),
        )

        optimizer.zero_grad()
        start_step_time = time.process_time()

        loss = model(inputs, inputs_length, targets, targets_length)
        loss = loss.mean()
        loss.backward()
        total_loss += loss.item()

        grad_norm = nn.utils.clip_grad_norm_(
            model.parameters(), config.training.max_grad_norm
        )

        optimizer.step()
        avg_loss = total_loss / (step + 1)

        if optimizer.global_step % config.training.show_every == 0:
            end_step_time = time.process_time()
            progress = (step / batch_steps) * 100
            logger.info(
                f"-Training-Epoch:{epoch}({progress:.5f}%), Global Step:{optimizer.global_step}, "
                f"Learning Rate:{optimizer.lr:.6f}, Grad Norm:{grad_norm:.5f}, Loss:{loss.item():.5f}, "
                f"AverageLoss:{avg_loss:.5f}, Run Time:{end_step_time - start_step_time:.3f}"
            )
    utils.add_gaussian_noise(model, device)
    if visualizer is not None:
        visualizer.add_scalar("avg_train_loss", avg_loss, epoch)

    end_epoch_time = time.process_time()
    logger.info(
        f"-Training-Epoch:{epoch}, Average Loss: {avg_loss:.5f}, Epoch Time: {end_epoch_time - start_epoch_time:.3f}"
    )


def eval(
    epoch: int,
    model: Transducer,
    test_data: DataLoader,
    logger: Logger,
    special_tokens: list[int],
    device: Union[str, torch.device],
    visualizer: SummaryWriter = None,
    is_test_data: bool = False,
):
    model.eval()
    total_loss = 0
    total_dist = 0
    total_word = 0
    batch_steps = len(test_data)

    for step, (inputs, inputs_length, targets, targets_length) in enumerate(test_data):
        inputs, inputs_length, targets, targets_length = (
            inputs.to(device),
            inputs_length.to(device),
            targets.to(device),
            targets_length.to(device),
        )

        predictions = model.recognize(inputs, inputs_length)

        transcripts = [
            targets.cpu().numpy()[i][: targets_length[i].item()]
            for i in range(targets.size(0))
        ]

        predictions = utils.remove_special_tokens(predictions, special_tokens)
        transcripts = utils.remove_special_tokens(transcripts, special_tokens)

        dist, num_words = utils.compute_cer(predictions, transcripts)
        total_dist += dist
        total_word += num_words

        process = step / batch_steps * 100
        cer = total_dist / total_word * 100
        logger.info(f"-Test-Epoch:{epoch}({process:.5f}%), CER: {cer:.5f}%")

    val_loss = total_loss / (step + 1)
    cer = total_dist / total_word * 100
    logger.info(
        f"-Test-Epoch:{epoch:4d}, AverageLoss:{val_loss:.5f}, AverageCER: {cer:.5f}%"
    )

    if visualizer is not None:
        var_name = "cer_test" if is_test_data else "cer_train"
        visualizer.add_scalar(var_name, cer, epoch)


def train_model(
    config: AttrDict,
    model: Transducer,
    optimizer: Optimizer,
    train_data: DataLoader,
    val_data: DataLoader,
    test_data: DataLoader,
    logger: Logger,
    device: Union[str, torch.device],
    visualizer: SummaryWriter,
    tokenizer: CharTokenizer,
):
    special_tokens = [tokenizer.stoi[i] for i in tokenizer.special_tokens.values()]

    for epoch in range(config.training.epochs):
        train_data.shuffle()
        train(epoch, config, model, train_data, optimizer, logger, device, visualizer)

        if config.training.evaluate and epoch % config.training.save_every == 0:
            eval(
                epoch,
                model,
                val_data,
                logger,
                special_tokens,
                device,
                visualizer,
                is_test_data=False,
            )
            eval(
                epoch,
                model,
                test_data,
                logger,
                special_tokens,
                device,
                visualizer,
                is_test_data=True,
            )

        if epoch % config.training.save_every == 0:
            utils.save_model_checkpoint(model, epoch, config, logger)

        utils.adjust_learning_rate(optimizer, epoch, config, logger)

    logger.info("The training process is OVER!")


def main():
    CONFIG_PATH = "config/config.yaml"
    #torch.set_float32_matmul_precision("high")

    config = utils.load_config(CONFIG_PATH)
    logger = utils.setup_logger(config)
    visualizer = utils.create_visualizer(config)
    device = utils.setup_device(logger)

    train_data, test_data, val_data, tokenizer = utils.prepare_data_loaders(config)
    model = utils.initialize_model(config, tokenizer.vocab_size, device)
    optimizer = utils.create_optimizer(model, config.optim)
    utils.log_model_parameters(model, logger)

    train_model(
        config=config,
        model=model,
        optimizer=optimizer,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        logger=logger,
        device=device,
        visualizer=visualizer,
        tokenizer=tokenizer,
    )


if __name__ == "__main__":
    main()
