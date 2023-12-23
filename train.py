import os
import shutil
import argparse
import yaml
import time
import torch
import torch.nn as nn
import torch.utils.data
from rnnt.model import Transducer
from rnnt.optim import Optimizer, optimizer_to
from rnnt.data import WordDataLoader, SentenceDataLoader
from rnnt.utils import AttrDict, init_logger, shuffle_csv, count_parameters, save_model, computer_cer, init_parameters
from rnnt.tokenizer import CharTokenizer
from tensorboardX import SummaryWriter



def train(epoch, config, model, training_data, optimizer, logger, visualizer=None):
    model.train()
    start_epoch = time.process_time()
    total_loss = 0
    optimizer.epoch()
    batch_steps = len(training_data)
    for step, (inputs, inputs_length, targets, targets_length) in enumerate(
        training_data
    ):
        inputs = torch.squeeze(inputs, dim=1)
        if config.training.num_gpu > 0:
            inputs, inputs_length = inputs.to(config.model.device), inputs_length.to(config.model.device)
            targets, targets_length = targets.to(config.model.device), targets_length.to(config.model.device)


        optimizer.zero_grad()
        start = time.process_time()
        loss = model(inputs, inputs_length, targets, targets_length)
        loss.backward()

        total_loss += loss.item()

        grad_norm = nn.utils.clip_grad_norm_(
            model.parameters(), config.training.max_grad_norm
        )

        optimizer.step()

        avg_loss = total_loss / (step + 1)
        if optimizer.global_step % config.training.show_interval == 0:
            end = time.process_time()
            process = step / batch_steps * 100
            logger.info(
                "-Training-Epoch:%d(%.5f%%), Global Step:%d, Learning Rate:%.6f, Grad Norm:%.5f, Loss:%.5f, "
                "AverageLoss: %.5f, Run Time:%.3f"
                % (
                    epoch,
                    process,
                    optimizer.global_step,
                    optimizer.lr,
                    grad_norm,
                    loss.item(),
                    avg_loss,
                    end - start,
                )
            )

    if visualizer is not None:
        visualizer.add_scalar("avg_train_loss", avg_loss, epoch)
        visualizer.add_scalar('learn_rate', optimizer.lr, epoch)   
        end_epoch = time.process_time()
    logger.info(
        "-Training-Epoch:%d, Average Loss: %.5f, Epoch Time: %.3f"
        % (epoch, total_loss / (step + 1), end_epoch - start_epoch)
    )


def eval(
    epoch, config, model, validating_data, logger, special_tokens, visualizer=None, is_test_data=False
):
    model.eval()
    total_loss = 0
    total_dist = 0
    total_word = 0
    batch_steps = len(validating_data)

    for step, (inputs, inputs_length, targets, targets_length) in enumerate(
        validating_data
    ):
        if config.training.num_gpu > 0 and config.model.device == "cuda":
            inputs, inputs_length = inputs.cuda(), inputs_length.cuda()
            targets, targets_length = targets.cuda(), targets_length.cuda()

        preds = model.recognize(inputs, inputs_length)

        transcripts = [
            targets.cpu().numpy()[i][: targets_length[i].item()]
            for i in range(targets.size(0))
        ]
        preds = [[elem for elem in sublist if elem not in special_tokens] for sublist in preds]
        transcripts = [[elem for elem in sublist if elem not in special_tokens] for sublist in transcripts]

        dist, num_words = computer_cer(preds, transcripts)
        total_dist += dist
        total_word += num_words
        cer = total_dist / total_word * 100

        if step % config.training.show_interval == 0:
            process = step / batch_steps * 100
            logger.info(
                "-Validation-Epoch:%d(%.5f%%), CER: %.5f %%" % (epoch, process, cer)
            )

    val_loss = total_loss / (step + 1)
    logger.info(
        "-Validation-Epoch:%4d, AverageLoss:%.5f, AverageCER: %.5f %%"
        % (epoch, val_loss, cer)
    )

    if visualizer is not None:
        var_name = "cer_test" if is_test_data else "cer_train"
        visualizer.add_scalar(var_name, cer, epoch)


def get_tokenizer():
    tokenizer = CharTokenizer()
    tokenizer = tokenizer.load_tokenizer("files/tokenizer.json")
    return tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-config", type=str, default="config/aishell.yaml")
    parser.add_argument("-log", type=str, default="train.log")
    parser.add_argument("-mode", type=str, default="retrain")
    opt = parser.parse_args()

    configfile = open(opt.config)
    config = AttrDict(yaml.load(configfile, Loader=yaml.FullLoader))

    exp_name = os.path.join( config.data.name, config.training.save_model)
    if not os.path.isdir(exp_name):
        os.makedirs(exp_name)
    logger = init_logger(os.path.join(exp_name, opt.log))

    shutil.copyfile(opt.config, os.path.join(exp_name, "config.yaml"))
    logger.info("Save config info.")

    tokenizer = get_tokenizer()

    if config.training.num_gpu > 0 and config.model.device == "cuda":
        torch.cuda.manual_seed(config.training.seed)
        torch.backends.cudnn.deterministic = True
    else:
        torch.manual_seed(config.training.seed)
    logger.info("Set random seed: %d" % config.training.seed)

    model = Transducer(config.model)

    optimizer = Optimizer(model.parameters(), config.optim)
    logger.info("Created a %s optimizer." % config.optim.type)

    if config.training.load_model:
        checkpoint = torch.load(config.training.load_model)
        model.encoder.load_state_dict(checkpoint["encoder"])
        model.decoder.load_state_dict(checkpoint["decoder"])
        model.joint.load_state_dict(checkpoint["joint"])
        # optimizer.load_state_dict(checkpoint["optimizer"])
        # optimizer.global_step = checkpoint["step"]
        # optimizer.current_epoch = checkpoint["epoch"]
        # optimizer_to(optimizer.optimizer, 'cuda')

        logger.info("Loaded model from %s" % config.training.load_model)
    else:
        init_parameters(model, type='xnormal')

    if config.training.num_gpu > 0 and config.model.device == "cuda":
        model = model.cuda()
        if config.training.num_gpu > 1:
            device_ids = list(range(config.training.num_gpu))
            model = torch.nn.DataParallel(model, device_ids=device_ids)
        logger.info("Loaded the model to %d GPUs" % config.training.num_gpu)

    n_params, enc, dec = count_parameters(model)
    logger.info("# the number of parameters in the whole model: %d" % n_params)
    logger.info("# the number of parameters in the Encoder: %d" % enc)
    logger.info("# the number of parameters in the Decoder: %d" % dec)
    logger.info(
        "# the number of parameters in the JointNet: %d" % (n_params - dec - enc)
    )

    if opt.mode == "continue":
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]
        logger.info("Load Optimizer State!")
    else:
        start_epoch = 0

    if config.training.visualization:
        visualizer = SummaryWriter(os.path.join(exp_name, "log"))
        logger.info("Created a visualizer.")
    else:
        visualizer = None

    validate_data = SentenceDataLoader(
        "files/test_sentences_val.csv", tokenizer, config.data, config.data.batch_size
    )
    training_data = SentenceDataLoader("files/asd.csv", tokenizer, config.data, config.data.batch_size)
    train_eval_data = SentenceDataLoader(
        "files/asd.csv", tokenizer, config.data, config.data.batch_size
    )
    sos_idx = tokenizer.special_tokens["sos"][1]
    eos_idx = tokenizer.special_tokens["eos"][1]
    special_tokens = [sos_idx, eos_idx]
    
    for epoch in range(start_epoch, config.training.epochs):
        shuffled_file_path = shuffle_csv("files/asd.csv")
        training_data = SentenceDataLoader(shuffled_file_path, tokenizer, config.data, config.data.batch_size)

        train(epoch, config, model, training_data, optimizer, logger, visualizer)

        if (
            config.training.eval_or_not
            and epoch % config.training.show_interval == 0
        ):
            eval(epoch, config, model, train_eval_data, logger, special_tokens, visualizer, is_test_data=False)
            #eval(epoch, config, model, validate_data, logger, special_tokens, visualizer, is_test_data=True)

        # import pandas as pd
        # data = pd.read_csv("files/train_sentences.csv")
        # shuffled_data = data.sample(frac=1).reset_index(drop=True)
        # shuffled_data.to_csv("files/train_sentences.csv", index=False)

        save_name = os.path.join(
            exp_name, "%s.epoch%d.chkpt" % (config.training.save_model, epoch)
        )
        if epoch % config.training.save_every == 0:
            save_model(model, optimizer, config, save_name)
        logger.info("Epoch %d model has been saved." % epoch)

        if epoch >= config.optim.begin_to_adjust_lr and epoch % config.optim.adjust_every == 0:
            optimizer.decay_lr()
            if optimizer.lr < 1e-6:
                logger.info("The learning rate is too low to train.")
                break
            logger.info("Epoch %d update learning rate: %.6f" % (epoch, optimizer.lr))

    logger.info("The training process is OVER!")


if __name__ == "__main__":
    main()
