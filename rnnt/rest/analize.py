import os
import shutil
import argparse
import yaml
import torch
import torch.utils.data
from model import Transducer
from data import SentenceDataLoader
from utils import AttrDict, init_logger, computer_cer
from tokenizer import CharTokenizer

def eval(
    config, model, validating_data, special_tokens, tokenizer
):
    model.eval()
    total_dist = 0
    total_word = 0
    cer_table = []
    for _, (inputs, inputs_length, targets, targets_length) in enumerate(
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
        preds = tokenizer.ids2tokens(preds)
        transcripts = tokenizer.ids2tokens(transcripts)
        pred_words = [pred.split('_') for pred in preds]
        trans_words = [t.split('_') for t in transcripts]
        cer_table = []
        for p, t in zip(pred_words, trans_words):
            for ap, at in zip(p, t):
                ap = tokenizer.tokens2ids(ap)
                at = tokenizer.tokens2ids(at)

                dist, num_words = computer_cer(ap, at)
                total_dist += dist
                total_word += num_words
                cer = total_dist / total_word * 100
                cer_table.append(cer)
    return cer_table

def eval_wrong(
    config, model, validating_data, special_tokens, other_data, tokenizer
):
    model.eval()
    total_dist = 0
    total_word = 0
    cer_table = []
    for (inputs, inputs_length, targets, targets_length), (inputs1, inputs_length1, targets1, targets_length1) in zip(
        validating_data, other_data
    ):
        if config.training.num_gpu > 0 and config.model.device == "cuda":
            inputs, inputs_length = inputs.cuda(), inputs_length.cuda()
            targets, targets_length = targets.cuda(), targets_length.cuda()
            inputs1, inputs_length1 = inputs1.cuda(), inputs_length1.cuda()
            targets1, targets_length1 = targets1.cuda(), targets_length1.cuda()

        preds = model.recognize(inputs, inputs_length)

        transcripts = [
            targets1.cpu().numpy()[i][: targets_length1[i].item()]
            for i in range(targets1.size(0))
        ]
        original = [
            targets.cpu().numpy()[i][: targets_length[i].item()]
            for i in range(targets.size(0))
        ]
        preds = [[elem for elem in sublist if elem not in special_tokens] for sublist in preds]
        transcripts = [[elem for elem in sublist if elem not in special_tokens] for sublist in transcripts]
        original = [[elem for elem in sublist if elem not in special_tokens] for sublist in original]
        transcripts = [t for t in transcripts if t not in original] 

        preds = tokenizer.ids2tokens(preds)
        transcripts = tokenizer.ids2tokens(transcripts)
        pred_words = [pred.split('_') for pred in preds]
        trans_words = [t.split('_') for t in transcripts]
        for preds, transcripts in zip(pred_words, trans_words):
            for p in preds:
                max_p = 100
                for t in transcripts:   
                    dist, num_words = computer_cer(p, t)
                    total_dist += dist
                    total_word += num_words
                    cer = total_dist / total_word * 100
                    max_p = min(max_p, cer)
                cer_table.append(max_p)
    return cer_table

def get_tokenizer():
    tokenizer = CharTokenizer()
    tokenizer = tokenizer.load_tokenizer("files/tokenizer.json")
    return tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-config", type=str, default="config/config.yaml")
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

    if config.training.load_model:
        checkpoint = torch.load(config.training.load_model)
        model.encoder.load_state_dict(checkpoint["encoder"])
        model.decoder.load_state_dict(checkpoint["decoder"])
        model.joint.load_state_dict(checkpoint["joint"])

        logger.info("Loaded model from %s" % config.training.load_model)


    if config.training.num_gpu > 0 and config.model.device == "cuda":
        model = model.cuda()
        if config.training.num_gpu > 1:
            device_ids = list(range(config.training.num_gpu))
            model = torch.nn.DataParallel(model, device_ids=device_ids)
        logger.info("Loaded the model to %d GPUs" % config.training.num_gpu)

    validate_data = SentenceDataLoader(
        "files/core_test_set.csv", tokenizer, config.data, config.data.batch_size
    )
    training_data = SentenceDataLoader("files/core_train_subset.csv", tokenizer, config.data, config.data.batch_size)

    sos_idx = tokenizer.special_tokens["sos"][1]
    eos_idx = tokenizer.special_tokens["eos"][1]
    pad_idx = tokenizer.special_tokens["pad"][1]
    special_tokens = [sos_idx, eos_idx, pad_idx]

    cer_table_correct = eval(config, model, training_data, special_tokens)
    cer_table_wrong = eval_wrong(config, model, training_data, special_tokens, validate_data)
    print(cer_table_correct, cer_table_wrong)