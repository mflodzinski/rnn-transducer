from tensorboard import notebook

# Provide the path to the directory containing the TensorBoard logs
log_dir = 'egs/aishell/exp/4blstm_320henc_1l512dec/log'  # Replace with your log directory

# Launch TensorBoard
notebook.start("--logdir=" + log_dir)

# import torch
# import os
# import shutil
# import argparse
# import yaml
# import time
# import torch
# import torch.nn as nn
# import torch.utils.data
# from model import Transducer
# from optim import Optimizer
# from data import TrainDataLoader, TestDataLoader
# from utils import AttrDict, init_logger, count_parameters, save_model, computer_cer
# from tokenizer import CharTokenizer
# from tqdm import tqdm
# from utils import init_parameters
# from tensorboardX import SummaryWriter
# from torchaudio.models import RNNT
# from torchaudio.models.rnnt_decoder import RNNTBeamSearch

# parser = argparse.ArgumentParser()
# parser.add_argument('-config', type=str, default='config/aishell.yaml')
# parser.add_argument('-log', type=str, default='train.log')
# parser.add_argument('-mode', type=str, default='retrain')
# opt = parser.parse_args()

# configfile = open(opt.config)
# config = AttrDict(yaml.load(configfile, Loader=yaml.FullLoader))
# model = Transducer(config.model)

# checkpoint = torch.load(config.training.load_model)
# model.encoder.load_state_dict(checkpoint['encoder'])
# model.decoder.load_state_dict(checkpoint['decoder'])
# model.joint.load_state_dict(checkpoint['joint'])

# joint = model.joint
# encoder = model.encoder
# decoder = model.decoder

# new_model = RNNT(transcriber=encoder, predictor=decoder, joiner=joint)
# new_model_search = RNNTBeamSearch(model=new_model.cuda(), blank=27)


# def get_tokenizer():
#     tokenizer = CharTokenizer()
#     tokenizer = tokenizer.load_tokenizer('files/tokenizer.json')
#     return tokenizer



# def eval_train(epoch, config, model: RNNTBeamSearch, training_data, logger, visualizer=None):
#     model.model.transcriber.lstm.dropout = 0
#     model.model.predictor.lstm.dropout = 0
#     model.model.eval()
#     total_loss = 0
#     total_dist = 0
#     total_word = 0
#     batch_steps = len(training_data)


#     for step, (inputs, inputs_length, targets, targets_length) in enumerate(training_data):
#         if config.training.num_gpu > 0:
#             inputs, inputs_length = inputs.cuda(), inputs_length.cuda()
#             targets, targets_length = targets.cuda(), targets_length.cuda()

#         max_inputs_length = inputs_length.max().item()
#         max_targets_length = targets_length.max().item()
#         inputs = inputs[:, :max_inputs_length, :]
#         targets = targets[:, :max_targets_length]
#         i = 0
#         batch_size = inputs.size(0)

#         for i in range(batch_size):
#             beam = 5
#             preds = model.forward(input=inputs[i], length=targets_length[i], beam_width=beam)

#             transcripts = [targets.cpu().numpy()[i][:targets_length[i].item()]
#                         for i in range(targets.size(0))][i]

#             preds = list(filter(lambda x: x != 28, preds))
#             transcripts = list(filter(lambda x: x != 28, transcripts))

#             dist, num_words = computer_cer(preds, transcripts)
#             total_dist += dist
#             total_word += num_words

#             cer = total_dist / total_word * 100
#             print(cer)
#             i +=1

#         if visualizer is not None:
#             visualizer.add_scalar('cer_train', cer, epoch)

#     return cer

# exp_name = os.path.join('egs', config.data.name, 'exp', config.training.save_model)
# logger = init_logger(os.path.join(exp_name, opt.log))
# train_eval_data = TrainDataLoader('files/train_fragment.csv', get_tokenizer(), 16, 50)
# visualizer = SummaryWriter(os.path.join(exp_name, 'log'))

# eval_train(30, config, new_model_search, train_eval_data, logger, visualizer)