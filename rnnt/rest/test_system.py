import argparse
import yaml
from utils import AttrDict
from model import Transducer
from tokenizer import CharTokenizer
import torch
import numpy as np
import torch.nn.functional as F
from utils import computer_cer
import argparse
import yaml
import torch
import torch.utils.data
from utils import AttrDict, computer_cer
from tokenizer import CharTokenizer
from data import SentenceDataLoader

def greedy_decode(model, inputs, inputs_length):
    blank = 30
    zero_token = torch.LongTensor([[blank]])
    f, _ = model.encoder(inputs, None)
    if inputs.is_cuda:
        zero_token = zero_token.cuda()

    def decode(inputs, lengths):
        token_list = []
        u = 0
        t = 0
        gu, hidden = model.decoder(zero_token)
        umax = model.config.max_length

        while t < lengths and u < umax:
            h = model.joint(inputs[t].view(-1), gu.view(-1))
            out = F.log_softmax(h, dim=0)
            _, pred = torch.max(out, dim=0)
            pred = int(pred.item())

            if pred != blank:
                token_list.append(pred)
                token = torch.LongTensor([[pred]])
                if zero_token.is_cuda:
                    token = token.cuda()
                gu, hidden = model.decoder(token, hidden=hidden)
                u += 1
            else:
                t += 1

        return token_list

    decoded_seq = decode(f.squeeze(0), inputs_length)
    return decoded_seq

def load_mfcc(path):
    first_slash_index = path.find("/")
    path = (
        path[: first_slash_index + 1]
        + "MFCC/"
        + path[first_slash_index + 1 :]
    )
    path, ext = path.split(".")
    path = f"{path}.npy"
    aud = torch.tensor(np.load(path))

    n = 0
    zeros = torch.zeros(size=(1, n, aud.shape[-1]))
    return torch.cat([zeros, aud], dim=1), aud.shape[1]

def recognize(model, mfcc, size):
    tokens = [0, 28, 29, 30]
    transcription = greedy_decode(model, mfcc, size)
    return [t for t in transcription if t not in tokens]

def separate_list_by_number(arr, separator):
    result = []
    sublist = []
    
    for num in arr:
        if num == separator:
            if sublist:
                result.append(sublist)
                sublist = []
        else:
            sublist.append(num)
    
    if sublist:
        result.append(sublist)
    
    return result

def get_similarity(preds, labels):
    dist, total = computer_cer([preds], [labels])
    percent = (1 - dist/total) * 100
    return percent

def recognizeKeywords_or_reset(transcriptions_ids, keywords_ids):
    max_percent_list = [0] * len(keywords_ids)
    for idx, keyword_ids in enumerate(keywords_ids):
        for transcription_ids in transcriptions_ids:
            percent = get_similarity(transcription_ids, keyword_ids)
            max_percent_list[idx] = max(max_percent_list[idx], percent)
    
    return max_percent_list

def get_tokenizer():
    tokenizer = CharTokenizer()
    tokenizer = tokenizer.load_tokenizer("files/tokenizer.json")
    return tokenizer

def calculate_accuracy(array1, array2):
    if len(array1) != len(array2):
        raise ValueError("Arrays must be of the same length")

    correct = sum(a1 == a2 for a1, a2 in zip(array1, array2))
    total = len(array1)
    accuracy = correct / total
    return accuracy

parser = argparse.ArgumentParser()
parser.add_argument("-config", type=str, default="config/config.yaml")
parser.add_argument("-log", type=str, default="train.log")
parser.add_argument("-mode", type=str, default="retrain")
opt = parser.parse_args()

configfile = open(opt.config)
config = AttrDict(yaml.load(configfile, Loader=yaml.FullLoader))
tokenizer = get_tokenizer()

validate_data = SentenceDataLoader(
    "files/core_test_set.csv", tokenizer, config.data, 1
)
train_eval_data = SentenceDataLoader(
    "files/core_train_subset.csv", tokenizer, config.data, 1
)
threshold = 40
truth_table = []
guess_table = []
model = Transducer(config.model)
checkpoint = torch.load('timit/rnnt/2enc1dec_model.chkpt')
model.encoder.load_state_dict(checkpoint["encoder"])
model.decoder.load_state_dict(checkpoint["decoder"])
model.joint.load_state_dict(checkpoint["joint"])

for tup1, tup2 in zip(validate_data, train_eval_data):
    (inputs1, inputs_length1, targets1, targets_length1) = tup1
    (inputs2, inputs_length2, targets2, targets_length2) = tup2
    transcription = recognize(model, inputs1, inputs_length1)
    keywords = separate_list_by_number(targets1[0].tolist(), separator=27)
    transcription = separate_list_by_number(transcription, separator=27)

    correct_len = len(keywords)
    for keyword in separate_list_by_number(targets2[0], separator=27):
        keyword = [t.item() for t in keyword]
        if keyword not in keywords:
            keywords.append(keyword)

    incorrect_len = len(keywords) - correct_len
    truth = [1]*correct_len + [0]* incorrect_len
    truth_table.extend(truth)

    max_percent_list = recognizeKeywords_or_reset(transcription, keywords)

    recognized_words = [x>=threshold for x in max_percent_list]
    guess_table.extend(recognized_words)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
import itertools

def evaluate_predictions(ground_truth_list, guess_list):
    cm = confusion_matrix(ground_truth_list, guess_list)
    precision = precision_score(ground_truth_list, guess_list, average='binary')
    recall = recall_score(ground_truth_list, guess_list, average='binary')
    accuracy = accuracy_score(ground_truth_list, guess_list)

    plt.rcParams.update({'font.size': 14}) 
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Macierz pomyłek dla progu - {threshold}%')
    plt.colorbar()
    tick_marks = np.arange(len(set(ground_truth_list)))
    plt.xticks(tick_marks, set(ground_truth_list))
    plt.yticks(tick_marks, set(ground_truth_list))

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                  fontsize=16)

    plt.tight_layout()
    plt.ylabel('Prawdziwa klasyfikacja')
    plt.xlabel('Przewidziana klasyfikacja')
    plt.show()

    return accuracy, precision, recall

accuracy, precision, recall = evaluate_predictions(truth_table, guess_table)
print(accuracy, precision, recall)

# find LR that does NOT optimize (too low)
# find LR that is unstable (to high)
# linspace between 'too low' and 'to high' BUT exponentially for instance... (these are candidates for LR)
# lre = torch.linspace(-3, 0, 1000)
# lrs = 10**lre
# try each of LR for one batch
# keep track of lrs and lossi
# plot and choose