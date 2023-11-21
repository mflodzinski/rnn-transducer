import torch
import torch.nn.functional as F


def GreedyDecode(model, inputs, input_lengths):

    assert inputs.dim() == 3
    # f = [batch_size, time_step, feature_dim]
    f, _ = model.encoder(inputs, input_lengths)

    zero_token = torch.LongTensor([[27]])
    if inputs.is_cuda:
        zero_token = zero_token.cuda()
    results = []
    batch_size = inputs.size(0)


    def decode(inputs, lengths):
        log_prob = 0
        token_list = []
        gu, hidden = model.decoder(zero_token)
        for t in range(lengths):
            h = model.joint(inputs[t].view(-1), gu.view(-1))
            out = F.log_softmax(h, dim=0)
            prob, pred = torch.max(out, dim=0)
            pred = int(pred.item())
            log_prob += prob.item()
            if pred != 27:
                token_list.append(pred)
                token = torch.LongTensor([[pred]])
                if zero_token.is_cuda:
                    token = token.cuda()
                gu, hidden = model.decoder(token, hidden=hidden)

        return token_list

    for i in range(batch_size):
        decoded_seq = decode(f[i], input_lengths[i])
        results.append(decoded_seq)

    return results

def GreedyDecodeRNNT(model, inputs, input_lengths):

    assert inputs.dim() == 3
    # f = [batch_size, time_step, feature_dim]
    f, _ = model.encoder(inputs, input_lengths)
    blank = 27
    zero_token = torch.LongTensor([[blank]])
    if inputs.is_cuda:
        zero_token = zero_token.cuda()
    results = []
    batch_size = inputs.size(0)


    def decode(inputs, lengths):
        log_prob = 0
        token_list = []
        umax = 200
        u = 0
        t = 0
        gu, hidden = model.decoder(zero_token)

        while t < lengths and u < umax:
            h = model.joint(inputs[t].view(-1), gu.view(-1))
            out = F.log_softmax(h, dim=0)
            prob, pred = torch.max(out, dim=0)
            pred = int(pred.item())
            log_prob += prob.item()

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

    for i in range(batch_size):
        decoded_seq = decode(f[i], input_lengths[i])
        results.append(decoded_seq)

    return results


if __name__ =="__main__":
    import argparse
    import yaml
    from utils import AttrDict
    from model import Transducer
    from data import DataLoader

    parser = argparse.ArgumentParser()

    parser.add_argument('-config', type=str, default='config/aishell.yaml')
    parser.add_argument('-log', type=str, default='train.log')
    parser.add_argument('-mode', type=str, default='retrain')
    opt = parser.parse_args()
    configfile = open(opt.config)
    config = AttrDict(yaml.load(configfile, Loader=yaml.FullLoader))
    checkpoint = torch.load(config.training.load_model)
    model = Transducer(config.model)

    from tokenizer import CharTokenizer
    tokenizer = CharTokenizer()
    tokenizer = tokenizer.load_tokenizer('files/tokenizer.json')

    model.encoder.load_state_dict(checkpoint['encoder'])
    model.decoder.load_state_dict(checkpoint['decoder'])
    model.joint.load_state_dict(checkpoint['joint'])
    test_data = DataLoader('files/test_newest.csv', tokenizer, 8, 250)
    training_data = DataLoader('files/train_newest.csv', tokenizer, 8, 250)

    path = "compare_basic_test.txt"
    for step, (inputs, inputs_length, targets, targets_length) in enumerate(test_data):
        preds = GreedyDecode(model, inputs, inputs_length)
        preds = tokenizer.ids2tokens(preds)
        for l in preds:
            sentence = "".join(l)
            print(sentence)
            with open(path, 'a') as file:
                file.write(sentence + '\n')  
