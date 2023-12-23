import torch
import torch.nn.functional as F



def GreedyDecode(model, inputs, input_lengths):

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
        umax = 150
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
        print(u)
        return token_list

    for i in range(batch_size):
        decoded_seq = decode(f[i], input_lengths[i])
        results.append(decoded_seq)

    return results

def GreedyDecodeRNNT(self, inputs, input_lengths):
    assert inputs.dim() == 3
    # f = [batch_size, time_step, feature_dim]
    f, _ = self.encoder(inputs, input_lengths)
    blank = 27
    space = 28
    space_token = torch.LongTensor([[space]])
    zero_token = torch.LongTensor([[blank]])
    if inputs.is_cuda:
        zero_token = zero_token.cuda()
        space_token = space_token.cuda()
    results = []
    batch_size = inputs.size(0)


    def decode(inputs, lengths):
        log_prob = 0
        token_list = []
        umax = 100
        u = 0
        t = 0
        gu, hidden = self.decoder(zero_token)

        while t < lengths and u < umax:
            h = self.joint(inputs[t].view(-1), gu.view(-1))
            out = F.log_softmax(h, dim=0)
            prob, pred = torch.max(out, dim=0)
            pred = int(pred.item())
            log_prob += prob.item()
            if pred == space and len(token_list) != 0:
                t+=1
                gu, hidden = self.decoder(zero_token)
                token_list.append(pred)
                u+=1
            elif pred == blank:
                t+=1
            else:
                token_list.append(pred)
                token = torch.LongTensor([[pred]])
                if zero_token.is_cuda:
                    token = token.cuda()
                gu, hidden = self.decoder(token, hidden=hidden)
                u += 1

        return token_list

    for i in range(batch_size):
        decoded_seq = decode(f[i], input_lengths[i])
        results.append(decoded_seq)

    return results

def test(model, inputs, input_lengths):

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
if __name__ =="__main__":
    import argparse
    import yaml
    from rnnt.utils import AttrDict
    from rnnt.model import Transducer
    from rnnt.tokenizer import CharTokenizer
    from rnnt.data import SentenceDataLoader, WordDataLoader

    parser = argparse.ArgumentParser()

    parser.add_argument('-config', type=str, default='config/aishell.yaml')
    parser.add_argument('-log', type=str, default='train.log')
    parser.add_argument('-mode', type=str, default='retrain')
    opt = parser.parse_args()
    configfile = open(opt.config)
    config = AttrDict(yaml.load(configfile, Loader=yaml.FullLoader))
    checkpoint = torch.load('timit/rnnt/rnnt.epoch780.chkpt')
    model = Transducer(config.model)

    tokenizer = CharTokenizer()
    tokenizer = tokenizer.load_tokenizer('files/tokenizer.json')

    model.encoder.load_state_dict(checkpoint['encoder'])
    model.decoder.load_state_dict(checkpoint['decoder'])
    model.joint.load_state_dict(checkpoint['joint'])

    batch_size = 2
    test_data = SentenceDataLoader('files/test_sentences_val.csv', tokenizer, config.data, batch_size)
    training_data = SentenceDataLoader('files/asd.csv', tokenizer, config.data, batch_size)
    model.eval()
    path = "dupa4.txt"   
    for step, (inputs, inputs_length, targets, targets_length) in enumerate(training_data):
        preds = GreedyDecode(model, inputs, inputs_length)
        preds = tokenizer.ids2tokens(preds)
        targs = tokenizer.ids2tokens(targets.tolist())
        for l, t in zip(preds, targs):
            sentence = "".join(l)
            print(sentence)
            if sentence != "".join(t).replace('!', ''):
                with open(path, 'a') as file:
                    file.write(sentence + " -> " + "".join(t).replace('!', '') + '\n')  
