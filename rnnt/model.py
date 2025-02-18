import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder import build_encoder
from decoder import build_decoder


class JointNet(nn.Module):
    def __init__(self, inner_size, vocab_size):
        super(JointNet, self).__init__()
        self.proj = nn.Linear(inner_size, vocab_size)

    def forward(self, enc_state, dec_state):
        if enc_state.dim() == 3 and dec_state.dim() == 3:
            dec_state = dec_state.unsqueeze(1)
            enc_state = enc_state.unsqueeze(2)

            t = enc_state.size(1)
            u = dec_state.size(2)

            enc_state = enc_state.repeat([1, 1, u, 1])
            dec_state = dec_state.repeat([1, t, 1, 1])
        else:
            assert enc_state.dim() == dec_state.dim()

        concat_state = enc_state + dec_state
        output = self.proj(concat_state)
        return output


class Transducer(nn.Module):
    def __init__(self, config, vocab_size, device):
        super(Transducer, self).__init__()
        self.config = config
        self.device = device
        self.encoder = build_encoder(config)
        self.decoder = build_decoder(config, vocab_size)

        self.joint = JointNet(config.joint.inner_size, vocab_size)
        self.blank = vocab_size - 1
        try:
            from warprnnt_pytorch import RNNTLoss
            self.crit = RNNTLoss(blank=self.blank, reduction="None")
        except ImportError:
            self.crit = None

    def forward(self, inputs, inputs_length, targets, targets_length):

        enc_state, _ = self.encoder(inputs, inputs_length)
        concat_targets = F.pad(targets, pad=(1, 0, 0, 0), value=self.blank)
        dec_state, _ = self.decoder(concat_targets, targets_length.add(1))

        logits = self.joint(enc_state, dec_state)
        loss = self.crit(logits, targets.contiguous(), inputs_length, targets_length)
        return loss

    @torch.no_grad()
    def recognize(self, inputs, input_lengths):
        zero_token = torch.tensor([self.blank], device=self.device)
        batch_size = inputs.shape[0]

        encoded_sequences, _ = self.encoder(inputs, input_lengths)
        decoded_sequences = [
            self.decode_sequence(
                encoded_sequences[i], input_lengths[i], zero_token
            )
            for i in range(batch_size)
        ]

        return decoded_sequences

    @torch.no_grad()
    def decode_sequence(self, encoded_sequence, input_length, zero_token):
        preds = []
        u = 0
        t = 0
        u_max = self.config.max_length
        gu, hidden = self.decoder(zero_token)

        while t < input_length and u < u_max:
            h = self.joint(encoded_sequence[t].view(-1), gu.view(-1))
            out = F.log_softmax(h, dim=0)
            _, pred = torch.max(out, dim=0)
            pred = int(pred.item())

            if pred != self.blank:
                preds.append(pred)
                token = torch.tensor([pred], device=self.device)
                gu, hidden = self.decoder(token, hidden=hidden)
                u += 1
            else:
                t += 1

        return preds
