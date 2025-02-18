import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(
        self,
        hidden_size,
        vocab_size,
        output_size,
        num_layers,
        dropout,
    ):
        super(Decoder, self).__init__()
        self.one_hot_size = vocab_size - 1
        self.embedding = self.indices2onehot

        self.lstm = nn.LSTM(
            input_size=vocab_size - 1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.proj = nn.Linear(hidden_size, output_size)

    def indices2onehot(self, indices):
        mask = indices == self.one_hot_size
        adjusted_indices = torch.where(mask, torch.zeros_like(indices), indices).long()
        one_hot_batch = F.one_hot(adjusted_indices, self.one_hot_size).float()
        one_hot_batch[mask] = 0.0
        return one_hot_batch

    def forward(self, inputs, length=None, hidden=None):
        embed_inputs = self.embedding(inputs)

        if length is not None:
            sorted_seq_lengths, indices = torch.sort(length, descending=True)
            embed_inputs = embed_inputs[indices]

            embed_inputs = nn.utils.rnn.pack_padded_sequence(
                embed_inputs, sorted_seq_lengths.cpu(), batch_first=True
            )

        self.lstm.flatten_parameters()
        outputs, hidden = self.lstm(embed_inputs, hidden)

        if length is not None:
            _, desorted_indices = torch.sort(indices, descending=False)
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
            outputs = outputs[desorted_indices]

        logits = self.proj(outputs)
        return logits, hidden


def build_decoder(config, vocab_size):
    if config.dec.type == "lstm":
        return Decoder(
            hidden_size=config.dec.hidden_size,
            vocab_size=vocab_size,
            output_size=config.dec.output_size,
            num_layers=config.dec.num_layers,
            dropout=config.dec.dropout,
        )
    else:
        raise NotImplementedError
