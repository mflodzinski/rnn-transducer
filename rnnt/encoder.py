import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_layers,
        dropout,
        bidirectional=True,
    ):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.proj = Projection(hidden_size, output_size)

    def forward(self, inputs, input_lengths):
        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(0)
        if input_lengths is not None:
            sorted_seq_lengths, indices = torch.sort(input_lengths, descending=True)
            inputs = inputs[indices]
            inputs = nn.utils.rnn.pack_padded_sequence(
                inputs, sorted_seq_lengths.cpu(), batch_first=True
            )

        self.lstm.flatten_parameters()
        outputs, hidden = self.lstm(inputs)

        if input_lengths is not None:
            _, desorted_indices = torch.sort(indices, descending=False)
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
            outputs = outputs[desorted_indices]

        logits = self.proj(outputs)
        return logits, hidden


class Projection(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Projection, self).__init__()
        self.linear1 = nn.Linear(hidden_size, output_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        forward_output, backward_output = x.chunk(2, dim=-1)
        forward_projected = self.linear1(forward_output)
        backward_projected = self.linear2(backward_output)
        return forward_projected + backward_projected


def build_encoder(config):
    if config.enc.type == "lstm":
        return Encoder(
            input_size=config.num_features,
            hidden_size=config.enc.hidden_size,
            output_size=config.enc.output_size,
            num_layers=config.enc.num_layers,
            dropout=config.enc.dropout,
            bidirectional=config.enc.bidirectional,
        )
    else:
        raise NotImplementedError
