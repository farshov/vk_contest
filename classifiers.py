import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch


class RNNClassifier(nn.Module):
    def __init__(self, input_size=100, hidden_size=100, num_layers=3):
        super(RNNClassifier, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=True)
        self.final_layer = nn.Sequential(
            nn.Linear(in_features=600, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=6)
        )

    def forward(self, x):  # (batch, c_seq, 2*hidden_size)
        # c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        # q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        # c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)
        # orig_len = x.size(1)
        # !!!lengths!!!!!

        lengths, sort_idx = lengths.sort(0, descending=True)
        x = x[sort_idx]  # (batch_size, seq_len, input_size)
        x = pack_padded_sequence(x, lengths, batch_first=True)
        _, h_n, c_n = self.rnn(x)  # (batch_size, seq_len, 2 * hidden_size)  # may be use stanford attentive reader
        # hn (num_layers * num_directions, batch, hidden_size)
        h_n = h_n.view(self.num_layers, 2, -1, self.hidden_size)# (num_layers, num_directions, batch, hidden_size)
        h_n = torch.transpose(h_n, 2, 0)
        h_n = h_n.view(-1, self.num_layers * 2 * self.hidden_size)
        output = self.final_layer(h_n)

        return torch.sigmoid(output)  # (bach_size, 6)
