import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import torch
import numpy as np


class RNNClassifier(nn.Module):

    def __init__(self, emb_size=100, hidden_size=100, num_layers=2, num_aux=6, device='cuda'):
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.emb_size = emb_size
        self.num_aux = num_aux
        self.device = device
        self.num_layer = num_layers

        self.emb = nn.Linear(in_features=emb_size, out_features=hidden_size)
        self.rnn = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=True)
        self.linear1 = nn.Linear(in_features=4*hidden_size, out_features=hidden_size)
        self.relu = nn.ReLU()
        self.linear_out = nn.Linear(in_features=hidden_size, out_features=1)
        self.linear_aux_out = nn.Linear(in_features=hidden_size, out_features=num_aux)

    def forward(self, x, lengths):  # (batch, c_seq, 2*hidden_size)
        # x = torch.tensor(self.emb.wv[np.array(x).flatten()].reshape((-1, 200, self.emb_size))).to(self.device)
        # x = torch.tensor([self.emb.wv[sen] for sen in x]).to(self.device)
        x = self.emb(x)
        orig_len = x.size(1)
        lengths, sort_idx = lengths.sort(0, descending=True)
        # x = torch.transpose(x, 0, 1)
        x = x[sort_idx]  # (batch_size, seq_len, input_size)
        x = pack_padded_sequence(x, lengths, batch_first=True)
        # h_0 = torch.zeros(self.num_  # (num_layers * num_directions, batch, hidden_size)
        x, _ = self.rnn(x, )  # (seq_len, batch, num_directions * hidden_size)
        x, _ = pad_packed_sequence(x, batch_first=True, total_length=orig_len)
        _, unsort_idx = sort_idx.sort(0)
        x = x[unsort_idx]  # (batch_size, seq_len, 2 * hidden_size)
        # x = torch.transpose(x, 1, 0)
        mean_pooling = torch.mean(x, 1)
        max_pooling, _ = torch.max(x, 1)

        x = torch.cat((mean_pooling, max_pooling), 1)  # (batch_size, 4 * hidden_size)
        x = self.relu(self.linear1(x))


        result = self.linear_out(x)
        aux_result = self.linear_aux_out(x)
        out = torch.cat([result, aux_result], 1)
        return out
