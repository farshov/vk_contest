import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import torch


class BiDAF(nn.Module):
    def __init__(self, word_vectors, hidden_size, drop_prob=0.):
        super(BiDAF, self).__init__()
        self.emb = layers.Embedding(word_vectors=word_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob)

        self.enc = layers.RNNEncoder(input_size=hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)

        self.att = layers.BiDAFAttention(hidden_size=2 * hidden_size,
                                         drop_prob=drop_prob)

        self.mod = layers.RNNEncoder(input_size=8 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)

        self.out = layers.BiDAFOutput(hidden_size=hidden_size,
                                      drop_prob=drop_prob)

    def forward(self, cw_idxs, qw_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        c_emb = self.emb(cw_idxs)         # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs)         # (batch_size, q_len, hidden_size)

        c_enc = self.enc(c_emb, c_len)    # (batch_size, c_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_len)    # (batch_size, q_len, 2 * hidden_size)

        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)

        mod = self.mod(att, c_len)        # (batch_size, c_len, 2 * hidden_size)

        out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out


class RNNClassifier(nn.Module):

    def __init__(self, embedding, emb_size=100, hidden_size=100, num_layers=2, num_aux=6):
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.emb_size = emb_size
        self.emb = embedding  # size: emb_size
        self.rnn = nn.LSTM(input_size=emb_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=True)
        self.linear1 = nn.Linear(in_features=4*hidden_size, out_features=hidden_size)
        self.relu = nn.ReLU()
        self.linear_out = nn.Linear(in_features=hidden_size, out_features=1)
        self.linear_aux_out = nn.Linear(in_features=hidden_size, out_features=num_aux)

    def forward(self, x):  # (batch, c_seq, 2*hidden_size)
        orig_len = x.size(1)
        lengths = torch.tensor([len(sen) for sen in x])
        x = pad_sequence(x, padding_value=0, batch_first=True)
        lengths, sort_idx = lengths.sort(0, descending=True)

        x = x[sort_idx]  # (batch_size, seq_len, input_size)
        x = pack_padded_sequence(x, lengths, batch_first=True)
        x, _ = self.rnn(x)  # (seq_len, batch, num_directions * hidden_size)
        x, _ = pad_packed_sequence(x, batch_first=True, total_length=orig_len)
        _, unsort_idx = sort_idx.sort(0)
        x = x[unsort_idx]  # (batch_size, seq_len, 2 * hidden_size)

        mean_pooling = torch.mean(x, 1)
        max_pooling = torch.max(x, 1)

        x = torch.cat((mean_pooling, max_pooling), 1)  # (batch_size, 4 * hidden_size)
        x = self.relu(self.linear1(x))


        result = self.linear_out(x)
        aux_result = self.linear_aux_out(x)
        out = torch.cat([result, aux_result], 1)

        return out
