import re
import torch
import torch.nn as nn
import numpy as np
from gensim.models.fasttext import FastText
from torch.utils.data import Dataset
import os
import random
import tqdm
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_)(:!?*%]')


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def preprocess(data):
    symbols = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~`" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'

    def clean_special_chars(text, symbols):
        for s in symbols:
            text = text.replace(s, ' ')
        return text

    data = data.astype(str).apply(lambda x: clean_special_chars(x, symbols).lower())
    return data


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# def clean_text(text):
#     text = text.lower()  # lowercase text
#     text = BAD_SYMBOLS_RE.sub('', text)
#     text = [word for word in text.split()]  # remove stopwors from text
#     return text


# class EmbeddingFastText():
#
#     def __init__(self, dim=100, window=5,  min_count=5):
#         # super(EmbeddingFastText, self).__init__()
#         self.model = FastText(size=dim, window=window, min_count=min_count, workers=-1)
#
#     def train(self, sent, epoch=10):
#         self.model.build_vocab(sentences=sent)
#         self.model.train(sentences=sent, total_examples=len(sent), epochs=epoch)
#
#     def forward(self, corpus):
#         return self.model.wv[corpus]


class CommentsDataset(Dataset):

    def __init__(self, sentences, targets, lenghts):
        self.sent = sentences
        self.targets = targets
        self.lengths = lenghts

    def __len__(self):
        return len(self.sent)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = (self.sent[idx], self.targets[idx], self.lengths[idx])

        return sample