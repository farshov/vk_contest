import re
import torch
import torch.nn as nn
from gensim.models.fasttext import FastText
from torch.utils.data import Dataset
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_)(:!?*%]')


def clean_text(text):
    text = text.lower()  # lowercase text
    text = BAD_SYMBOLS_RE.sub('', text)
    text = [word for word in text.split()]  # remove stopwors from text
    return text


class EmbeddingFastText(nn.Module):

    def __init__(self, dim=100, window=5,  min_count=5):
        super(EmbeddingFastText, self).__init__()
        self.model = FastText(size=dim, window=window, min_count=min_count, workers=-1)

    def train(self, sent, epoch=10):
        self.model.build_vocab(sentences=sent)
        self.model.train(sentences=sent, total_examples=100_000, epochs=epoch)

    def forward(self, corpus):
        return self.model.wv[corpus]


class CommentsDataset(Dataset):

    def __init__(self, sentences, targets):
        self.sent = sentences
        self.targets = targets

    def __len__(self):
        return len(self.sent)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = (self.sent[idx], self.targets[idx])

        return sample