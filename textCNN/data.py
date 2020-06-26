import gensim
import torch as t
import numpy as np
from torch.utils.data import Dataset
from config import *


class TextDataset(Dataset):
    def __init__(self, fname, word2id):
        self.texts = []
        line = 'tmp'
        with open(fname, encoding='UTF-8') as f:
            while line:
                line = f.readline().strip()
                if not line:
                    continue
                self.texts.append(line)
        self.word2id = word2id

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        tokens = self.texts[index].split()
        clazz = int(tokens[0])
        word_idx = []
        for word in tokens[1:]:
            word_idx.append(self.word2id[word])
        length = len(word_idx)
        if length < Config.max_sen_len:
            word_idx.extend([self.word2id['_PAD_']] * (Config.max_sen_len - length))
        else:
            word_idx = word_idx[:Config.max_sen_len]
        return t.from_numpy(np.array(word_idx)).type(t.LongTensor), clazz


def build_word2id(path):
    word2id = {'_PAD_': 0}
    for p in path:
        line = 'tmp'
        with open(p, encoding='UTF-8') as f:
            while line:
                line = f.readline().strip()
                if not line:
                    continue
                tokens = line.split()
                for word in tokens[1:]:
                    if word not in word2id.keys():
                        word2id[word] = len(word2id)
    return word2id


def build_word2vec(fname, word2id):
    n_words = max(word2id.values()) + 1
    model = gensim.models.KeyedVectors.load_word2vec_format(fname, binary=True)
    word_vecs = np.array(np.random.uniform(-1., 1., [n_words, model.vector_size]))
    for word in word2id.keys():
        try:
            word_vecs[word2id[word]] = model[word]
        except KeyError:
            pass
    return word_vecs
