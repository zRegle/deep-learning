import torch.nn as nn
from config import *


class Poem(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Poem, self).__init__()
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=Config.num_layer)
        self.linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        seq_len, batch_size = x.size()
        if hidden is None:
            h_0 = x.data.new(Config.num_layer, batch_size, self.hidden_dim).fill_(0).float()
            c_0 = x.data.new(Config.num_layer, batch_size, self.hidden_dim).fill_(0).float()
        else:
            h_0, c_0 = hidden
        embeds = self.embeddings(x)
        out, hidden = self.lstm(embeds, (h_0, c_0))
        out = out.view(seq_len * batch_size, -1)
        out = self.linear(out)
        return out, hidden
