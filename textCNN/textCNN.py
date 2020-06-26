import torch as t
import torch.nn as nn
import torch.nn.functional as F
from config import Config

device = t.device('cuda' if t.cuda.is_available() else 'cpu')


class TextCNN(nn.Module):
    def __init__(self, weight):
        super(TextCNN, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(weight)
        self.embedding.weight.requires_grad = Config.update_w2v
        self.convs = nn.ModuleList([nn.Conv2d(1, Config.num_filters, (K, Config.embedding_dim))
                                    for K in Config.kernel_sizes])
        self.dropout = nn.Dropout(Config.drop_keep_prob)
        self.fc = nn.Linear(len(Config.kernel_sizes) * Config.num_filters, Config.classes)

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)
        x = x.type(t.FloatTensor).to(device)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = t.cat(x, 1)
        x = self.dropout(x)
        out = self.fc(x)
        return F.log_softmax(out, 1)
