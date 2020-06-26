import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torchnet.meter as meter
import tqdm
from torch.utils.data import DataLoader
from config import *
from model import Poem
from generate import generate


def train():
    datas = np.load(Config.data_path, allow_pickle=True)
    data = datas['data']
    ix2word = datas['ix2word'].item()
    word2ix = datas['word2ix'].item()
    data = torch.from_numpy(data)
    dl = DataLoader(data, batch_size=Config.batch_size, shuffle=True)

    device = 'cuda' if Config.use_gpu else 'cpu'

    model = Poem(len(word2ix), Config.embedding_dim, Config.hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    if Config.model_path:
        model.load_state_dict(torch.load(Config.model_path))
    model = model.to(device)

    loss_meter = meter.AverageValueMeter()

    for epoch in range(Config.epoch):
        loss_meter.reset()
        for i, _data in tqdm.tqdm(enumerate(dl)):
            _data = _data.long().transpose(1, 0).contiguous().to(device)
            optimizer.zero_grad()
            _input, target = _data[:-1, :].to(device), _data[1:, :].to(device)
            output, _ = model(_input)
            loss = criterion(output, target.view(-1))
            loss.backward()
            optimizer.step()
            if (i + 1) % Config.interval == 0:
                print('epoch:{}, loss:{}'.format(epoch, loss_meter.mean))
                for word in list(u'春江花朝秋月夜'):
                    gen_poetry = ''.join(generate(model, word, ix2word, word2ix))
                    print(gen_poetry)
        torch.save(model.state_dict(), "{}_{}.pth".format(Config.model_prefix, epoch))


train()
