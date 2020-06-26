import torch.optim as optim
from torch.utils.data import DataLoader
from data import *
from textCNN import *

word2id = build_word2id([Config.train_path, Config.dev_path])
word2vecs = build_word2vec(Config.pre_word2vec_path, word2id)

train_data = TextDataset(Config.train_path, word2id)
dev_data = TextDataset(Config.dev_path, word2id)
train_loader = DataLoader(train_data, batch_size=Config.batch_size, shuffle=True)
dev_loader = DataLoader(dev_data, batch_size=Config.batch_size, shuffle=True)

device = t.device('cuda' if t.cuda.is_available() else 'cpu')
weight = t.from_numpy(word2vecs)
model = TextCNN(weight).to(device)
optimizer = optim.Adam(model.parameters(), lr=Config.lr)


def train(epoch, f):
    info = '\nEpoch: %d' % epoch
    f.write(info + '\n')
    print(info)
    model.train()
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()
        if batch_idx % Config.interval == 0:
            info = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(x), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item())
            f.write(info + '\n')
            print(info)
            t.save(model, Config.save_path + 'textCNN-{}.pth'.format(epoch))


def validate(epoch, f):
    info = '\nValidation Epoch: %d' % epoch
    f.write(info + '\n')
    print(info)
    model.eval()
    loss = 0
    correct = 0
    with t.no_grad():
        for batch_idx, (x, y) in enumerate(dev_loader):
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss += F.cross_entropy(out, y, reduction='sum')
            pred = out.data.max(1, keepdim=True)[1]
            correct += pred.eq(y.data.view_as(pred)).cpu().sum()
        loss /= len(dev_loader.dataset)
        info = 'Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            loss, correct, len(dev_loader.dataset),
            100. * correct / len(dev_loader.dataset))
        f.write(info + '\n')
        print(info)


file = open('train_log.txt', 'w')
for e in range(Config.epoch):
    train(e, file)
    validate(e, file)
file.close()
