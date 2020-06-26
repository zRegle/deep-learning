from torch.utils.data import DataLoader
from data import *
from textCNN import *

word2id = build_word2id([Config.train_path, Config.dev_path, Config.test_path])

test_data = TextDataset(Config.test_path, word2id)
test_loader = DataLoader(test_data, batch_size=Config.batch_size, shuffle=True)

dev = 'cuda' if t.cuda.is_available() else 'cpu'
model = t.load(Config.save_path + 'textCNN-9.pth', map_location=dev)
model = model.to(t.device(dev))


def test():
    print('Test: ', end='')
    model.eval()
    loss = 0
    correct = 0
    with t.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss += F.cross_entropy(out, y, reduction='sum')
            pred = out.data.max(1, keepdim=True)[1]
            correct += pred.eq(y.data.view_as(pred)).cpu().sum()
        loss /= len(test_loader.dataset)
        print('Accuracy: {}/{} ({:.0f}%)'.format(
            correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)
        ))


test()
