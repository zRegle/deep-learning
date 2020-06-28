import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from vgg import VGG11
from config import Config

trans = {
    "train": transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ]),
    "validation": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataset = datasets.ImageFolder('data/train', trans['train'])
validate_dataset = datasets.ImageFolder('data/validation', trans['validation'])

train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)
validate_loader = DataLoader(validate_dataset, batch_size=Config.batch_size, shuffle=True)

net = VGG11().to(device)
optimizer = optim.SGD(net.parameters(), lr=1e-3)


def train(epoch):
    info = '\nEpoch: %d' % epoch
    print(info)
    log.write(info + '\n')
    net.train()
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.cuda(), y.cuda()
        optimizer.zero_grad()
        out = net(x)
        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()
        if batch_idx % Config.interval == 0:
            info = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(x), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item())
            print(info)
            log.write(info + '\n')
    torch.save(net.state_dict(), 'checkpoints/{}-{}.pth'.format(type(net).__name__, epoch))


def test(epoch, data_loader):
    info = '\nValidation Epoch: %d' % epoch
    print(info)
    log.write(info)
    net.eval()
    loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(data_loader):
            x, y = x.cuda(), y.cuda()
            out = net(x)
            loss += F.cross_entropy(out, y, reduction='sum')
            pred = out.data.max(1, keepdim=True)[1]
            correct += pred.eq(y.data.view_as(pred)).cpu().sum()
        loss /= len(validate_loader.dataset)
        info = '\nAverage loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            loss, correct, len(validate_loader.dataset),
            100. * correct / len(validate_loader.dataset))
        print(info)
        log.write(info + '\n')


log = open('train_log.txt', 'w')
for e in range(Config.epoch):
    train(e)
    test(e, validate_loader)
log.close()

