import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.dataloader as dataloader
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms


class VGG11(nn.Module):
    def __init__(self):
        super(VGG11, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
        )
        self.fc = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),

            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),

            nn.Linear(4096, 2)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x


EPOCH = 15
BATCH_SIZE = 40
trans = {
    "train": transforms.Compose([
        # 先Resize到256
        transforms.Resize((256, 256)),
        # 随机剪切一部分到224, 因为VGG的输入是224
        transforms.RandomCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ]),
    "validate": transforms.Compose([
        # 直接resize到224
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataset = datasets.ImageFolder('data/train', trans["train"])
validate_dataset = datasets.ImageFolder('data/validation', trans["validate"])

train_loader = dataloader.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
validate_loader = dataloader.DataLoader(validate_dataset, batch_size=BATCH_SIZE, shuffle=False)

net = VGG11().to(device)
optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)


def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = Variable(x.cuda()), Variable(y.cuda())
        optimizer.zero_grad()
        out = net(x)
        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(x), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()
            ))


def test(epoch):
    print('\nValidation Epoch: %d' % epoch)
    net.eval()
    loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(validate_loader):
            x, y = Variable(x.cuda()), Variable(y.cuda())
            out = net(x)
            loss += F.cross_entropy(out, y, reduction='sum')
            pred = out.data.max(1, keepdim=True)[1]
            correct += pred.eq(y.data.view_as(pred)).cpu().sum()
        loss /= len(validate_loader.dataset)
        print('\nAverage loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            loss, correct, len(validate_loader.dataset),
            100. * correct / len(validate_loader.dataset)
        ))


for e in range(EPOCH + 1):
    train(e)
    test(e)
torch.save(net, 'dog-cat.pth')
