import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from vgg import VGG11
from config import Config

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

test_dataset = datasets.ImageFolder('data/test', transform)
test_loader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loc = 'cuda' if torch.cuda.is_available() else 'cpu'

model = VGG11()
model.load_state_dict(torch.load(Config.model_path, map_location=loc))
model = model.to(device)


model.eval()
loss = 0
correct = 0
with torch.no_grad():
    for batch_idx, (x, y) in enumerate(test_loader):
        x, y = x.cuda(), y.cuda()
        out = model(x)
        loss += F.cross_entropy(out, y, reduction='sum')
        pred = out.data.max(1, keepdim=True)[1]
        correct += pred.eq(y.data.view_as(pred)).cpu().sum()
    loss /= len(test_loader.dataset)
    print('\nAverage loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)
    ))
