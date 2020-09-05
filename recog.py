import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
from PIL import Image
import os

data_root = './celeba/'
txt_file = 'identity_CelebA.txt'


class Celeb(datasets.VisionDataset):
    def __init__(self, dir, txt_file):
        super(Celeb, self).__init__(dir)
        self.transform = transforms.Compose([
            transforms.RandomCrop((178, 178)),
            transforms.Resize(128),
            transforms.toTensor()
        ])
        self.dir = dir
        self.labels = {}
        with open(txt_file) as f:
            lines = f.readlines()
            for line in lines:
                name, label = line.split("\t")
                self.labels[name] = eval(label)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        name = "%06d" % idx
        label = self.labels[name]
        img = Image.open(os.path.join(self.dir, name))
        img = self.transform(img)
        return img, label


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.mp = nn.MaxPool2d(2)
        self.fc = nn.Linear(320, 10)

    def forward(self, x):
        in_size = x.size(0)  # one batch
        x = F.relu(self.mp(self.conv1(x)))
        x = F.relu(self.mp(self.conv2(x)))
        x = x.view(in_size, -1)
        x = self.fc(x)
        # return F.log_softmax(x)
        return x


def train(epoch, type, batch_size):

    train_dataset = MNIST_half('./data_processed/train_{}.pt'.format(type))
    test_dataset = MNIST_half('./data_processed/test_{}.pt'.format(type))
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    model = Net()
    opt = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    for e in range(epoch):
        training(e, model, opt, train_loader)
        test(model, test_loader)
    torch.save(model.state_dict(), 'model{}.pkl'.format(type))


def training(epoch, model, opt, train_loader):
    for batch_idx, (data, target) in enumerate(train_loader):
        data.requires_grad = False
        opt.zero_grad()
        output = model(data)
        # loss = F.nll_loss(output, target)
        loss = F.cross_entropy(output, target)
        loss.backward()
        opt.step()
        if batch_idx % 100 == 0:
            print("1-Train Epoch:{} [{}/{}({:.0f}%)]\tLoss:{:.6f}".format(
                epoch, batch_idx*len(data), len(train_loader.dataset), 100.*batch_idx/len(train_loader), loss.item()))


def test(model, test_loader):
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data.requires_grad = False
        # data, target = Variable(data, requires_grad=False), Variable(target)
        output = model(data)
        test_loss += F.cross_entropy(output, target, size_average=False).item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss = test_loss / len(test_loader.dataset)
    print('\n1-test set: average loss:{:.4f}, acc: {}/{}({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), 100.*correct/(len(test_loader.dataset))))


def main():
    batchsize = 64
    epoch = 10
    train(epoch, 1, batchsize)
    train(epoch, 2, batchsize)

# if __name__ == '__main__':
#     main()






