#Modified from https://github.com/soapisnotfat/pytorch-cifar10/blob/master/models/LeNet.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
from torchvision import transforms

from src.torchshape import tensorshape

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        image_size = (4,3,32,32)

        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        out_size = tensorshape(self.conv1, image_size)

        self.maxpool1 = nn.MaxPool2d(2)
        out_size = tensorshape(self.maxpool1, out_size)

        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        out_size = tensorshape(self.conv2, out_size)
        
        self.maxpool2 = nn.MaxPool2d(2)
        out_size = tensorshape(self.maxpool2, out_size)

        self.flatten = nn.Flatten()
        out_size = tensorshape(self.flatten, out_size)

        self.fc1 = nn.Linear(out_size[-1], 120)
        out_size = tensorshape(self.fc1, out_size)

        self.fc2 = nn.Linear(120, 84)
        out_size = tensorshape(self.fc2, out_size)

        self.fc3 = nn.Linear(84, 10)
        out_size = tensorshape(self.fc3, out_size)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=32, shuffle=False)

model = LeNet()

model.eval()
with torch.no_grad():
    data, target = next(iter(test_loader))
    output = model(data)


