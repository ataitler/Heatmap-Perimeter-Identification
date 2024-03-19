import torch
import torch.nn as nn
import torch.nn.functional as F


class Simple(nn.Module):

    def __init__(self, in_c=3, out=10):
        super(Model, self).__init__()
        # 3 input image channels, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(in_c, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        torch.nn.init.xavier_normal_(self.conv1.weight)
        self.conv1.bias.data.fill_(0.01)
        torch.nn.init.xavier_normal_(self.conv2.weight)
        self.conv2.bias.data.fill_(0.01)

        self.adaPool = nn.AdaptiveAvgPool2d(5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, out)

        torch.nn.init.xavier_normal_(self.fc1.weight)
        self.fc1.bias.data.fill_(0.01)
        torch.nn.init.xavier_normal_(self.fc2.weight)
        self.fc2.bias.data.fill_(0.01)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = self.adaPool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class LeNet5(nn.Module):
    def __init__(self, in_c=3, out=10):
        super(LeNet5, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_c, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4))
        self.fc1 = nn.Linear(32*4*4, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, out)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out