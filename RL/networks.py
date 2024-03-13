import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):

    def __init__(self, in_c=3, out=10):
        super(Model, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(in_c, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        torch.nn.init.xavier_uniform(self.conv1.weight)
        self.conv1.bias.data.fill_(0.01)
        torch.nn.init.xavier_uniform(self.conv2.weight)
        self.conv2.bias.data.fill_(0.01)

        self.adaPool = nn.AdaptiveAvgPool2d(5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, out)
        # self.fc3 = nn.Linear(84, out)

        torch.nn.init.xavier_uniform(self.fc1.weight)
        self.fc1.bias.data.fill_(0.01)
        torch.nn.init.xavier_uniform(self.fc2.weight)
        self.fc2.bias.data.fill_(0.01)
        # torch.nn.init.xavier_uniform(self.fc3.weight)
        # self.fc3.bias.data.fill_(0.01)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = self.adaPool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        return x
