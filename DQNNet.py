import torch
import torch.nn as nn
import torchvision
import config


class DQNNet(torch.nn.Module):
    def __init__(self):
        super(DQNNet, self).__init__()

        self.resnet18 = torchvision.models.resnet18()
        in_ch = 3
        if config.use_simple:
            in_ch = 1
        self.resnet18.conv1 = nn.Conv2d(
            in_channels=in_ch, out_channels=64,
            kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet18.fc = nn.Linear(in_features=512, out_features=4)

    def forward(self, x):
        x = self.resnet18(x)
        return x