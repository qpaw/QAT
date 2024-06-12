# import os
# import sys
# import inspect
import torch

# from torchvision.models import resnet50
from torch import nn

from qrelu import tmp_Relu


class simpleModel(nn.Module):
    """
    Only forward, only tmp activation tmprelu
    """

    def __init__(self) -> None:
        super(simpleModel, self).__init__()
        self.relu = tmp_Relu

    def forward(self, x):
        out = self.relu(x)
        return out


class simpleModel2(nn.Module):
    """
    Linear and another activation tmp_Relu
    """

    def __init__(self, constant_weight=0) -> None:
        super(simpleModel2, self).__init__()

        self.fc1 = nn.Linear(in_features=10, out_features=10, bias=False)
        if constant_weight != 0:
            self.fc1.weight.data = torch.full((10, 10), constant_weight)

        self.relu = tmp_Relu

    def forward(self, x):
        x = self.fc1(x)
        out = self.relu(x)
        return out


class simpleModel3(nn.Module):
    """
    Linear and origin activation nn.ReLU()
    """

    def __init__(self, constant_weight=0) -> None:
        super(simpleModel3, self).__init__()

        self.fc1 = nn.Linear(in_features=10, out_features=10, bias=False)
        if constant_weight != 0:
            self.fc1.weight.data = torch.full((10, 10), constant_weight)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        out = self.relu(x)
        return out
