import inspect
import os
import sys

# import pytest
# import torch
# from conftest import simpleModel, simpleModel2, simpleModel3
# from torch import nn
# from torch.nn import ReLU
# from torchvision.models import resnet50

# from qrelu import (
#     calibrate,
#     optimal_replacer,
#     tmp_Relu,
#     tmp_Relu_2,
#     tmp_Relu_3,
#     tmpRelu_4,
# )

currentdir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe()))
)
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)


# from resnet50 import ResNet50  # noqa: E402

def test_train_loop():
    pass