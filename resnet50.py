import torch
import torch.nn as nn
from torch.nn import functional as F

import qrelu

# import numpy as np
# import os


class ResidualBlock(nn.Module):
    """
    Basic block for network

    version :
            if equal "v1.5", then used modified version ResNet50 v1.5 model,
              else original version on https://arxiv.org/abs/1512.03385
              TODO: recheck dim for v1.0 model
    bottleneck_type:
            1 or 2. First type use schem [256 -> 64 -> 64 -> 256] (example),
            second - [256 -> 128-> 128 -> 512] (example)

    """

    def __init__(
        self,
        inchanal,
        outchanal,
        stride=1,
        shortcut=None,
        version="v1.5",
        bottleneck_type=1,
    ) -> None:
        super(ResidualBlock, self).__init__()
        if version != "v1.5":
            self.left = nn.Sequential(
                nn.Conv2d(
                    in_channels=inchanal,
                    out_channels=outchanal,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(num_features=outchanal),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    in_channels=outchanal,
                    out_channels=outchanal,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(num_features=outchanal),
            )
        else:
            self.left = nn.Sequential(
                nn.Conv2d(
                    in_channels=(
                        inchanal if bottleneck_type == 1 else inchanal // 2
                    ),
                    out_channels=outchanal,
                    kernel_size=1,
                    stride=stride,
                    # padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(num_features=outchanal),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    in_channels=outchanal,
                    out_channels=outchanal,
                    kernel_size=3,
                    stride=1 if bottleneck_type == 1 else 2,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(num_features=outchanal),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    in_channels=outchanal,
                    out_channels=(
                        inchanal if inchanal != 64 else 4 * inchanal
                    ),  # 64 only for first layer
                    kernel_size=1,
                    stride=1,
                    # padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(
                    num_features=inchanal if inchanal != 64 else 4 * inchanal
                ),  # 64 only for first layer
            )

        self.relu = nn.ReLU(inplace=True)

        self.downsample = shortcut

    def forward(self, x):

        identity = x.clone()
        out = self.left(x)
        residual = (
            identity if self.downsample is None else self.downsample(identity)
        )
        out += residual
        return self.relu(out)


class ResNet50(nn.Module):
    """
    ResNet50 network
    """

    def __init__(
        self,
        version="v1.5",
        num_classes=1000,
        activ_func="RELU",
        bits=8,
        arange=(-1, 1),
    ) -> None:
        super(ResNet50, self).__init__()
        # first layer

        self.conv_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # residual block by scheme [3, 4, 6, 3] - block_num in conv_y_x,
        # see page 5 in original paper 'Deep Residual Learning
        # for Image Recognition' https://arxiv.org/pdf/1512.03385v1.pdf

        self.conv_2_x = self._create_ResBlock(
            inchannel=256, outchanal=64, block_num=3, version=version
        )
        self.conv_3_x = self._create_ResBlock(
            inchannel=512,
            outchanal=128,
            block_num=4,
            stride=1,
            version=version,
        )
        self.conv_4_x = self._create_ResBlock(
            inchannel=1024,
            outchanal=256,
            block_num=6,
            stride=1,
            version=version,
        )
        self.conv_5_x = self._create_ResBlock(
            inchannel=2048,
            outchanal=512,
            block_num=3,
            stride=1,
            version=version,
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # last fullyconnected layer

        self.fc = nn.Linear(2048, num_classes)

        # weighth initialization
        self.set_kaiming_initialization()

        if activ_func == "LeakyReLU":
            self.set_activations(nn.LeakyReLU())

        if activ_func == "qRelu":  # only for dev
            self.set_activations(qrelu.qRelu())

        if activ_func == "qReLU":
            with torch.no_grad():
                calibred_range = qrelu.calibrate(nn.ReLU(), bits, arange)
            self.set_activations(
                qrelu.tmpRelu_4(calibred_range[0], calibred_range[1])
            )

        if activ_func == "FinalQReLU":
            with torch.no_grad():
                # calibred_range = qrelu.calibrate(nn.ReLU(), bits, arange)
                calibred_range = qrelu.calibrate(qrelu.dReLU, bits, arange)

            self.set_activations(
                qrelu.FinalQRelu(calibred_range[0], calibred_range[1])
            )

        if activ_func == "QSigmoid":
            with torch.no_grad():
                # calibred_range = qrelu.calibrate(nn.ReLU(), bits, arange)
                calibred_range = qrelu.calibrate(qrelu.dSigmoid, bits, arange)

            self.set_activations(
                qrelu.FinalQSigmoid(calibred_range[0], calibred_range[1])
            )

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2_x(x)
        x = self.conv_3_x(x)
        x = self.conv_4_x(x)
        x = self.conv_5_x(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return F.sigmoid(x)

    def _create_ResBlock(
        self, inchannel, outchanal, block_num, stride=1, version="v1.5"
    ):
        """
        Return ResidualBlocks in list block_num len
        """
        shortcut = nn.Sequential(
            nn.Conv2d(
                in_channels=int(
                    outchanal * (1 if inchannel == 256 else 2)
                ),  # only for first layer
                out_channels=inchannel,
                kernel_size=1,
                stride=(1 if inchannel == 256 else 2),
                bias=False,
            ),
            nn.BatchNorm2d(num_features=inchannel),
        )
        layers = []

        layers.append(
            ResidualBlock(
                inchanal=int(
                    inchannel * (1 / 4 if inchannel == 256 else 1)
                ),  # only for first layer
                outchanal=outchanal,
                stride=stride,
                shortcut=shortcut,
                version=version,
                bottleneck_type=(
                    1 if inchannel == 256 else 2
                ),  # only for first layer
            )
        )
        for _ in range(1, block_num):
            layers.append(
                ResidualBlock(
                    inchanal=inchannel,
                    outchanal=outchanal,
                    version=version,
                )
            )

        return nn.Sequential(*layers)

    def set_kaiming_initialization(self):
        """
        from
        https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def set_activations(self, new_activ_func, model=None):
        if not model:
            model = self

        for child_name, child in model.named_children():
            if isinstance(child, nn.ReLU):
                setattr(model, child_name, new_activ_func)
            else:
                self.set_activations(new_activ_func, child)

    def get_activations_count(self, model=None):
        c = 0
        if not model:
            model = self

        for child_name, child in model.named_children():
            if (
                isinstance(child, nn.ReLU)
                or isinstance(child, qrelu.FinalQRelu)
                or isinstance(child, qrelu.FinalQSigmoid)
            ):
                c += 1
            else:
                c += self.get_activations_count(child)
        return c

    def set_act_by_index(self, new_activ_func, start=0, model=None):
        ind = start
        if not model:
            model = self

        for child_name, child in model.named_children():
            if isinstance(child, nn.ReLU):

                # if ind == index:
                setattr(model, child_name, new_activ_func[ind])
                ind += 1
            else:

                ind += self.set_act_by_index(new_activ_func, ind, child)
        if ind == start:
            return 0
        return ind - start

    def change_by_qactivations(
        self,
        arange=(-1, 1),
        bitlist=[],
        active_func=qrelu.FinalQRelu,
        active_deriv=qrelu.dReLU,
    ):
        if len(bitlist) != self.get_activations_count():
            print("Ne equal len in model and new quants")

        act_list = []
        with torch.no_grad():
            for bit in bitlist:
                if bit != 0:
                    calibred_range = qrelu.calibrate(
                        func=active_deriv, bits=bit, ranges=arange
                    )
                    act_list.append(
                        active_func(calibred_range[0], calibred_range[1])
                    )
                else:
                    act_list.append(nn.ReLU())

        # for i, new_act in enumerate(act_list):
        self.set_act_by_index(act_list)
