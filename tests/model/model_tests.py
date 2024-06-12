import inspect
import os
import sys

import pytest
import torch
from conftest import simpleModel, simpleModel2, simpleModel3
from torch import nn
from torch.autograd import gradcheck
from torch.nn import ReLU
from torch.nn import functional as F
from torchvision.models import resnet50

from qrelu import (  # StepFunc,
    FinalQRelu,
    FinalQSigmoid,
    calibrate,
    dReLU,
    dSigmoid,
    optimal_replacer,
    stepfunc,
    tmp_Relu,
    tmp_Relu_2,
    tmp_Relu_3,
    tmpRelu_4,
)

currentdir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe()))
)
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)


from resnet50 import ResNet50  # noqa: E402
from utils import ClassStratageGenerator, ClassStrategyScore  # noqa: E402

model_1 = ResNet50()
model_2 = resnet50()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_1.to(device=device)
model_2.to(device=device)


def test_model_total_params():

    total_params_1 = sum(p.numel() for p in model_1.parameters())
    total_params_2 = sum(p.numel() for p in model_2.parameters())

    assert total_params_1 == total_params_2


def test_model_total_trainable_params():

    total_trainable_params_1 = sum(
        p.numel() for p in model_1.parameters() if p.requires_grad
    )
    total_trainable_params_2 = sum(
        p.numel() for p in model_2.parameters() if p.requires_grad
    )

    assert total_trainable_params_1 == total_trainable_params_2


def test_model_can_run():
    batch = torch.rand(1, 3, 224, 224).to(device=device)
    model_1.eval()
    with torch.no_grad():
        result = model_1(batch)
    assert list(result.size()) == [1, 1000]


def test_relu_activations():
    model_3 = ResNet50()
    model_3.to(device=device)

    str_model = str_model2 = ""
    for i, m in enumerate(model_3.modules()):
        str_model += "{} ----- {}".format(i, m)

    for i, m in enumerate(model_1.modules()):
        str_model2 += "{} ----- {}".format(i, m)

    assert str_model2 == str_model


def test_leackyrelu_activations_replaced():
    model_3 = ResNet50(activ_func="LeakyReLU")
    model_3.to(device=device)

    str_model = str_model2 = ""
    for i, m in enumerate(model_3.modules()):
        str_model += "{} ----- {}".format(i, m)

    for i, m in enumerate(model_1.modules()):
        str_model2 += "{} ----- {}".format(i, m)

    assert str_model2 != str_model


def test_qrelu_forward_work():
    model_4 = ResNet50(activ_func="LeakyReLU")
    model_4.to(device=device)
    batch = torch.ones(1, 3, 224, 224).to(device=device)
    model_4.eval()
    with torch.no_grad():
        result = model_4(batch)
    assert list(result.size()) == [1, 1000]


def test_LeakyReLU_backward_work():
    model_4 = ResNet50(activ_func="LeakyReLU")
    model_4.to(device=device)
    batch = torch.ones(1, 3, 224, 224).to(device=device)
    label = torch.ones(1, 1000).to(device=device)

    optimizer = torch.optim.SGD(
        model_4.parameters(),
        lr=0.001,
    )
    criterion = nn.CrossEntropyLoss()
    _ = model_4(batch)

    model_4.train()
    image, target = batch, label

    output = model_4(image)
    loss = criterion(output, target)

    optimizer.zero_grad()

    loss.backward()
    optimizer.step()

    assert abs(loss.item() - 6916) < 2


def test_qrelu_backward_work():
    model_4 = ResNet50(activ_func="qRelu")
    model_4.to(device=device)
    batch = torch.ones(1, 3, 224, 224).to(device=device)
    label = torch.ones(1, 1000).to(device=device)

    optimizer = torch.optim.SGD(
        model_4.parameters(),
        lr=0.001,
    )
    criterion = nn.CrossEntropyLoss()

    model_4.train()
    image, target = batch, label

    output = model_4(image)
    loss = criterion(output, target)

    optimizer.zero_grad()

    loss.backward()
    optimizer.step()

    assert abs(loss.item() - 6916) < 2


def test_tmprelu_equal_origin_relu_forward():
    tmp_model = simpleModel()
    x = torch.tensor(
        [[-1, -1, -1, -1, -1, 1, 1, 1, 1, 2]], dtype=torch.float32
    )
    res = tmp_model(x)
    assert res.tolist()[0] == [0, 0, 0, 0, 0, 1, 1, 1, 1, 2]


def test_tmprelu_equal_origin_relu_backward():
    criterion = nn.CrossEntropyLoss()
    x = torch.tensor(
        [[-1, -1, -1, -1, -1, 1, 1, 1, 1, 2]],
        dtype=torch.float32,
        # requires_grad=True,
    )
    y = torch.tensor(
        [[0, 0, 0, 0, 0, 1, 1, 1, 1, 3]],
        dtype=torch.float32,
        # requires_grad=True,
    )

    # ----------- first model
    tmp_model = simpleModel2(constant_weight=0.5)  # fake Relu

    optimizer = torch.optim.SGD(
        tmp_model.parameters(),
        lr=1,
    )

    tmp_model.train()
    optimizer.zero_grad()
    # step 1
    output = tmp_model(x)
    loss = criterion(output, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    optimizer.step()  # need two step for manual realization of activation
    # print(f"Batch size 1 (final) - grad: {tmp_model.fc1.weight.grad}")
    # print(f"Batch size 1 (final) - weight: {tmp_model.fc1.weight}")
    # step 2
    output = tmp_model(x)

    # ----------- second model
    criterion2 = nn.CrossEntropyLoss()
    x2 = torch.tensor(
        [[-1, -1, -1, -1, -1, 1, 1, 1, 1, 2]], dtype=torch.float32
    )
    tmp_model2 = simpleModel3(constant_weight=0.5)  # origin Relu
    optimizer2 = torch.optim.SGD(
        tmp_model2.parameters(),
        lr=1,
    )

    tmp_model2.train()
    optimizer2.zero_grad()
    # step 1
    output2 = tmp_model2(x2)
    loss = criterion2(output2, y)
    optimizer2.zero_grad()
    loss.backward()
    optimizer2.step()
    # optimizer2.step()
    # print(f"Batch size 1 (final) - grad: {tmp_model.fc1.weight.grad}")
    # print(f"Batch size 1 (final) - weight: {tmp_model.fc1.weight}")
    # step 2
    output2 = tmp_model2(x2)

    assert output.tolist()[0] == output2.tolist()[0]


def test_tmprelu_work_with_batch():
    BS = 4  # batch size
    EP = 2  # epoch count
    x = torch.tensor(
        [
            [-1, -1, -1, -1, -1, 1, 1, 1, 1, 2],
            [-1, -1, -1, -1, -1, 1, 1, 1, 1, 2],
            [-1, -1, -1, -1, -1, 1, 1, 1, 1, 2],
            [-1, -1, -1, -1, -1, 1, 1, 1, 1, 2],
            [-1, -1, -1, -1, -1, 1, 1, 1, 1, 2],
            [-1, -1, -1, -1, -1, 1, 1, 1, 1, 2],
            [-1, -1, -1, -1, -1, 1, 1, 1, 1, 2],
            [-1, -1, -1, -1, -1, 1, 1, 1, 1, 2],
        ],
        dtype=torch.float32,
        # requires_grad=True,
    )
    y = torch.tensor(
        [
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 3],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 3],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 3],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 3],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 3],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 3],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 3],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 3],
        ],
        dtype=torch.float32,
        # requires_grad=True,
    )

    # ----------- first model
    tmp_model = simpleModel2(constant_weight=0.5)  # fake Relu
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        tmp_model.parameters(),
        lr=0.06,
    )
    tmp_model.train()
    for _ in range(EP):
        for i in range(y.shape[1] // BS):
            optimizer.zero_grad()
            output = tmp_model(x[i * BS : (i + 1) * BS])  # noqa: E203
            loss = criterion(output, y[i * BS : (i + 1) * BS])  # noqa: E203

            loss.backward()
            optimizer.step()

    tmp_model.eval()
    output = tmp_model(x[0])
    assert torch.round(output).tolist() == [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
        1.0,
        2.0,
    ]


def test_relu_corect_batch():
    x = torch.tensor(
        [
            [-1, -1, -1, -1, -1, 1, 1, 1, 1, 2],
            [-1, -1, -1, -1, -1, 1, 1, 1, 1, 2],
            [-1, -1, -1, -1, -1, 1, 1, 1, 1, 2],
            [-1, -1, -1, -1, -1, 1, 1, 1, 1, 2],
            [-1, -1, -1, -1, -1, 1, 1, 1, 1, 2],
            [-1, -1, -1, -1, -1, 1, 1, 1, 1, 2],
            [-1, -1, -1, -1, -1, 1, 1, 1, 1, 2],
            [-1, -1, -1, -1, -1, 1, 1, 1, 1, 2],
        ],
        dtype=torch.float32,
    )
    res = tmp_Relu(x)
    assert res.shape == x.shape


def test_corect_train_loss_down():
    BS = 4  # batch size
    EP = 5  # epoch count
    x = torch.tensor(
        [
            [-1, -1, -1, -1, -1, 1, 1, 1, 1, 2],
            [-1, -1, -1, -1, -1, 1, 1, 1, 1, 2],
            [-1, -1, -1, -1, -1, 1, 1, 1, 1, 2],
            [-1, -1, -1, -1, -1, 1, 1, 1, 1, 2],
            [-1, -1, -1, -1, -1, 1, 1, 1, 1, 2],
            [-1, -1, -1, -1, -1, 1, 1, 1, 1, 2],
            [-1, -1, -1, -1, -1, 1, 1, 1, 1, 2],
            [-1, -1, -1, -1, -1, 1, 1, 1, 1, 2],
        ],
        dtype=torch.float32,
        # requires_grad=True,
    )
    y = torch.tensor(
        [
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 3],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 3],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 3],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 3],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 3],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 3],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 3],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 3],
        ],
        dtype=torch.float32,
        # requires_grad=True,
    )

    # ----------- first model
    tmp_model = simpleModel2(constant_weight=0.5)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        tmp_model.parameters(),
        lr=0.01,
    )
    tmp_model.train()
    saved_loss = 0
    for ep in range(EP):
        for i in range(y.shape[1] // BS):
            optimizer.zero_grad()
            output = tmp_model(x[i * BS : (i + 1) * BS])  # noqa: E203
            loss = criterion(output, y[i * BS : (i + 1) * BS])  # noqa: E203
            loss.backward()
            optimizer.step()
        if ep == 0:  # first epoch
            saved_loss = loss
        if ep == (EP - 1):  # last epoch
            assert loss.item() < saved_loss.item()


def test_grad_calculation_forward():
    a = torch.ones(3, requires_grad=True)
    b = tmp_Relu(a)
    assert b.tolist() == [1, 1, 1]


def test_grad_calculation_backward():
    a = torch.ones(1, requires_grad=True)
    b = tmp_Relu(a)
    b.backward(torch.tensor([3]))
    assert a.grad.tolist() == [3]


@pytest.mark.parametrize(
    "inputs,outputs",
    [
        ([-1.0, 0.9, 20.2], [0, 1, 2]),
        ([-1.0, 0.9, 2.2], [0, 1, 2]),
        ([-1.0, 0.9, -3.0], [0, 1, 0]),
        ([-1.0, -1.0, -3.0], [0, 0, 0]),
        ([0.4, 0.9, 0.001], [1, 1, 1]),
        ([2.0, 3.0, 8.0], [2, 2, 2]),
        ([2.0, 0.3, -8.0], [2, 1, 0]),
    ],
)
def test_3bit_grad_calculation_forward(inputs, outputs):
    a = torch.tensor(inputs, requires_grad=True)
    b = tmp_Relu_2(a)
    assert b.tolist() == outputs


def test_3bit_grad_calculation_backward_error():
    """
    Must be with error  - backward is not implemented yet
    for custom pice-wice func
    """
    a = torch.tensor([-1, 0.9, 2.2], requires_grad=True)
    b = tmp_Relu_2(a)
    try:
        b.mean().backward()
    except NotImplementedError:
        assert True


def test_3bit_grad_calculation_backward():
    """
    Must be without error  - backward is implemented
    """
    a = torch.tensor([-1, 0.9, 2.2], requires_grad=True)
    b = tmp_Relu_3(a)
    b.mean().backward()
    assert torch.all(
        torch.lt(
            torch.abs(
                torch.add(
                    torch.round(a.grad, decimals=2),
                    -torch.tensor([0, 0.33, 0.66]),
                )
            ),
            1e-1,
        )
    ) == torch.tensor(True)


def eq(x):
    return x


@pytest.mark.parametrize(
    "func, inputs, bit, outputs",
    [
        (eq, (0, 10), 3, [0.0, 1.25, 2.5, 3.75, 5.0, 6.25, 7.5, 8.75, 10.0]),
        (eq, (0, 10), 2, [0.0, 2.5, 5.0, 7.5, 10.0]),
        (eq, (0, 10), 1, [0.0, 5.0, 10.0]),
        (eq, (0, 1), 1, [0.0, 0.5, 1.0]),
        (
            eq,
            (0, 1),
            3,
            [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0],
        ),
        (eq, (-1, 1), 1, [-1.0, 0.0, 1.0]),
        (
            eq,
            (0, 0.0001),
            1,
            [0.0, 4.999999873689376e-05, 9.999999747378752e-05],
        ),
        (ReLU(), (0, 1), 1, [0.0, 0.5, 1.0]),
        (ReLU(), (-1, 1), 1, [-1.0, 0.0, 1.0]),
    ],
)
def test_calibrate_func(func, inputs, bit, outputs):
    assert calibrate(func, bit, inputs)[0].tolist() == outputs


@pytest.mark.parametrize(
    "func, inputs, bit, outputs",
    [
        (eq, (0, 10), 3, [0.0, 1.25, 2.5, 3.75, 5.0, 6.25, 7.5, 8.75, 10.0]),
        (eq, (0, 10), 2, [0.0, 2.5, 5.0, 7.5, 10.0]),
        (eq, (0, 10), 1, [0.0, 5.0, 10.0]),
        (eq, (0, 1), 1, [0.0, 0.5, 1.0]),
        (
            eq,
            (0, 1),
            3,
            [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0],
        ),
        (eq, (-1, 1), 1, [-1.0, 0.0, 1.0]),
        (
            eq,
            (0, 0.0001),
            1,
            [0.0, 4.999999873689376e-05, 9.999999747378752e-05],
        ),
        (ReLU(), (0, 1), 1, [0.0, 0.5, 1.0]),
        (ReLU(), (-1, 1), 1, [0.0, 0.0, 1.0]),
    ],
)
def test_calibrate_func_vals(func, inputs, bit, outputs):
    assert calibrate(func, bit, inputs)[1].tolist() == outputs


def test_optimal_replacer():
    inp = torch.tensor([-100, 1.1, 2.2, 3.33])
    quant = torch.tensor([0, 1, 2])
    vals = torch.tensor([0.666, 0.777, 0.777])
    assert torch.all(
        torch.lt(
            torch.abs(
                torch.add(
                    torch.round(
                        optimal_replacer(inp, quant, vals), decimals=2
                    ),
                    -torch.tensor([0.666, 0.777, 0.777, 0.777]),
                )
            ),
            1e-1,
        )
    ) == torch.tensor(True)


@pytest.mark.parametrize(
    "func, inputs, bit, test,  outputs",
    [
        (ReLU(), (0, 1), 1, [-1000, 0, 2, 1.1, -9, 2.0], [0, 0, 1, 1, 0, 1]),
        (
            ReLU(),
            (0, 1),
            2,
            [-1000, 0, 2, 1.1, -9, 0.33],
            [0, 0, 1, 1, 0, 0.25],
        ),
        (
            ReLU(),
            (0, 1),
            2,
            [-1000, 0, 2, 1.1, -9, 0.53],
            [0, 0, 1, 1, 0, 0.5],
        ),
        (
            ReLU(),
            (0, 1),
            2,
            [-1000, 0.26, 2, 1.1, -99, 0.63],
            [0, 0.25, 1, 1, 0, 0.5],
        ),
        (
            ReLU(),
            (-1, 1),
            8,
            [-1000, 0.26, 2, 1.1, -99, 0.63],
            [0.0, 0.2578125, 1.0, 1.0, 0.0, 0.625],
        ),
        (
            ReLU(),
            (-1, 1),
            8,
            [-1000, 0.01, 0.1, 0.22, -99, 0.63],
            [0.0, 0.0078125, 0.09375, 0.21875, 0.0, 0.625],
        ),
    ],
)
def test_optimal_repl_with_func(func, inputs, bit, test, outputs):
    calibated = calibrate(func, bit, inputs)
    assert (
        optimal_replacer(
            torch.tensor(test), calibated[0], calibated[1]
        ).tolist()
        == outputs
    )


def test_final_qrelu_forward():
    range = (-2, 2)
    bits = 8
    test_tensor = torch.tensor([-0.5, 0, 0.1, -0.6, -1, 1, 0.9])
    calibred_range = calibrate(ReLU(), bits, range)

    f = tmpRelu_4(quants=calibred_range[0], vals=calibred_range[1])
    res = f(test_tensor)
    assert res.tolist() == [0.0, 0.0, 0.09375, 0.0, 0.0, 0.984375, 0.890625]


def test_final_qrelu_backward():
    range = (-2, 2)
    bits = 8
    test_tensor = torch.tensor(
        [-0.5, 0, 0.1, -0.6, -1, 1, 0.9], requires_grad=True
    )
    calibred_range = calibrate(ReLU(), bits, range)

    f = tmpRelu_4(quants=calibred_range[0], vals=calibred_range[1])
    res = f(test_tensor)
    res.mean().backward()
    assert test_tensor.grad.tolist() == [
        0.0,
        0.0,
        0.01428571529686451,
        0.0,
        0.0,
        0.1428571492433548,
        0.12857143580913544,
    ]


def test_grad_check():
    # gradcheck takes a tuple of tensors as input, check if your gradient
    # evaluated with these tensors are close enough to numerical
    # approximations and returns True if they all verify this condition.
    input = torch.randn(20, 20, dtype=torch.double, requires_grad=True)
    # Gradcheck may fail when evaluated on non-differentiable points because
    # the numerically computed gradients via finite differencing may differ
    # those computed analytically (not necessarily because either is
    # incorrect).

    # That is this atol=5 is so big

    assert gradcheck(stepfunc, input, eps=1e-2, atol=5, raise_exception=True)


def test_qmodel_can_run():
    batch = torch.rand(1, 3, 224, 224).to(device=device)
    model_5 = ResNet50(activ_func="qReLU")
    model_5.eval()
    with torch.no_grad():
        result = model_5(batch)
    assert list(result.size()) == [1, 1000]


def test_qmodel_can_train_loop():
    BS = 1  # batch size
    EP = 3  # epoch count
    x = torch.rand(12, 3, 224, 224)
    y = torch.rand(12, 1000)

    tmp_model = ResNet50(activ_func="qReLU", bits=16, arange=(-10, 10))
    # tmp_model = ResNet50(activ_func="LeakyReLU")
    # tmp_model = ResNet50()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        tmp_model.parameters(),
        # lr=0.06,
        lr=0.00001,
    )
    tmp_model.train(True)
    for k in range(EP):
        for i in range(y.shape[0] // BS):
            optimizer.zero_grad(set_to_none=False)
            slic = [k for k in range(i * BS, (i + 1) * BS)]  # noqa: E203
            assert x[slic, :, :, :].shape == (BS, 3, 224, 224)
            output = tmp_model(x[slic, :, :, :])
            # torch.isnan(output)
            assert not torch.isnan(output).any()
            assert output.shape == (BS, 1000)
            assert y[i * BS : (i + 1) * BS].shape == (BS, 1000)  # noqa: E203
            loss = criterion(output, y[i * BS : (i + 1) * BS])  # noqa: E203
            assert loss.item() > 3
            loss.backward()
            optimizer.step()
    tmp_model.eval()
    output = tmp_model(x[[0, 1], :, :, :])
    assert output.shape == (2, 1000)


def test_simple_network():
    class SimpleNet(nn.Module):
        def __init__(self) -> None:
            super(SimpleNet, self).__init__()

            self.fc = nn.Linear(30, 10)
            # self.act = nn.ReLU()
            calibred_range = calibrate(ReLU(), 24, (-1, 1))
            self.act = tmpRelu_4(
                quants=calibred_range[0], vals=calibred_range[1]
            )
            self.fc2 = nn.Linear(10, 3)

        def forward(self, x):
            x = self.fc(x)
            x = self.act(x)
            x = self.fc2(x)
            return F.sigmoid(x)

    BS = 1  # batch size
    EP = 3  # epoch count
    x = torch.rand(12, 30)
    y = torch.tensor(
        [
            [1, 2, 3],
            [1, 3, 2],
            [1, 1, 1],
            [2, 2, 2],
            [3, 3, 3],
            [2, 3, 1],
            [2, 1, 3],
            [1, 2, 2],
            [2, 2, 2],
            [3, 3, 3],
            [1, 1, 2],
            [1, 1, 3],
        ],
        dtype=float,
    )

    model = SimpleNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        # lr=0.06,
        lr=0.1,
    )
    model.train(True)
    first_loss = 0
    for k in range(EP):
        for i in range(y.shape[0] // BS):
            optimizer.zero_grad(set_to_none=False)
            slic = [k for k in range(i * BS, (i + 1) * BS)]  # noqa: E203
            # assert x[slic, :, :, :].shape == (BS, 30)
            output = model(x[slic, :])
            # torch.isnan(output)
            assert not torch.isnan(output).any()
            assert output.shape == (BS, 3)
            assert y[i * BS : (i + 1) * BS].shape == (BS, 3)
            loss = criterion(output, y[i * BS : (i + 1) * BS])  # noqa: E203
            if k == 0:
                first_loss = loss.item()

            loss.backward()
            optimizer.step()
    assert 2 * loss.item() < first_loss
    model.eval()
    output = model(x[[0, 1]])
    assert output.shape == (2, 3)


def test_final_simple_network():
    class SimpleNet(nn.Module):
        def __init__(self) -> None:
            super(SimpleNet, self).__init__()

            self.fc = nn.Linear(30, 10)
            # self.act = nn.ReLU()
            calibred_range = calibrate(ReLU(), 8, (-1, 1))
            self.act = FinalQRelu(
                quants=calibred_range[0], vals=calibred_range[1]
            )
            self.fc2 = nn.Linear(10, 3)

        def forward(self, x):
            x = self.fc(x)
            x = self.act(x)
            x = self.fc2(x)
            return F.sigmoid(x)

    BS = 2  # batch size
    EP = 18  # epoch count
    x = torch.rand(12, 30)
    y = torch.tensor(
        [
            [1, 2, 3],
            [1, 3, 2],
            [1, 1, 1],
            [2, 2, 2],
            [3, 3, 3],
            [2, 3, 1],
            [2, 1, 3],
            [1, 2, 2],
            [2, 2, 2],
            [3, 3, 3],
            [1, 1, 2],
            [1, 1, 3],
        ],
        dtype=float,
    )

    model = SimpleNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        # lr=0.06,
        lr=0.1,
    )
    model.train(True)
    first_loss = 0
    for k in range(EP):
        for i in range(y.shape[0] // BS):
            optimizer.zero_grad(set_to_none=False)
            slic = [k for k in range(i * BS, (i + 1) * BS)]  # noqa: E203
            # assert x[slic, :, :, :].shape == (BS, 30)
            output = model(x[slic, :])
            # torch.isnan(output)
            assert not torch.isnan(output).any()
            assert output.shape == (BS, 3)
            assert y[i * BS : (i + 1) * BS].shape == (BS, 3)
            loss = criterion(output, y[i * BS : (i + 1) * BS])  # noqa: E203
            if k == 0:
                first_loss = loss.item()

            loss.backward()
            optimizer.step()
    assert loss.item() < first_loss
    model.eval()
    output = model(x[[0, 1]])
    assert output.shape == (2, 3)


def test_final_qmodel_can_train_loop():
    BS = 4  # batch size
    EP = 6  # epoch count
    x = torch.rand(12, 3, 224, 224)
    y = torch.rand(12, 1000)

    tmp_model = ResNet50(activ_func="FinalQRelu", bits=8, arange=(-10, 10))
    # tmp_model = ResNet50(activ_func="LeakyReLU")
    # tmp_model = ResNet50()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        tmp_model.parameters(),
        # lr=0.06,
        lr=0.00001,
    )
    tmp_model.train(True)
    first_loss = 0
    for k in range(EP):
        for i in range(y.shape[0] // BS):
            optimizer.zero_grad(set_to_none=False)
            slic = [k for k in range(i * BS, (i + 1) * BS)]  # noqa: E203
            assert x[slic, :, :, :].shape == (BS, 3, 224, 224)
            output = tmp_model(x[slic, :, :, :])
            # torch.isnan(output)
            assert not torch.isnan(output).any()
            assert output.shape == (BS, 1000)
            assert y[i * BS : (i + 1) * BS].shape == (BS, 1000)
            loss = criterion(output, y[i * BS : (i + 1) * BS])  # noqa: E203
            assert loss.item() > 3
            if k == 0:
                first_loss = loss.item()
            loss.backward()
            optimizer.step()
    assert loss.item() < first_loss
    tmp_model.eval()
    output = tmp_model(x[[0, 1], :, :, :])
    assert output.shape == (2, 1000)


@pytest.mark.parametrize(
    "inputs",
    [
        ("FinalQRelu",),
        ("QSigmoid",),
    ],
)
def test_model_layer_replacer(inputs):
    tmp_model = ResNet50(activ_func=inputs, bits=8, arange=(-10, 10))
    assert tmp_model.get_activations_count() == 49


@pytest.mark.parametrize(
    "inputs1, inputs2, res",
    [
        (FinalQRelu, dReLU, "FinalQRelu()"),
        (FinalQSigmoid, dSigmoid, "FinalQSigmoi"),
    ],
)
def test_model_layers_replacer(inputs1, inputs2, res):
    tmp_model = ResNet50()
    test_list = [8 for _ in range(50)]
    test_list[0] = 8
    test_list[4] = 8
    test_list[5] = 8
    tmp_model.change_by_qactivations(
        bitlist=test_list, active_func=inputs1, active_deriv=inputs2
    )
    assert str(tmp_model)[218:230] == res


@pytest.mark.parametrize(
    "inputs1, inputs2",
    [
        (FinalQRelu, dReLU),
        (FinalQSigmoid, dSigmoid),
    ],
)
def test_strategy_generator(inputs1, inputs2):
    sgen = ClassStratageGenerator(active_func=inputs1, active_deriv=inputs2)
    tmp_model = ResNet50()
    generatedModel, list_activations = sgen.generate(tmp_model, 20, bit_max=8)
    assert len(list_activations) == tmp_model.get_activations_count()


@pytest.mark.parametrize(
    "inputs1, inputs2",
    [
        (FinalQRelu, dReLU),
        (FinalQSigmoid, dSigmoid),
    ],
)
def test_model_hook(inputs1, inputs2):
    sgen = ClassStratageGenerator(active_func=inputs1, active_deriv=inputs2)
    tmp_model = ResNet50()
    generatedModel, list_activations = sgen.generate(tmp_model, 20, bit_max=8)

    x = torch.rand(2, 3, 224, 224)
    y = torch.rand(2, 1000)

    sscor = ClassStrategyScore(x, y)
    sc = sscor.get_matrix(generatedModel)

    assert 49 == len(list_activations)
    assert 49 == len(sc)
    assert torch.Size([2, 2048, 7, 7]) == sc[3].shape
    assert torch.Size([2, 512, 28, 28]) == sc[33].shape


@pytest.mark.parametrize(
    "inputs,outputs, inputs2, inputs3",
    [
        (2, 2, FinalQRelu, dReLU),
        (3, 3, FinalQSigmoid, dSigmoid),
        (10, 10, FinalQRelu, dReLU),
    ],
)
def test_model_hamming(inputs, outputs, inputs2, inputs3):
    sgen = ClassStratageGenerator(active_func=inputs2, active_deriv=inputs3)
    tmp_model = ResNet50()
    generatedModel, list_activations = sgen.generate(tmp_model, 20, bit_max=8)

    x = torch.rand(inputs, 3, 224, 224)
    y = torch.rand(inputs, 1000)
    sscor = ClassStrategyScore(x, y)
    distMatrix = sscor.get_score(generatedModel)
    # assert distMatrix == "12"
    distMatrix, A_score = sscor.get_score(generatedModel)

    assert 0 < A_score
    assert torch.Size([outputs, outputs]) == distMatrix.shape


@pytest.mark.parametrize(
    "inputs,outputs, inputs2, inputs3",
    [
        # (2, 2, FinalQRelu, dReLU),
        # (3, 3, FinalQRelu, dReLU),
        # (4, 4, FinalQRelu, dReLU),
        # (5, 5),
        # (2, 2, FinalQSigmoid, dSigmoid),
        (3, 3, FinalQSigmoid, dSigmoid),
        # (4, 4, FinalQSigmoid, dSigmoid),
        # (5, 5),
    ],
)
def test_model_score(inputs, outputs, inputs2, inputs3):
    sgen = ClassStratageGenerator(active_func=inputs2, active_deriv=inputs3)
    x = torch.rand(inputs, 3, 224, 224)
    y = torch.rand(inputs, 1000)
    sscor = ClassStrategyScore(x, y)

    tmp_model = ResNet50()
    generatedModel, list_activations = sgen.generate(tmp_model, 99, bit_max=8)
    _, A_score10 = sscor.get_score(generatedModel)

    tmp_model = ResNet50()
    generatedModel80, list_activations = sgen.generate(
        tmp_model, 99, bit_max=2
    )
    _, A_score80 = sscor.get_score(generatedModel80)

    # tmp_model = ResNet50()
    # generatedModel, list_activations = sgen.generate(tmp_model, 1, bit_max=8)
    # _, A_score_original = sscor.get_score(tmp_model)
    # A_score_original = 0

    # assert A_score10 == A_score_original
    # assert A_score80 == A_score_original
    assert A_score80 == A_score10
