# import torch
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn.modules import Module


class tmpRelu(Function):
    """
    Implemetation of quantizied function for test and correct
    """

    @staticmethod
    def forward(ctx, input):
        real = torch.maximum(torch.zeros_like(input), input)
        ctx.save_for_backward(real)
        return real

    @staticmethod
    def backward(ctx, grad_output):
        (result,) = ctx.saved_tensors
        return torch.mul(result, grad_output)


def tmp_Relu(x):
    return tmpRelu.apply(x)


class tmpRelu_2(Function):
    """
    Implemetation of quantizied function for test
    and correct (without backward)
    """

    @staticmethod
    def forward(ctx, input):
        real = torch.maximum(torch.zeros_like(input), input)

        y = torch.mul(2, torch.ones_like(input))
        real = torch.where(real > 1, y, real)

        y = torch.ones_like(input)
        real = torch.where((real > 0) & (real <= 1), y, real)

        ctx.save_for_backward(real)
        return real


def tmp_Relu_2(x):
    return tmpRelu_2.apply(x)


class tmpRelu_3(Function):
    """
    Implemetation of quantizied function for test
    and correct (with backward)
    """

    @staticmethod
    def forward(ctx, input):
        real = torch.maximum(torch.zeros_like(input), input)

        y = torch.mul(2, torch.ones_like(input))
        real = torch.where(real > 1, y, real)

        y = torch.ones_like(input)
        real = torch.where((real > 0) & (real <= 1), y, real)

        ctx.save_for_backward(real)
        return real

    @staticmethod
    def backward(ctx, grad_output):

        (result,) = ctx.saved_tensors

        # return torch.mul(result, torch.mul(grad_output, 2))
        return torch.mul(result, grad_output)


def tmp_Relu_3(x):
    return tmpRelu_3.apply(x)


def calibrate(
    func: torch.nn = torch.nn.ReLU,
    bits: int = 8,
    ranges: tuple = (0, 1),
    iven: bool = False,  # TODO
) -> torch.Tensor:
    """
    Return pice-wise tensor range that approximate given
    function use uniform distribution from -inf to +inf

    last value not mater

    Args:
        func: function from torch.nn
        bits: int how many bits is to store result. Min bits
                is 1. It can store 1 value x and 2 intervals:
                [-inf, x],(x. +inf]
        ranges: (int, int) min and max intervals for generate
                ranges for approximate
        iven: bool If func to approximate is iven it can
                helps to save memory by storin only half inteval

    Returns:
        Tuple of approximated points by X and Y
    """

    if not bits >= 1:
        raise ValueError("Num of bits must be equal or more 1")

    if ranges[1] <= ranges[0]:
        raise ValueError(
            "The right border of the interval must be greater \
                than the left one"
        )

    steps = 2**bits

    s = (ranges[1] - ranges[0]) / steps
    range_val = [ranges[0]]
    for _ in range(steps - 1):
        range_val.append(range_val[len(range_val) - 1] + s)
    range_val.append(ranges[1])
    return (
        torch.tensor(range_val, requires_grad=False),
        func(torch.tensor(range_val, requires_grad=False)),
    )


def optimal_replacer(input, quants, vals):
    """
    This code must be replaced by fast implementation on c++

    First and last value is ignoring - inf
    """
    for i in range(quants.shape[0]):
        min = -10000 if i == 0 else quants[i]
        max = 10000 if i >= quants.shape[0] - 1 else quants[i + 1]

        input = torch.where((input > min) & (input <= max), vals[i], input)
    return input  # .to(torch.float64)


class qRelu(Module):
    """
    Implemetation of quantizied function
    """

    def __init__(
        self, quants=torch.tensor([0]), vals=torch.tensor([0])
    ) -> None:
        super(qRelu, self).__init__()
        self._quants = quants
        self._vals = vals

    @staticmethod
    def forward(input):
        b = nn.ReLU()
        c = b(input)
        return c


class StepFunc(Function):
    @staticmethod
    def forward(input, quants, vals):

        # ctx.mark_non_differentiable(quants, vals)
        with torch.no_grad():
            r = optimal_replacer(input, quants, vals)
        return r

    @staticmethod
    def setup_context(ctx, inputs, output):
        input, _, _ = inputs
        rl = nn.ReLU()
        real = rl(input)
        ctx.save_for_backward(real)

    @staticmethod
    def backward(ctx, grad_output):
        # input, weight, bias = ctx.saved_tensors
        # grad_input = grad_weight = grad_bias = None

        # if ctx.needs_input_grad[0]:
        #     grad_input = grad_output.mm(weight)
        # if ctx.needs_input_grad[1]:
        #     grad_weight = grad_output.t().mm(input)
        # if bias is not None and ctx.needs_input_grad[2]:
        #     grad_bias = grad_output.sum(0)
        result = ctx.saved_tensors[0]

        return (
            # torch.zeros_like(result),
            torch.mul(result, grad_output),
            # quants, vals
            None,
            None,
        )  # grad_input , grad_weight, grad_bias


def stepfunc(input, quants=None, vals=None):
    if quants is None:
        with torch.no_grad():
            calibred_range = calibrate(nn.ReLU(), 8, (-1, 1))
        return StepFunc.apply(input, calibred_range[0], calibred_range[1])
    return StepFunc.apply(input, quants, vals)


class tmpRelu_4(nn.Module):
    """
    Implemetation of quantizied function for test
    and correct (with backward)
    """

    def __init__(
        self, quants=torch.tensor([0]), vals=torch.tensor([0])
    ) -> None:
        super().__init__()
        self._quants = quants
        self._vals = vals
        # self._stepfunc = ( self._quants, self._vals)

    # @staticmethod
    def forward(self, input):
        # sf = StepFunc()
        return stepfunc(input, self._quants, self._vals)
        # return StepFunc.apply(input, self._quants, self._vals)

    # @staticmethod
    # def backward(in_gr):
    #     return in_gr


# --------------DERIVIATIVES FOR USE IN NETS ON BACKWARD


def dReLU(x):
    return 1.0 * (x > 0)


def dSigmoid(x):
    return torch.sigmoid(x) * (1 - torch.sigmoid(x))


# --------------FINAL VERSION OF STEP RELU
class FinalStepFunc(Function):
    @staticmethod
    def forward(input, quants, vals):
        rl = nn.ReLU()
        return rl(input)

    @staticmethod
    def setup_context(ctx, inputs, output):
        input, quants, vals = inputs
        with torch.no_grad():
            r = optimal_replacer(input, quants, vals)
        ctx.save_for_backward(r)

    @staticmethod
    def backward(ctx, grad_output):
        r = ctx.saved_tensors[0]

        return (
            torch.mul(r, grad_output),
            None,
            None,
        )


def finalstepfunc(input, quants=None, vals=None):
    """
    RELU
    """
    if quants is None:
        with torch.no_grad():
            calibred_range = calibrate(nn.ReLU(), 8, (-1, 1))
        return FinalStepFunc.apply(input, calibred_range[0], calibred_range[1])
    return FinalStepFunc.apply(input, quants, vals)


class FinalQRelu(nn.Module):
    """
    Implemetation qRELU of quantizied function for test
    and correct (with backward)
    """

    def __init__(
        self, quants=torch.tensor([0]), vals=torch.tensor([0])
    ) -> None:
        super().__init__()
        self._quants = quants
        self._vals = vals

    def forward(self, input):
        return finalstepfunc(input, self._quants, self._vals)


# --------------FINAL VERSION OF STEP SIGMOID
class FinalStepFuncSigmoid(Function):
    @staticmethod
    def forward(input, quants, vals):
        return torch.sigmoid(input)

    @staticmethod
    def setup_context(ctx, inputs, output):
        input, quants, vals = inputs
        with torch.no_grad():
            r = optimal_replacer(input, quants, vals)
        ctx.save_for_backward(r)

    @staticmethod
    def backward(ctx, grad_output):
        r = ctx.saved_tensors[0]

        return (
            torch.mul(r, grad_output),
            None,
            None,
        )


def finalstepfuncSigmoid(input, quants=None, vals=None):
    """
    Вспомогательная функции Sigmoid
    """
    if quants is None:
        with torch.no_grad():
            calibred_range = calibrate(torch.sigmoid(), 8, (-1, 1))
        return FinalStepFuncSigmoid.apply(
            input, calibred_range[0], calibred_range[1]
        )
    return FinalStepFuncSigmoid.apply(input, quants, vals)


class FinalQSigmoid(nn.Module):
    """
    Квантизованная версия функции Sigmoid
    Implemetation qSigmoid
    of quantizied function for test
    and correct (with backward)
    """

    def __init__(
        self, quants=torch.tensor([0]), vals=torch.tensor([0])
    ) -> None:
        super().__init__()
        self._quants = quants
        self._vals = vals

    def forward(self, input):
        return finalstepfuncSigmoid(input, self._quants, self._vals)
