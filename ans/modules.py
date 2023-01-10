from typing import Any, Optional

import torch
import math
from ans.autograd import Variable
import ans.functional


class Module:

    def __init__(self) -> None:
        self.training = True

    def __call__(self, *x: Variable) -> Variable:
        return self.forward(*x)

    def device(self) -> torch.device:
        return next(iter(self.parameters())).data.device

    def dtype(self) -> torch.dtype:
        return next(iter(self.parameters())).data.dtype

    def forward(self, *x: Variable) -> Variable:
        raise NotImplementedError

    def named_modules(self) -> list[tuple[str, 'Module']]:
        named_modules = []

        def depth_first_append(obj, prefix=''):
            if isinstance(obj, Module):
                named_modules.append((prefix, obj))
                for name in dir(obj):
                    attr = getattr(obj, name)
                    if isinstance(attr, (list, tuple)):
                        for i, item in enumerate(attr):
                            depth_first_append(item, prefix=f"{prefix}.{i}" if prefix else str(i))
                    else:
                        depth_first_append(attr, prefix=f"{prefix}.{name}" if prefix else name)

        depth_first_append(self)
        return named_modules

    def named_parameters(self) -> list[tuple[str, Variable]]:
        return [
            (f"{name + '.' if name else ''}{attr}", getattr(module, attr))
            for name, module in self.named_modules()
            for attr in dir(module)
            if isinstance(getattr(module, attr), Variable)
        ]

    def parameters(self) -> list[Variable]:
        return [p for n, p in self.named_parameters()]

    def num_params(self) -> int:
        return sum(p.data.numel() for p in self.parameters())

    def to(self, dtype: Optional[torch.dtype] = None, device: Optional[str] = None) -> 'Module':
        def to(obj: Any) -> None:
            if isinstance(obj, torch.Tensor):
                obj.data = obj.to(dtype=dtype, device=device)
            elif isinstance(obj, (tuple, list)):
                for elem in obj:
                    to(elem)
            elif isinstance(obj, dict):
                for val in obj.values():
                    to(val)
            elif isinstance(obj, Variable):
                to(obj.data)
                to(obj.grad)
            elif isinstance(obj, Module):
                for attr in dir(obj):
                    to(getattr(obj, attr))

        to(self)
        return self

    def train(self) -> None:
        for name, layer in self.named_modules():
            layer.training = True

    def eval(self) -> None:
        for name, layer in self.named_modules():
            layer.training = False

    def zero_grad(self) -> None:
        for name, par in self.named_parameters():
            par.grad = None


class Linear(Module):

    def __init__(self, num_in: int, num_out: int) -> None:
        super().__init__()
        ########################################
        i = 1 / math.sqrt(num_in)
        self.weight = Variable((2 * i) * torch.rand(num_in, num_out) - i)
        self.bias = Variable(torch.zeros(num_out))
        ########################################

    def forward(self, x: Variable) -> Variable:
        ########################################
        return ans.functional.Linear.apply(x, self.weight, self.bias)

        ########################################


class Sigmoid(Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Variable) -> Variable:
        ########################################

        return ans.functional.Sigmoid.apply(x)

        ########################################


class SoftmaxCrossEntropy(Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Variable, y: torch.Tensor) -> Variable:
        ########################################
        return ans.functional.SoftmaxCrossEntropy.apply(x, y)
        ########################################


class ReLU(Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Variable) -> Variable:
        ########################################

        return ans.functional.ReLU.apply(x)

        ########################################


class Dropout(Module):

    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: Variable) -> Variable:
        ########################################

        return ans.functional.Dropout().apply(x, p=self.p, training=self.training)

        ########################################


class Tanh(Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Variable) -> Variable:
        ########################################
        # TODO: implement

        raise NotImplementedError

        # ENDTODO
        ########################################


class Sequential(Module):

    def __init__(self, *layers: Module) -> None:
        super().__init__()
        self.layers = layers

    def forward(self, x: Variable) -> Variable:
        for layer in self.layers:
            x = layer(x)
        return x


class BatchNorm1d(Module):

    def __init__(
            self,
            num_features: int,
            momentum: float = 0.9,
            eps: float = 1e-5,
            affine: bool = True
    ) -> None:
        super().__init__()

        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        self.affine = affine

        ########################################

        self.gamma = torch.ones(num_features) if affine else None
        self.beta = torch.zeros(num_features) if affine else None
        self.running_mean = torch.zeros(num_features)
        self.running_var = torch.ones(num_features)

        ########################################

    def forward(self, x: Variable) -> Variable:
        ########################################

        return ans.functional.BatchNorm1d().apply(x,
                                                  self.gamma,
                                                  self.beta,
                                                  self.running_mean,
                                                  self.running_var,
                                                  momentum=self.momentum,
                                                  eps=self.eps,
                                                  training=self.training
                                                  )

        ########################################


class BatchNorm2d(BatchNorm1d):

    def forward(self, x: Variable) -> Variable:
        ########################################
        return ans.functional.BatchNorm2d().apply(x,
                                                  self.gamma,
                                                  self.beta,
                                                  self.running_mean,
                                                  self.running_var,
                                                  momentum=self.momentum,
                                                  eps=self.eps,
                                                  training=self.training)
        ########################################


class Conv2d(Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            dilation: int = 1,
            groups: int = 1,
            bias: bool = True
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        ########################################
        # if bias is True, then bias should be zeros, otherwise set to None
        i = 1 / math.sqrt(in_channels * (kernel_size ** 2))
        self.weight = Variable((2 * i) * torch.rand(out_channels, in_channels, kernel_size, kernel_size) - i)

        self.bias = Variable(torch.zeros(out_channels)) if bias else None

        ########################################

    def forward(self, x: Variable) -> Variable:
        ########################################

        return ans.functional.Conv2d().apply(x,
                                             self.weight,
                                             self.bias,
                                             stride=self.stride,
                                             padding=self.padding,
                                             dilation=self.dilation,
                                             groups=self.groups)

        ########################################


class MaxPool2d(Module):

    def __init__(self, window_size: int) -> None:
        super().__init__()

        self.window_size = window_size

    def forward(self, x: Variable) -> Variable:
        ########################################

        return ans.functional.MaxPool2d.apply(x, self.window_size)

        ########################################


class Flatten(Module):

    def forward(self, x: Variable) -> Variable:
        ########################################
        N = x.data.shape[0]
        D = int(torch.prod(torch.Tensor(list(x.data.shape[1:]))).item())  # product of other shapes
        return x.reshape(N, D)

        ########################################
