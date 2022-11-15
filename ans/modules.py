from typing import Any, Optional

import torch

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
        # TODO: initialize weight and bias

        self.weight = ...
        self.bias = ...

        # ENDTODO
        ########################################

    def forward(self, x: Variable) -> Variable:
        ########################################
        # TODO: implement

        ...

        # ENDTODO
        ########################################


class Sigmoid(Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Variable) -> Variable:
        ########################################
        # TODO: implement

        ...

        # ENDTODO
        ########################################


class SoftmaxCrossEntropy(Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Variable, y: torch.Tensor) -> Variable:
        ########################################
        # TODO: implement

        ...

        # ENDTODO
        ########################################


class ReLU(Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Variable) -> Variable:
        ########################################
        # TODO: implement

        raise NotImplementedError

        # ENDTODO
        ########################################


class Dropout(Module):

    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: Variable) -> Variable:
        ########################################
        # TODO: implement

        raise NotImplementedError

        # ENDTODO
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
        # TODO: initialize gamma, beta, running_mean, running_var
        # if affine, then gamma to ones (learnable), otherwise set to None
        # if affine, then beta to zeros (learnable), otherwise set to None
        # running_mean to zeros
        # running_var to ones

        self.gamma = ...
        self.beta = ...
        self.running_mean = ...
        self.running_var = ...

        # ENDTODO
        ########################################

    def forward(self, x: Variable) -> Variable:
        ########################################
        # TODO: implement

        raise NotImplementedError

        # ENDTODO
        ########################################
