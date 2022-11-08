import inspect
from typing import Optional

import torch

from ans.autograd import Variable
import ans.functional


class Module:

    def __call__(self, *x: Variable) -> Variable:
        return self.forward(*x)

    def forward(self, *x: Variable) -> Variable:
        raise NotImplementedError

    def parameters(self) -> list[Variable]:
        return [p for n, p in self.named_parameters()]

    def named_parameters(self) -> list[tuple[str, Variable]]:
        named_parameters = []

        def depth_first_append(layer, prefix=''):
            for name in dir(layer):
                attr = getattr(layer, name)
                if isinstance(attr, (list, tuple)):
                    for i, l in enumerate(attr):
                        depth_first_append(l, prefix=f"{prefix}{i}.")
                elif isinstance(attr, Module):
                    for n, p in layer.named_parameters():
                        named_parameters.append((f"{prefix}{name}.{n}"))
                elif isinstance(attr, Variable):
                    named_parameters.append((f"{prefix}{name}", attr))

        depth_first_append(self)
        return named_parameters

    def to(self, dtype: Optional[torch.dtype] = None, device: Optional[str] = None) -> 'Module':
        for name, par in self.named_parameters():
            par.data = par.data.to(dtype=dtype, device=device)
        return self

    def zero_grad(self) -> None:
        for name, par in self.named_parameters():
            par.grad = None


class Linear(Module):

    def __init__(self, num_in: int, num_out: int) -> None:
        ########################################
        # TODO: implement

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

    def forward(self, x: Variable) -> Variable:
        ########################################
        # TODO: implement

        ...

        # ENDTODO
        ########################################


class SoftmaxCrossEntropy(Module):

    def forward(self, x: Variable, y: torch.Tensor) -> Variable:
        ########################################
        # TODO: implement

        ...

        # ENDTODO
        ########################################


class Sequential(Module):

    def __init__(self, *layers: Module) -> None:
        self.layers = layers

    def forward(self, x: Variable) -> Variable:
        for layer in self.layers:
            x = layer(x)
        return x
