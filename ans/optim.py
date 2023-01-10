import numpy as np
import torch

from ans.autograd import Variable


class Optimizer:

    def __init__(self, parameters: list[Variable]) -> None:
        self.parameters = parameters

    def step(self) -> None:
        raise NotImplementedError

    def zero_grad(self) -> None:
        for param in self.parameters:
            param.grad = None


class SGD(Optimizer):

    def __init__(
            self,
            parameters: list[Variable],
            learning_rate: float = 1e-3,
            momentum: float = 0.,
            weight_decay: float = 0.
    ) -> None:
        super().__init__(parameters)

        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay

        ########################################

        self._velocities: dict[Variable, torch.Tensor] = \
            dict(zip(parameters, len(parameters) * [torch.zeros(1).to(dtype=torch.float16)]))

        ########################################

    def step(self) -> None:
        ########################################
        for par in self.parameters:
            if par.grad is not None:
                theta = par.grad + (self.weight_decay * par.data)
                self._velocities[par] = (self.momentum * self._velocities[par]) - self.learning_rate * theta
                par.data += self._velocities[par]

        ########################################


class Adam(Optimizer):

    def __init__(
            self,
            parameters: list[Variable],
            learning_rate: float = 1e-3,
            beta1: float = 0.9,
            beta2: float = 0.999,
            eps: float = 1e-08,
            weight_decay: float = 0.,
    ) -> None:
        super().__init__(parameters)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        ########################################
        device = device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self._num_steps = torch.zeros(1).to(device= device)

        self._m: dict[Variable, torch.Tensor] = dict(zip(parameters, len(parameters) *
                                                         [torch.zeros(1).to(dtype=torch.float16,device= device)]))
        self._v: dict[Variable, torch.Tensor] = dict(zip(parameters, len(parameters) *
                                                         [torch.zeros(1).to(dtype=torch.float16,device= device)]))

        ########################################

    def step(self) -> None:
        ########################################
        self._num_steps += 1
        for par in self.parameters:
            if par.grad is not None:
                theta = par.grad + (self.weight_decay * par.data)
                self._m[par] = (self.beta1 * self._m[par]) + (1 - self.beta1) * theta
                self._v[par] = (self.beta2 * self._v[par]) + (1 - self.beta2) * (theta ** 2)
                mt = self._m[par] / (1 - self.beta1**self._num_steps)
                vt = self._v[par] / (1 - self.beta2**self._num_steps)

                par.data += -self.learning_rate * mt / (torch.sqrt(vt) + self.eps)
        ########################################
